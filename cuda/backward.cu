/**
 * @file backward.cu
 * @brief Backward pass CUDA kernels for hypergraph attention.
 *
 * Computes gradients for Q, R, S, and all V tensors using online softmax
 * statistics for numerical stability. Includes Jacobian correction terms
 * for proper gradient flow through the softmax.
 *
 * Copyright (c) 2026 Springtail AI. MIT License.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <tuple>

#include "common.cuh"
#include "../cpp/cuda_bindings.h"

// Backward-specific tile aliases for gradient kernels
#ifndef T_I
#define T_I TILE_I
#endif
#ifndef T_J
#define T_J TILE_J
#endif
#ifndef T_K
#define T_K TILE_K
#endif

// =============================================================================
// NOTE: Softmax stats (m_i, l_i, m_j, l_j, m_k, l_k) are computed during the
// forward pass and passed to backward. The backward pass does NOT recompute
// these stats - this avoids redundant work.
// =============================================================================

// Both members of a pair must be visible from mask row `row`.
__device__ __forceinline__ bool mask_pair_allowed(
    const bool* mask,
    int N,
    int b,
    int row,
    int a,
    int c
) {
    if (mask == nullptr) {
        return true;
    }
    const int64_t base = ((int64_t)b * N + row) * N;
    return mask[base + a] && mask[base + c];
}

// =============================================================================
// Gradient Kernels for V Tensors (Gather Path)
// =============================================================================

// Split V_1 gradients (grad_Vq_1 / grad_Vr_1 / grad_Vs_1), one kernel with
// roles permuted: `out` is the mode whose V gradient this launch accumulates,
// `reg` its register-resident partner, `loop` the mode streamed through shared
// memory. Each output picks up one term from each of the other two Y gathers.
// OUT_IS_Y puts out on thread y, keeping the atomicAdd address warp-uniform.
template<int D_CONST, bool OUT_IS_Y>
__global__ void V_gather_grad(
    const bf16* __restrict__ X_out,     // [B,H,N,D]
    const bf16* __restrict__ X_reg,     // [B,H,N,D]
    const bf16* __restrict__ X_loop,    // [B,H,N,D]
    const bf16* __restrict__ V_reg,     // [B,H,N,D]   (V_1 slice)
    const bf16* __restrict__ V_loop,    // [B,H,N,D]   (V_1 slice)
    const bf16* __restrict__ gY_loop,   // [B,H,N,D] upstream grad for Y_loop
    const bf16* __restrict__ gY_reg,    // [B,H,N,D] upstream grad for Y_reg
    const float* __restrict__ m_loop,   // [B,H,N]
    const float* __restrict__ l_loop,   // [B,H,N]
    const float* __restrict__ m_reg,    // [B,H,N]
    const float* __restrict__ l_reg,    // [B,H,N]
    float*       __restrict__ gradV_out, // [B,H,N,D] (output)
    const bool*  __restrict__ mask,
    int N, int H, float scale)
{
    const int bh = blockIdx.z;          // flattened (batch, head)
    const int b = bh / H;
    const int tx = blockIdx.x * T_I + threadIdx.x;
    const int ty = blockIdx.y * T_K + threadIdx.y;
    const int out0 = OUT_IS_Y ? ty : tx;
    const int reg0 = OUT_IS_Y ? tx : ty;

    // No early return: every thread is needed for the cooperative loads.
    const bool active = (out0 < N && reg0 < N);

    const int64_t stride_BH = (int64_t)N * D_CONST;
    const bf16* X_outBH   = X_out   + bh * stride_BH;
    const bf16* X_regBH   = X_reg   + bh * stride_BH;
    const bf16* X_loopBH  = X_loop  + bh * stride_BH;
    const bf16* V_regBH   = V_reg   + bh * stride_BH;
    const bf16* V_loopBH  = V_loop  + bh * stride_BH;
    const bf16* gY_loopBH = gY_loop + bh * stride_BH;
    const bf16* gY_regBH  = gY_reg  + bh * stride_BH;
    const float* m_loopBH = m_loop  + (int64_t)bh * N;
    const float* l_loopBH = l_loop  + (int64_t)bh * N;
    const float* m_regBH  = m_reg   + (int64_t)bh * N;
    const float* l_regBH  = l_reg   + (int64_t)bh * N;
          float* gV_outBH = gradV_out + bh * stride_BH;

    // Fusing prod = X_out*X_reg halves the inner-loop FMAs; the reg-side rows
    // are hoisted out of the loop-mode sweep.
    float prod_vec[D_CONST];
    float v_reg_vec[D_CONST];
    float gy_reg_vec[D_CONST];
    float grad_acc[D_CONST] = {0.0f};

    // Inactive threads still load, from a clamped row.
    const int out_safe = min(out0, N-1);
    const int reg_safe = min(reg0, N-1);

    #pragma unroll
    for (int d=0; d<D_CONST; ++d){
        prod_vec[d]   = bf2f(X_outBH[out_safe*D_CONST + d]) * bf2f(X_regBH[reg_safe*D_CONST + d]);
        v_reg_vec[d]  = bf2f(V_regBH[reg_safe*D_CONST + d]);
        gy_reg_vec[d] = bf2f(gY_regBH[reg_safe*D_CONST + d]);
    }

    // Hoist reg-only stats out of the loop; pre-invert l_reg once.
    const float m_reg_val = m_regBH[reg_safe];
    const float inv_l_reg = 1.0f / fmaxf(l_regBH[reg_safe], DENOM_EPS);

    __shared__ float sh_X [T_J][D_CONST];
    __shared__ float sh_V [T_J][D_CONST];
    __shared__ float sh_gY[T_J][D_CONST];
    __shared__ float sh_m[T_J];
    __shared__ float sh_l_inv[T_J];      // pre-inverted, multiply not divide

    for (int lBase=0; lBase<N; lBase+=T_J){
        // Cooperative load: all threads together cover the D range.
        int lt = threadIdx.y;                       // 0..T_K-1 (<= T_J)
        if (lt < T_J && (lBase+lt) < N){
            int lGlob = lBase + lt;
            #pragma unroll
            for (int d=threadIdx.x; d<D_CONST; d+=T_I){
                sh_X [lt][d] = bf2f(X_loopBH[lGlob*D_CONST + d]);
                sh_V [lt][d] = bf2f(V_loopBH[lGlob*D_CONST + d]);
                sh_gY[lt][d] = bf2f(gY_loopBH[lGlob*D_CONST + d]);
            }
            if (threadIdx.x == 0){
                sh_m[lt]     = m_loopBH[lGlob];
                sh_l_inv[lt] = 1.0f / fmaxf(l_loopBH[lGlob], DENOM_EPS);
            }
        }
        __syncthreads();

        if (active) {
            for (int lOff=0; lOff<T_J && (lBase+lOff)<N; ++lOff){
                const int lGlob = lBase + lOff;
                // logits = (X_out*X_reg) · X_loop — halved FMA vs the triple product
                float logits=0.f;
                #pragma unroll
                for (int d=0; d<D_CONST; ++d)
                    logits += prod_vec[d] * sh_X[lOff][d];
                logits *= scale;

                const bool loop_valid = mask_pair_allowed(mask, N, b, lGlob, out0, reg0);
                const bool reg_valid  = mask_pair_allowed(mask, N, b, reg0, out0, lGlob);
                float w_loop = loop_valid ? (__expf(fminf(logits - sh_m[lOff], EXP_CLIP)) * sh_l_inv[lOff]) : 0.0f;
                float w_reg  = reg_valid  ? (__expf(fminf(logits - m_reg_val,  EXP_CLIP)) * inv_l_reg) : 0.0f;

                #pragma unroll
                for (int d=0; d<D_CONST; ++d){
                    grad_acc[d] += w_loop * sh_gY[lOff][d] * v_reg_vec[d]  /* Y_loop path */
                                +  w_reg  * gy_reg_vec[d]  * sh_V[lOff][d]; /* Y_reg path */
                }
            }
        }
        __syncthreads();
    }

    if (active) {
        #pragma unroll
        for (int d=0; d<D_CONST; ++d)
            atomicAdd(&gV_outBH[out0*D_CONST + d], grad_acc[d]);
    }
}

// =============================================================================
// Gradient Kernels for V Tensors (Scatter Path)
// =============================================================================

// Same roles as V_gather_grad, but each scatter output is a product of two
// attention weights, so the out mode's own stats are consumed as well.
template<int D_CONST, bool OUT_IS_Y>
__global__ void V_scatter_grad(
    const bf16* __restrict__ X_out,     // [B,H,N,D]
    const bf16* __restrict__ X_reg,     // [B,H,N,D]
    const bf16* __restrict__ X_loop,    // [B,H,N,D]
    const bf16* __restrict__ V_reg,     // [B,H,N,D]   (V_2 slice)
    const bf16* __restrict__ V_loop,    // [B,H,N,D]   (V_2 slice)
    const bf16* __restrict__ gY_loop,   // [B,H,N,D]
    const bf16* __restrict__ gY_reg,    // [B,H,N,D]
    const float* __restrict__ m_out,    // [B,H,N]
    const float* __restrict__ l_out,    // [B,H,N]
    const float* __restrict__ m_loop,   // [B,H,N]
    const float* __restrict__ l_loop,   // [B,H,N]
    const float* __restrict__ m_reg,    // [B,H,N]
    const float* __restrict__ l_reg,    // [B,H,N]
    float*       __restrict__ gradV_out, // [B,H,N,D]
    const bool*  __restrict__ mask,
    int N, int H, float scale)
{
    const int tx = blockIdx.x * T_I + threadIdx.x;     // 0..N-1 (column)
    const int ty = blockIdx.y * T_K + threadIdx.y;     // 0..N-1 (row)
    const int out0 = OUT_IS_Y ? ty : tx;
    const int reg0 = OUT_IS_Y ? tx : ty;
    const int bh = blockIdx.z;          // flattened (batch, head)
    const int b = bh / H;

    // No early return: every thread is needed for the cooperative loads.
    const bool active = (out0 < N && reg0 < N);
    // Inactive threads still load, from a clamped row.
    const int out_safe = min(out0, N - 1);
    const int reg_safe = min(reg0, N - 1);

    const int64_t stride_BH = (int64_t)N * D_CONST;
    const bf16* X_outBH   = X_out   + (int64_t)bh * stride_BH;
    const bf16* X_regBH   = X_reg   + (int64_t)bh * stride_BH;
    const bf16* X_loopBH  = X_loop  + (int64_t)bh * stride_BH;
    const bf16* V_regBH   = V_reg   + (int64_t)bh * stride_BH;
    const bf16* V_loopBH  = V_loop  + (int64_t)bh * stride_BH;
    const bf16* gY_loopBH = gY_loop + (int64_t)bh * stride_BH;
    const bf16* gY_regBH  = gY_reg  + (int64_t)bh * stride_BH;
    const float* m_outBH  = m_out  + (int64_t)bh * N;
    const float* l_outBH  = l_out  + (int64_t)bh * N;
    const float* m_loopBH = m_loop + (int64_t)bh * N;
    const float* l_loopBH = l_loop + (int64_t)bh * N;
    const float* m_regBH  = m_reg  + (int64_t)bh * N;
    const float* l_regBH  = l_reg  + (int64_t)bh * N;
          float* gV_outBH = gradV_out + (int64_t)bh * stride_BH;

    float x_out_vec[D_CONST];
    float x_reg_vec[D_CONST], v_reg_vec[D_CONST];
    #pragma unroll
    for (int d=0; d<D_CONST; ++d){
        x_out_vec[d]  = bf2f(X_outBH[out_safe*D_CONST + d]);
        x_reg_vec[d]  = bf2f(X_regBH[reg_safe*D_CONST + d]);
        v_reg_vec[d]  = bf2f(V_regBH[reg_safe*D_CONST + d]);
    }
    // gY_reg row: on thread y it is warp-uniform, so read it straight from
    // global (broadcast, L1-hot) instead of spending a fourth D-long register
    // array that spills. On thread x each lane wants its own row, so cache it.
    const bf16* gy_reg_row = &gY_regBH[reg_safe*D_CONST];
    float gy_reg_cache[OUT_IS_Y ? D_CONST : 1];
    if constexpr (OUT_IS_Y) {
        #pragma unroll
        for (int d=0; d<D_CONST; ++d) gy_reg_cache[d] = bf2f(gy_reg_row[d]);
    }
    float grad_acc[D_CONST] = {0.0f};

    extern __shared__ float shmem[];
    float* sh_X  = shmem;                       // T_J * D_CONST
    float* sh_V  = sh_X  + T_J * D_CONST;       // T_J * D_CONST
    float* sh_gY = sh_V  + T_J * D_CONST;       // T_J * D_CONST
    float* sh_m  = sh_gY + T_J * D_CONST;       // T_J scalars
    float* sh_l  = sh_m  + T_J;

    for (int lBase=0; lBase < N; lBase+=T_J){
        // Cooperative load: all threads together cover the D range.
        const int lt = threadIdx.y;  // reuse y for co-load rows
        if (lt < T_J && (lBase+lt) < N){
            const int lGlob = lBase + lt;
            for (int d=threadIdx.x; d<D_CONST; d+=T_I){
                sh_X [lt*D_CONST + d] = bf2f(X_loopBH[lGlob*D_CONST + d]);
                sh_V [lt*D_CONST + d] = bf2f(V_loopBH[lGlob*D_CONST + d]);
                sh_gY[lt*D_CONST + d] = bf2f(gY_loopBH[lGlob*D_CONST + d]);
            }
            if (threadIdx.x == 0){
                sh_m[lt] = m_loopBH[lGlob];
                sh_l[lt] = l_loopBH[lGlob];
            }
        }
        __syncthreads();

        if (active) {
            for (int lOff=0; lOff<T_J && (lBase+lOff)<N; ++lOff){
                const int lGlob = lBase + lOff;
                float dot = 0.f;
                #pragma unroll
                for (int d=0; d<D_CONST; ++d)
                    dot += x_out_vec[d] * sh_X[lOff*D_CONST + d] * x_reg_vec[d];
                float logits = dot * scale;

                // Weight products in log space to avoid overflow:
                // A_a * A_b = exp(2*logits - m_a - m_b) / (l_a * l_b)
                float log_A_out  = logits - m_outBH[out_safe];
                float log_A_loop = logits - sh_m[lOff];
                float log_A_reg  = logits - m_regBH[reg_safe];

                float l_out_val  = fmaxf(l_outBH[out_safe], DENOM_EPS);
                float l_loop_val = fmaxf(sh_l[lOff], DENOM_EPS);
                float l_reg_val  = fmaxf(l_regBH[reg_safe], DENOM_EPS);

                const bool out_valid  = mask_pair_allowed(mask, N, b, out0, lGlob, reg0);
                const bool loop_valid = mask_pair_allowed(mask, N, b, lGlob, out0, reg0);
                const bool reg_valid  = mask_pair_allowed(mask, N, b, reg0, out0, lGlob);
                float w1 = (out_valid && reg_valid)
                    ? (__expf(fminf(log_A_out + log_A_reg, EXP_CLIP)) / (l_out_val * l_reg_val))
                    : 0.0f;
                float w2 = (out_valid && loop_valid)
                    ? (__expf(fminf(log_A_out + log_A_loop, EXP_CLIP)) / (l_out_val * l_loop_val))
                    : 0.0f;

                const float* gy_loop_vec = &sh_gY[lOff*D_CONST];
                const float* v_loop_vec  = &sh_V[lOff*D_CONST];

                #pragma unroll
                for (int d=0; d<D_CONST; ++d){
                    float gy_reg;
                    if constexpr (OUT_IS_Y) gy_reg = gy_reg_cache[d];
                    else                    gy_reg = bf2f(gy_reg_row[d]);
                    grad_acc[d] += w1 * gy_loop_vec[d] * v_reg_vec[d]
                                 + w2 * gy_reg * v_loop_vec[d];
                }
            }
        }
        __syncthreads();
    }

    if (active) {
        #pragma unroll
        for (int d=0; d<D_CONST; ++d)
            atomicAdd(&gV_outBH[out0*D_CONST + d], grad_acc[d]);
    }
}

// =============================================================================
// Tensor-core backward (gather-only path, cuda_docs/backward_tensor_cores.md)
// =============================================================================
// Engaged when the scatter cotangents are all zero (scatter unused): the cross
// terms d4/d5/d6 vanish, so every correction sum collapses FlashAttention-style
// to rowsum(dY o Y), computed host-side with ATen. What remains is ONE cube
// pass, run three times with permuted roles exactly like Y_gather_tc:
//
//   anchor a (one CTA per (b,h,a)) / rows r (16-row warp tiles) / cols c
//   (shared-memory resident). Score-shaped GEMMs per (r,c) tile, D contracted:
//
//     x   = scale * sum_d Xa[d]  * Xr[r,d]  * Xc[c,d]      (logits)
//     d_a =         sum_d gYa[d] * Vr[r,d]  * Vc[c,d]
//     d_r =         sum_d Va[d]  * gYr[r,d] * Vc[c,d]
//     d_c =         sum_d Va[d]  * Vr[r,d]  * gYc[c,d]
//
//   The anchor vector is folded into the A operand as a diagonal rescale of
//   raw row fragments in registers (packed bf16 __hmul2), so the four A
//   operands need no extra shared-memory buffers. Softmax weights come
//   straight from the forward stats (no online pass):
//
//     P_a = exp(x - m_a)/l_a    P_r = exp(x - m_r[r])/l_r[r]    P_c likewise
//     grad_A = (d_a - sum_a)*P_a + (d_r - sum_r[r])*P_r + (d_c - sum_c[c])*P_c
//
//   Output GEMMs contract c, with score C-fragments feeding A fragments in
//   registers (identical layouts, the FA2 trick):
//
//     Ug += grad_A @ Xc     U1 += P_r @ Vc     U2 += P_c @ gYc
//
//   and a Hadamard row-collapse epilogue emits both outputs with direct
//   stores (no atomics):
//
//     gradXa[a,d] = scale * sum_r Xr[r,d]*Ug[r,d]
//     gradVa[a,d] =         sum_r gYr[r,d]*U1[r,d] + Vr[r,d]*U2[r,d]
//
// Padded rows/cols (N not a multiple of the tile) are zero-filled with their
// inv-l set to 0, which zeroes P_r/P_c there; the epilogue's raw-row factors
// zero any remaining garbage.
//
// MASKED=true adds attention-mask support. Each of the three softmaxes is
// gated by its own anchor's mask row (mask_pair_allowed in the scalar path):
//
//   P_a live iff mask[a][r] && mask[a][c]
//   P_r live iff mask[r][a] && mask[r][c]
//   P_c live iff mask[c][a] && mask[c][r]
//
// Four of the six factors are separable in the iteration space and are folded
// into scales that the cell loop already multiplies by, so they cost nothing
// per cell: mask[r][a] and (via a 0/1 float) mask[a][r] are per-row, while
// mask[c][a] folds into P_c's inv-l and mask[a][c] into ilac_sm, the anchor's
// inv-l staged per column. Folding requires exp to be finite even for dead
// cells — a fully masked anchor carries m = NEG_INF and l = 0 from the forward,
// and exp(x + 1e30) * 1e6 would be inf — so the exponent is clamped at 0. That
// is free of consequence for live cells, where the forward's stats already
// guarantee x <= m, and it also subsumes the pad test: pad rows and cols carry
// a zero inv-l and read zero mask bits.
//
// The remaining two factors are genuinely 2-D non-separable, so the whole
// [N][N] mask goes resident, bit-packed, twice: row-major (mask[r][c] is bit c
// of row r) and bit-transposed (mask[c][r] is bit c of row r of mskT). One
// row-major copy can serve both orientations, but reading mask[c][r] from it is
// column-major — four loads and variable shifts per fragment against two
// aligned loads from the transpose. Each tile is (K_pad+1)*(K_pad/32) words,
// the spare row being the all-zero row that pad rows point at; 17.4 KB total at
// N=256. mskT is built in the prologue by 32x32 __ballot_sync bit transposes
// off the row-major tile, entirely in shared memory.
//
// The collapsed correction sums are unaffected by masking: sum = rowsum(dY o Y)
// holds for any weight matrix the forward actually used, and the forward
// zeroes masked cells, so a fully masked row gets Y = 0 and sum = 0.

constexpr int BTC_BJ = 128;       // rows per block iteration (8 warps x 16)
constexpr int BTC_BK = 32;        // cols per inner iteration
constexpr int BTC_WARPS = 8;

template<int D_CONST, bool MASKED>
__global__ __launch_bounds__(BTC_WARPS * 32, 1)
void Bwd_gather_tc(
    const bf16* __restrict__ Xa_bf,  // anchor side [B,H,N,D]
    const bf16* __restrict__ Va_bf,
    const bf16* __restrict__ gYa_bf,
    const bf16* __restrict__ Xr_bf,  // row side
    const bf16* __restrict__ Vr_bf,
    const bf16* __restrict__ gYr_bf,
    const bf16* __restrict__ Xc_bf,  // col side (smem resident)
    const bf16* __restrict__ Vc_bf,
    const bf16* __restrict__ gYc_bf,
    const float* __restrict__ m_a, const float* __restrict__ l_a, const float* __restrict__ sum_a,
    const float* __restrict__ m_r, const float* __restrict__ l_r, const float* __restrict__ sum_r,
    const float* __restrict__ m_c, const float* __restrict__ l_c, const float* __restrict__ sum_c,
    float* __restrict__ gradXa,      // [B,H,N,D] fp32, direct store
    float* __restrict__ gradVa,      // [B,H,N,D] fp32, direct store
    const bool* __restrict__ mask,   // [B,N,N] or null (MASKED only)
    int H, int N, int K_pad, float scale)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    static_assert(D_CONST == 64, "Bwd_gather_tc supports D=64 only");
    constexpr int D = D_CONST;
    constexpr int DPAD = D + 8;

    const int a = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;

    const int tid  = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int g    = lane / 4;
    const int tig  = lane % 4;
    const int lrow  = (lane & 7) + ((lane >> 3) & 1) * 8;
    const int lcol8 = (lane >> 4) * 8;
    const int brow  = (lane & 7) + ((lane >> 4) & 1) * 8;
    const int bcol8 = ((lane >> 3) & 1) * 8;

    extern __shared__ char smem_raw[];
    bf16* xr_sm   = reinterpret_cast<bf16*>(smem_raw);            // [BTC_BJ][DPAD]
    bf16* vr_sm   = xr_sm + BTC_BJ * DPAD;
    bf16* gyr_sm  = vr_sm + BTC_BJ * DPAD;
    bf16* xc_sm   = gyr_sm + BTC_BJ * DPAD;                       // [K_pad][DPAD]
    bf16* vc_sm   = xc_sm + K_pad * DPAD;
    bf16* gyc_sm  = vc_sm + K_pad * DPAD;
    float* anchX  = reinterpret_cast<float*>(gyc_sm + K_pad * DPAD);  // [D] scale*Xa
    float* anchV  = anchX + D;                                    // [D]
    float* anchG  = anchV + D;                                    // [D]
    float* mc_sm  = anchG + D;                                    // [K_pad]
    float* ilc_sm = mc_sm + K_pad;
    float* sc_sm  = ilc_sm + K_pad;
    float* mr_sm  = sc_sm + K_pad;                               // [BTC_BJ]
    float* ilr_sm = mr_sm + BTC_BJ;
    float* sr_sm  = ilr_sm + BTC_BJ;
    float* wOut   = sr_sm + BTC_BJ;                              // [BTC_WARPS][2*D]
    float* redOut = wOut + BTC_WARPS * 2 * D;                    // [2*D]
    // MASKED only (host omits these bytes when MASKED=false). Both tiles carry
    // one extra all-zero row at index K_pad, which pad rows read so that their
    // cells gate themselves off with no per-cell pad test (see below).
    const int NW = K_pad >> 5;    // bit-packing words per mask row
    uint32_t* msk_sm  = reinterpret_cast<uint32_t*>(redOut + 2 * D);  // [K_pad+1][NW]
    uint32_t* mskT_sm = msk_sm + (K_pad + 1) * NW;                    // [K_pad+1][NW]
    float* ilac_sm = reinterpret_cast<float*>(mskT_sm + (K_pad + 1) * NW);  // [K_pad]
    float* par_sm  = ilac_sm + K_pad;                                 // [BTC_BJ]

    const int64_t bh = (int64_t)b * H + h;
    const int64_t nd_off = bh * N * D;
    const int64_t a_off  = nd_off + (int64_t)a * D;
    const int64_t st_off = bh * N;

    // ---- one-time loads: resident col side (async), anchor, col stats ----
    constexpr int DV = D / 8;
    for (int idx = tid; idx < K_pad * DV; idx += blockDim.x) {
        const int k = idx / DV, dv = (idx % DV) * 8;
        if (k < N) {
            const int64_t off = nd_off + (int64_t)k * D + dv;
            cp_async16(xc_sm + k * DPAD + dv, Xc_bf + off);
            cp_async16(vc_sm + k * DPAD + dv, Vc_bf + off);
            cp_async16(gyc_sm + k * DPAD + dv, gYc_bf + off);
        } else {
            const uint4 z = make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(xc_sm + k * DPAD + dv) = z;
            *reinterpret_cast<uint4*>(vc_sm + k * DPAD + dv) = z;
            *reinterpret_cast<uint4*>(gyc_sm + k * DPAD + dv) = z;
        }
    }
    for (int d = tid; d < D; d += blockDim.x) {
        anchX[d] = scale * bf2f(Xa_bf[a_off + d]);
        anchV[d] = bf2f(Va_bf[a_off + d]);
        anchG[d] = bf2f(gYa_bf[a_off + d]);
    }
    for (int k = tid; k < K_pad; k += blockDim.x) {
        if (k < N) {
            mc_sm[k]  = m_c[st_off + k];
            ilc_sm[k] = 1.0f / fmaxf(l_c[st_off + k], DENOM_EPS);
            sc_sm[k]  = sum_c[st_off + k];
        } else {
            mc_sm[k] = 0.0f; ilc_sm[k] = 0.0f; sc_sm[k] = 0.0f;
        }
    }
    if (tid < 2 * D) redOut[tid] = 0.0f;

    const float ma  = m_a[st_off + a];
    const float ila = 1.0f / fmaxf(l_a[st_off + a], DENOM_EPS);
    const float sa  = sum_a[st_off + a];

    // ---- resident bit-packed mask: one word = 32 consecutive cols of a row ----
    if constexpr (MASKED) {
        const bool* mb = mask + (int64_t)b * N * N;
        for (int idx = tid; idx < (K_pad + 1) * NW; idx += blockDim.x) {
            const int r = idx / NW, w = idx - r * NW;
            uint32_t bits = 0u;
            const int cbase = w * 32;
            if (r < N && cbase < N) {
                // N % 16 == 0 and cbase % 32 == 0, so the row slice is
                // 4-byte aligned and lim is a multiple of 4.
                const int lim = min(32, N - cbase);
                const bool* row = mb + (int64_t)r * N + cbase;
                int t = 0;
                for (; t + 4 <= lim; t += 4) {
                    const uchar4 v = *reinterpret_cast<const uchar4*>(row + t);
                    bits |= ((v.x ? 1u : 0u) << t)       | ((v.y ? 1u : 0u) << (t + 1))
                          | ((v.z ? 1u : 0u) << (t + 2)) | ((v.w ? 1u : 0u) << (t + 3));
                }
                for (; t < lim; t++) if (row[t]) bits |= 1u << t;
            }
            msk_sm[idx] = bits;
        }
        // The two separable column factors fold into per-column floats, so they
        // cost nothing per cell: mask[c][a] zeroes P_c's inv-l outright, and
        // mask[a][c] rides along with the anchor's inv-l in ilac_sm. Both loops
        // walk the same c per thread as the ilc_sm staging above, so the
        // read-modify-write of ilc_sm needs no barrier.
        for (int c = tid; c < K_pad; c += blockDim.x) {
            const bool ac = (c < N) && mb[(int64_t)a * N + c];   // mask[a][c]
            const bool ca = (c < N) && mb[(int64_t)c * N + a];   // mask[c][a]
            ilac_sm[c] = ac ? ila : 0.0f;
            if (!ca) ilc_sm[c] = 0.0f;
        }
        __syncthreads();
        // Bit-transpose msk_sm into mskT_sm so that mskT[r][c] == mask[c][r]:
        // P_c's factor then reads the same shape as P_r's (one word per row per
        // 32-col tile, immediate shifts) instead of four column-major loads.
        // One __ballot_sync per output row of each 32x32 bit block.
        const int NB = NW;
        for (int blk = warp; blk < NB * NB; blk += BTC_WARPS) {
            const int bI = blk / NB, bJ = blk - bI * NB;
            const uint32_t w = msk_sm[(bI * 32 + lane) * NW + bJ];
            #pragma unroll
            for (int t = 0; t < 32; t++) {
                const uint32_t col = __ballot_sync(0xFFFFFFFFu, (w >> t) & 1u);
                if (lane == t) mskT_sm[(bJ * 32 + t) * NW + bI] = col;
            }
        }
        for (int w = tid; w < NW; w += blockDim.x) mskT_sm[K_pad * NW + w] = 0u;
    }

    asm volatile("cp.async.wait_all;\n" ::);
    __syncthreads();

    // ---- row blocks of BTC_BJ rows, one 16-row tile per warp ----
    for (int j0 = 0; j0 < N; j0 += BTC_BJ) {
        __syncthreads();  // previous iteration's smem reads done

        for (int idx = tid; idx < BTC_BJ * DV; idx += blockDim.x) {
            const int jl = idx / DV, dv = (idx % DV) * 8;
            const int j = j0 + jl;
            uint4 xq = make_uint4(0, 0, 0, 0), vq = xq, gq = xq;
            if (j < N) {
                const int64_t off = nd_off + (int64_t)j * D + dv;
                xq = *reinterpret_cast<const uint4*>(Xr_bf + off);
                vq = *reinterpret_cast<const uint4*>(Vr_bf + off);
                gq = *reinterpret_cast<const uint4*>(gYr_bf + off);
            }
            *reinterpret_cast<uint4*>(xr_sm + jl * DPAD + dv) = xq;
            *reinterpret_cast<uint4*>(vr_sm + jl * DPAD + dv) = vq;
            *reinterpret_cast<uint4*>(gyr_sm + jl * DPAD + dv) = gq;
        }
        for (int jl = tid; jl < BTC_BJ; jl += blockDim.x) {
            const int j = j0 + jl;
            if (j < N) {
                mr_sm[jl]  = m_r[st_off + j];
                float il   = 1.0f / fmaxf(l_r[st_off + j], DENOM_EPS);
                sr_sm[jl]  = sum_r[st_off + j];
                if constexpr (MASKED) {
                    // Both row-side factors are staged rather than kept in
                    // registers: at 255 registers the kernel spills, so a live
                    // register costs more than a rematerializable smem read.
                    if (!((msk_sm[j * NW + (a >> 5)] >> (a & 31)) & 1u))
                        il = 0.0f;                                   // mask[r][a]
                    par_sm[jl] = ((msk_sm[a * NW + (j >> 5)] >> (j & 31)) & 1u)
                               ? 1.0f : 0.0f;                        // mask[a][r]
                }
                ilr_sm[jl] = il;
            } else {
                mr_sm[jl] = 0.0f; ilr_sm[jl] = 0.0f; sr_sm[jl] = 0.0f;
                if constexpr (MASKED) par_sm[jl] = 0.0f;
            }
        }
        __syncthreads();

        const int jw = warp * 16;
        const float mr0 = mr_sm[jw + g],     sr0 = sr_sm[jw + g];
        const float mr1 = mr_sm[jw + g + 8], sr1 = sr_sm[jw + g + 8];
        const float ilr0 = ilr_sm[jw + g], ilr1 = ilr_sm[jw + g + 8];
        const float par0 = MASKED ? par_sm[jw + g]     : 1.0f;
        const float par1 = MASKED ? par_sm[jw + g + 8] : 1.0f;
        // Zero-filled pad rows produce x = 0, which with extreme stats can
        // push exp(x - m)/l past bf16 range (inf -> 0*inf = NaN downstream).
        // Forcing pad cells to NEG_INF zeroes all three weights exactly.
        const bool rpad0 = (j0 + jw + g)     >= N;
        const bool rpad1 = (j0 + jw + g + 8) >= N;

        // The two row-side mask factors were folded into ilr_sm / par_sm during
        // staging above. Pad rows point at the tiles' spare zero row, so their
        // 2-D factors read 0 and no per-cell pad test is needed; their inv-l and
        // par are already 0 from that same loop.
        const int rw0 = MASKED ? (rpad0 ? K_pad : (j0 + jw + g))     : 0;
        const int rw1 = MASKED ? (rpad1 ? K_pad : (j0 + jw + g + 8)) : 0;

        // Raw row fragments are col-invariant: load once, rescale into the
        // four A operands (A1/A3 share the Vr fragment). The rescale is done
        // in fp32 with a single bf16 rounding, matching Y_gather_tc's Qp
        // precision. A-fragment register e covers columns
        // ks*16 + (e<2 ? 0 : 8) + {2*tig, 2*tig+1}, identical for both rows.
        uint32_t A0[4][4], A1[4][4], A2[4][4], A3[4][4];
        #pragma unroll
        for (int ks = 0; ks < 4; ks++) {
            const int c0 = ks * 16 + 2 * tig;
            const float aX[4] = {anchX[c0], anchX[c0 + 1], anchX[c0 + 8], anchX[c0 + 9]};
            const float aV[4] = {anchV[c0], anchV[c0 + 1], anchV[c0 + 8], anchV[c0 + 9]};
            const float aG[4] = {anchG[c0], anchG[c0 + 1], anchG[c0 + 8], anchG[c0 + 9]};
            uint32_t rX[4], rV[4], rG[4];
            const int roff = (jw + lrow) * DPAD + ks * 16 + lcol8;
            ldmatrix_x4(rX, xr_sm + roff);
            ldmatrix_x4(rV, vr_sm + roff);
            ldmatrix_x4(rG, gyr_sm + roff);
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                const int hf = (e >> 1) * 2;
                const float2 fX = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&rX[e]));
                const float2 fV = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&rV[e]));
                const float2 fG = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&rG[e]));
                A0[ks][e] = pack_bf162(fX.x * aX[hf], fX.y * aX[hf + 1]);
                A1[ks][e] = pack_bf162(fV.x * aG[hf], fV.y * aG[hf + 1]);
                A2[ks][e] = pack_bf162(fG.x * aV[hf], fG.y * aV[hf + 1]);
                A3[ks][e] = pack_bf162(fV.x * aV[hf], fV.y * aV[hf + 1]);
            }
        }

        float Ug[8][4], U1[8][4], U2[8][4];
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            #pragma unroll
            for (int e = 0; e < 4; e++) { Ug[nt][e] = 0.0f; U1[nt][e] = 0.0f; U2[nt][e] = 0.0f; }
        }

        for (int k0 = 0; k0 < K_pad; k0 += BTC_BK) {
            // Score-shaped GEMMs: 4 accumulator sets over BTC_BK cols.
            float ax[4][4], ada[4][4], adr[4][4], adc[4][4];
            #pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                #pragma unroll
                for (int e = 0; e < 4; e++) {
                    ax[nt][e] = 0.0f; ada[nt][e] = 0.0f; adr[nt][e] = 0.0f; adc[nt][e] = 0.0f;
                }
            }
            #pragma unroll
            for (int p = 0; p < BTC_BK / 16; p++) {
                const bf16* bx = xc_sm + (k0 + p * 16 + brow) * DPAD + bcol8;
                const bf16* bv = vc_sm + (k0 + p * 16 + brow) * DPAD + bcol8;
                const bf16* bg = gyc_sm + (k0 + p * 16 + brow) * DPAD + bcol8;
                #pragma unroll
                for (int ks = 0; ks < 4; ks++) {
                    uint32_t bfr[4];
                    ldmatrix_x4(bfr, bx + ks * 16);
                    mma_bf16_m16n8k16(ax[2 * p],     A0[ks], bfr);
                    mma_bf16_m16n8k16(ax[2 * p + 1], A0[ks], bfr + 2);
                    ldmatrix_x4(bfr, bv + ks * 16);   // shared by d_a and d_r
                    mma_bf16_m16n8k16(ada[2 * p],     A1[ks], bfr);
                    mma_bf16_m16n8k16(ada[2 * p + 1], A1[ks], bfr + 2);
                    mma_bf16_m16n8k16(adr[2 * p],     A2[ks], bfr);
                    mma_bf16_m16n8k16(adr[2 * p + 1], A2[ks], bfr + 2);
                    ldmatrix_x4(bfr, bg + ks * 16);
                    mma_bf16_m16n8k16(adc[2 * p],     A3[ks], bfr);
                    mma_bf16_m16n8k16(adc[2 * p + 1], A3[ks], bfr + 2);
                }
            }

            // Elementwise: weights from forward stats, Jacobian-corrected
            // grad_A; repack C-fragments as output-GEMM A-fragments.
            uint32_t gAf[BTC_BK / 16][4], Prf[BTC_BK / 16][4], Pcf[BTC_BK / 16][4];
            #pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                const int c = k0 + nt * 8 + 2 * tig;
                const float mc0 = mc_sm[c],     ilc0 = ilc_sm[c],     sc0 = sc_sm[c];
                const float mc1 = mc_sm[c + 1], ilc1 = ilc_sm[c + 1], sc1 = sc_sm[c + 1];
                // Only the two genuinely 2-D factors are left per cell, and the
                // transposed tile makes them the same shape: bits 0/1 of rc*
                // are mask[r][c] and mask[r][c+1], bits 0/1 of cr* are
                // mask[c][r] and mask[c+1][r] (c is even, so each pair shares a
                // word). Four column-major loads became two aligned ones.
                // (Hoisting these out of the nt loop is possible — c >> 5 is
                // tile-invariant since BTC_BK == 32 — but measured 2% slower on
                // H100: the kernel is register-bound, and keeping the words live
                // across the tile costs more spill than the saved loads.)
                uint32_t rc0 = ~0u, rc1 = ~0u, cr0 = ~0u, cr1 = ~0u;
                float ilac0 = 0.0f, ilac1 = 0.0f;
                if constexpr (MASKED) {
                    const int cw = c >> 5, cb = c & 31;
                    rc0 = msk_sm[rw0 * NW + cw] >> cb;
                    rc1 = msk_sm[rw1 * NW + cw] >> cb;
                    cr0 = mskT_sm[rw0 * NW + cw] >> cb;
                    cr1 = mskT_sm[rw1 * NW + cw] >> cb;
                    ilac0 = ilac_sm[c];
                    ilac1 = ilac_sm[c + 1];
                }
                float gA[4], Pr[4], Pc[4];
                #pragma unroll
                for (int e = 0; e < 4; e++) {
                    const bool hi = (e >= 2), c1 = (e & 1);
                    const float mrr = hi ? mr1 : mr0;
                    const float ilr = hi ? ilr1 : ilr0;
                    const float srr = hi ? sr1 : sr0;
                    const float mcc = c1 ? mc1 : mc0;
                    const float ilc = c1 ? ilc1 : ilc0;
                    const float scc = c1 ? sc1 : sc0;
                    float Pa;
                    if constexpr (MASKED) {
                        // Clamping the exponent at 0 is what lets every gate be
                        // a plain multiply or select: live cells always have
                        // x <= m so the clamp never touches them, while dead
                        // ones can no longer reach inf and turn 0*inf into NaN.
                        // That kills the pad test too (pad rows/cols carry a
                        // zero inv-l and zero mask bits), so no `pad` term
                        // appears below. Remaining per cell: mask[r][c] and
                        // mask[c][r]; the anchor's own factors already rode in
                        // via ilac (col) and par (row).
                        const float x = ax[nt][e];
                        const float ilac = c1 ? ilac1 : ilac0;
                        const float par  = hi ? par1 : par0;
                        // Gate the *scale*, never the exp. Guarding the whole
                        // expression lets nvcc branch around the MUFU, and a
                        // scattered mask then diverges inside the warp and runs
                        // both sides: that cost `random` 21% over `causal`.
                        // Selecting on inv-l keeps every cell's cost identical.
                        const float ilrg = (((hi ? rc1 : rc0) >> c1) & 1u) ? ilr : 0.0f;
                        const float ilcg = (((hi ? cr1 : cr0) >> c1) & 1u) ? ilc : 0.0f;
                        Pa    = __expf(fminf(x - ma,  0.0f)) * ilac * par;
                        Pr[e] = __expf(fminf(x - mrr, 0.0f)) * ilrg;
                        Pc[e] = __expf(fminf(x - mcc, 0.0f)) * ilcg;
                    } else {
                        const bool pad = (hi ? rpad1 : rpad0) | (c + c1 >= N);
                        const float x = pad ? NEG_INF : ax[nt][e];
                        Pa    = __expf(x - ma)  * ila;
                        Pr[e] = __expf(x - mrr) * ilr;
                        Pc[e] = __expf(x - mcc) * ilc;
                    }
                    gA[e] = (ada[nt][e] - sa) * Pa
                          + (adr[nt][e] - srr) * Pr[e]
                          + (adc[nt][e] - scc) * Pc[e];
                }
                const int s2 = nt >> 1, hf = nt & 1;
                gAf[s2][2 * hf + 0] = pack_bf162(gA[0], gA[1]);
                gAf[s2][2 * hf + 1] = pack_bf162(gA[2], gA[3]);
                Prf[s2][2 * hf + 0] = pack_bf162(Pr[0], Pr[1]);
                Prf[s2][2 * hf + 1] = pack_bf162(Pr[2], Pr[3]);
                Pcf[s2][2 * hf + 0] = pack_bf162(Pc[0], Pc[1]);
                Pcf[s2][2 * hf + 1] = pack_bf162(Pc[2], Pc[3]);
            }

            // Output GEMMs: contract cols (B fragments via ldmatrix.trans).
            #pragma unroll
            for (int s2 = 0; s2 < BTC_BK / 16; s2++) {
                const bf16* px = xc_sm + (k0 + s2 * 16 + lrow) * DPAD + lcol8;
                const bf16* pv = vc_sm + (k0 + s2 * 16 + lrow) * DPAD + lcol8;
                const bf16* pg = gyc_sm + (k0 + s2 * 16 + lrow) * DPAD + lcol8;
                #pragma unroll
                for (int np = 0; np < 4; np++) {
                    uint32_t bfr[4];
                    ldmatrix_x4_trans(bfr, px + np * 16);
                    mma_bf16_m16n8k16(Ug[2 * np],     gAf[s2], bfr);
                    mma_bf16_m16n8k16(Ug[2 * np + 1], gAf[s2], bfr + 2);
                    ldmatrix_x4_trans(bfr, pv + np * 16);
                    mma_bf16_m16n8k16(U1[2 * np],     Prf[s2], bfr);
                    mma_bf16_m16n8k16(U1[2 * np + 1], Prf[s2], bfr + 2);
                    ldmatrix_x4_trans(bfr, pg + np * 16);
                    mma_bf16_m16n8k16(U2[2 * np],     Pcf[s2], bfr);
                    mma_bf16_m16n8k16(U2[2 * np + 1], Pcf[s2], bfr + 2);
                }
            }
        }

        // ---- epilogue: Hadamard row-collapse of this warp's 16 rows ----
        float ng[16], nv[16];
        const bf16* xr0 = xr_sm + (jw + g) * DPAD + 2 * tig;
        const bf16* xr1 = xr_sm + (jw + g + 8) * DPAD + 2 * tig;
        const bf16* vr0 = vr_sm + (jw + g) * DPAD + 2 * tig;
        const bf16* vr1 = vr_sm + (jw + g + 8) * DPAD + 2 * tig;
        const bf16* gy0 = gyr_sm + (jw + g) * DPAD + 2 * tig;
        const bf16* gy1 = gyr_sm + (jw + g + 8) * DPAD + 2 * tig;
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            const float2 x0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(xr0 + nt * 8));
            const float2 x1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(xr1 + nt * 8));
            const float2 v0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(vr0 + nt * 8));
            const float2 v1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(vr1 + nt * 8));
            const float2 g0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(gy0 + nt * 8));
            const float2 g1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(gy1 + nt * 8));
            ng[2 * nt + 0] = x0.x * Ug[nt][0] + x1.x * Ug[nt][2];
            ng[2 * nt + 1] = x0.y * Ug[nt][1] + x1.y * Ug[nt][3];
            nv[2 * nt + 0] = g0.x * U1[nt][0] + g1.x * U1[nt][2]
                           + v0.x * U2[nt][0] + v1.x * U2[nt][2];
            nv[2 * nt + 1] = g0.y * U1[nt][1] + g1.y * U1[nt][3]
                           + v0.y * U2[nt][1] + v1.y * U2[nt][3];
        }
        #pragma unroll
        for (int off = 4; off <= 16; off <<= 1) {
            #pragma unroll
            for (int e = 0; e < 16; e++) {
                ng[e] += __shfl_xor_sync(0xFFFFFFFF, ng[e], off);
                nv[e] += __shfl_xor_sync(0xFFFFFFFF, nv[e], off);
            }
        }
        if (lane < 4) {
            #pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                wOut[warp * 2 * D + nt * 8 + 2 * lane]         = ng[2 * nt + 0];
                wOut[warp * 2 * D + nt * 8 + 2 * lane + 1]     = ng[2 * nt + 1];
                wOut[warp * 2 * D + D + nt * 8 + 2 * lane]     = nv[2 * nt + 0];
                wOut[warp * 2 * D + D + nt * 8 + 2 * lane + 1] = nv[2 * nt + 1];
            }
        }
        __syncthreads();
        if (tid < 2 * D) {
            float acc = 0.0f;
            #pragma unroll
            for (int w = 0; w < BTC_WARPS; w++) acc += wOut[w * 2 * D + tid];
            redOut[tid] += acc;
        }
    }
    __syncthreads();

    // ---- direct stores: this CTA exclusively owns row a of both outputs ----
    if (tid < D) {
        gradXa[a_off + tid] = scale * redOut[tid];
    } else if (tid < 2 * D) {
        gradVa[a_off + tid - D] = redOut[tid];
    }
#endif  // __CUDA_ARCH__ >= 800
}

// =============================================================================
// NOTE: The old 3D jacobian_corrections kernel has been replaced by two-pass
// 2D-tiled correction passes using QS_grad_kernel<true> and R_grad_kernel<true>.
// This eliminates the 3D thread grid (8192 blocks) in favor of 2D grids
// (128 blocks each), reducing atomic contention by 32x.
// =============================================================================



// =============================================================================
// Gradient Kernels for Q, R, S (with integrated Jacobian corrections)
// =============================================================================
//
// Architecture: Two-pass 2D-tiled approach
//   Phase 1: QS_grad_kernel<true>  → computes sum_q, sum_s (correction sums)
//            R_grad_kernel<true>   → computes sum_r (correction sum)
//   Phase 2: QS_grad_kernel<false> → computes gradQ, gradS using corrections
//            R_grad_kernel<false>  → computes gradR using corrections
//
// This replaces the old 3D jacobian_corrections kernel (8192 blocks, 512
// threads, 256 atomics/element) with 2D-tiled correction passes (128 blocks,
// 256 threads, 8 atomics/element) — a 32x reduction in atomic contention.
//
// Both kernels use a compile-time template bool CORRECTION_ONLY to share
// 95% of the code between correction and gradient modes.
//
// =============================================================================

/**
 * QS_grad_kernel - Computes gradQ and gradS with Jacobian corrections.
 * 
 * CORRECTION_ONLY=true:  Computes correction sums (sum_q, sum_r, sum_s)
 * CORRECTION_ONLY=false: Computes gradQ and gradS using precomputed corrections
 */
template<bool CORRECTION_ONLY, int BLOCK_I, int BLOCK_J, int BLOCK_K, int D_CONST, int REG_CAP = D_CONST>
__global__ void __launch_bounds__(256, 1) QS_grad_kernel(
    const bf16* __restrict__ Q,
    const bf16* __restrict__ R,
    const bf16* __restrict__ S,
    const bf16* __restrict__ Vq1, const bf16* __restrict__ Vq2,
    const bf16* __restrict__ Vr1, const bf16* __restrict__ Vr2,
    const bf16* __restrict__ Vs1, const bf16* __restrict__ Vs2,
    const bf16* __restrict__ grad_Yq,
    const bf16* __restrict__ grad_Yr,
    const bf16* __restrict__ grad_Ys,
    const bf16* __restrict__ grad_Yq_,
    const bf16* __restrict__ grad_Yr_,
    const bf16* __restrict__ grad_Ys_,
    const float* __restrict__ m_i, const float* __restrict__ l_i,
    const float* __restrict__ m_j, const float* __restrict__ l_j,
    const float* __restrict__ m_k, const float* __restrict__ l_k,
    float* __restrict__ sum_q,
    float* __restrict__ sum_r,
    float* __restrict__ sum_s,
    float* __restrict__ gradQ,
    float* __restrict__ gradS,
    const bool* __restrict__ mask,
    int  N, int H, float scale)
{
    const int i0 = blockIdx.x * BLOCK_I + threadIdx.x;
    const int k0 = blockIdx.y * BLOCK_K + threadIdx.y;
    const int bh = blockIdx.z;
    const int b = bh / H;
    const bool valid = (i0 < N && k0 < N);

    // Per (B,H) base pointers
    const int64_t stride_BH = (int64_t)N * D_CONST;
    const bf16* Qbh    = Q   + bh * stride_BH;
    const bf16* Rbh    = R   + bh * stride_BH;
    const bf16* Sbh    = S   + bh * stride_BH;
    const bf16* Vq1bh  = Vq1 + bh * stride_BH;
    const bf16* Vq2bh  = Vq2 + bh * stride_BH;
    const bf16* Vr1bh  = Vr1 + bh * stride_BH;
    const bf16* Vr2bh  = Vr2 + bh * stride_BH;
    const bf16* Vs1bh  = Vs1 + bh * stride_BH;
    const bf16* Vs2bh  = Vs2 + bh * stride_BH;
    const bf16* gYqbh  = grad_Yq + bh * stride_BH;
    const bf16* gYrbh  = grad_Yr + bh * stride_BH;
    const bf16* gYsbh  = grad_Ys + bh * stride_BH;
    const bf16* gYq2bh = grad_Yq_ + bh * stride_BH;
    const bf16* gYr2bh = grad_Yr_ + bh * stride_BH;
    const bf16* gYs2bh = grad_Ys_ + bh * stride_BH;
    const float* miBH  = m_i + bh * N;
    const float* liBH  = l_i + bh * N;
    const float* mjBH  = m_j + bh * N;
    const float* ljBH  = l_j + bh * N;
    const float* mkBH  = m_k + bh * N;
    const float* lkBH  = l_k + bh * N;
    float* sum_qBH     = sum_q + bh * N;
    float* sum_sBH     = sum_s + bh * N;

    constexpr int D_PAD = D_CONST + 1;  // bank-conflict-free stride
    extern __shared__ float shmem[];
    float* sh_Qi   = shmem;
    float* sh_Vq1i = sh_Qi   + BLOCK_I * D_PAD;
    float* sh_Vq2i = sh_Vq1i + BLOCK_I * D_PAD;
    float* sh_dYi  = sh_Vq2i + BLOCK_I * D_PAD;
    float* sh_dYi2 = sh_dYi  + BLOCK_I * D_PAD;
    float* sh_Sk   = sh_dYi2 + BLOCK_I * D_PAD;
    float* sh_Vs1k = sh_Sk   + BLOCK_K * D_PAD;
    float* sh_Vs2k = sh_Vs1k + BLOCK_K * D_PAD;
    float* sh_dYk  = sh_Vs2k + BLOCK_K * D_PAD;
    float* sh_dYk2 = sh_dYk  + BLOCK_K * D_PAD;
    float* sh_R    = sh_dYk2 + BLOCK_K * D_PAD;
    float* sh_Vr1  = sh_R    + BLOCK_J * D_CONST;
    float* sh_Vr2  = sh_Vr1  + BLOCK_J * D_CONST;
    float* sh_gYj  = sh_Vr2  + BLOCK_J * D_CONST;
    float* sh_gYj2 = sh_gYj  + BLOCK_J * D_CONST;
    float* sh_mj   = sh_gYj2 + BLOCK_J * D_CONST;
    float* sh_lj   = sh_mj   + BLOCK_J;
    float* sh_sumr = sh_lj   + BLOCK_J;

    {
        const int tid = threadIdx.x + threadIdx.y * BLOCK_I;
        const int nThreads = BLOCK_I * BLOCK_K;
        for (int idx = tid; idx < BLOCK_I * D_CONST; idx += nThreads) {
            const int ii = idx / D_CONST;
            const int dd = idx % D_CONST;
            const int iGlob = blockIdx.x * BLOCK_I + ii;
            if (iGlob < N) {
                sh_Qi  [ii * D_PAD + dd] = bf2f(Qbh[iGlob * D_CONST + dd]);
                sh_Vq1i[ii * D_PAD + dd] = bf2f(Vq1bh[iGlob * D_CONST + dd]);
                sh_Vq2i[ii * D_PAD + dd] = bf2f(Vq2bh[iGlob * D_CONST + dd]);
                sh_dYi [ii * D_PAD + dd] = bf2f(gYqbh[iGlob * D_CONST + dd]);
                sh_dYi2[ii * D_PAD + dd] = bf2f(gYq2bh[iGlob * D_CONST + dd]);
            } else {
                sh_Qi  [ii * D_PAD + dd] = 0.0f;
                sh_Vq1i[ii * D_PAD + dd] = 0.0f;
                sh_Vq2i[ii * D_PAD + dd] = 0.0f;
                sh_dYi [ii * D_PAD + dd] = 0.0f;
                sh_dYi2[ii * D_PAD + dd] = 0.0f;
            }
        }
        for (int idx = tid; idx < BLOCK_K * D_CONST; idx += nThreads) {
            const int kk = idx / D_CONST;
            const int dd = idx % D_CONST;
            const int kGlob = blockIdx.y * BLOCK_K + kk;
            if (kGlob < N) {
                sh_Sk  [kk * D_PAD + dd] = bf2f(Sbh[kGlob * D_CONST + dd]);
                sh_Vs1k[kk * D_PAD + dd] = bf2f(Vs1bh[kGlob * D_CONST + dd]);
                sh_Vs2k[kk * D_PAD + dd] = bf2f(Vs2bh[kGlob * D_CONST + dd]);
                sh_dYk [kk * D_PAD + dd] = bf2f(gYsbh[kGlob * D_CONST + dd]);
                sh_dYk2[kk * D_PAD + dd] = bf2f(gYs2bh[kGlob * D_CONST + dd]);
            } else {
                sh_Sk  [kk * D_PAD + dd] = 0.0f;
                sh_Vs1k[kk * D_PAD + dd] = 0.0f;
                sh_Vs2k[kk * D_PAD + dd] = 0.0f;
                sh_dYk [kk * D_PAD + dd] = 0.0f;
                sh_dYk2[kk * D_PAD + dd] = 0.0f;
            }
        }
    }

    float mi = 0.0f, li = 1.0f, mk = 0.0f, lk = 1.0f;
    if (valid) {
        mi = miBH[i0];  li = liBH[i0];
        mk = mkBH[k0];  lk = lkBH[k0];
    }

    __syncthreads();

    float reg_sum_q = 0.0f, reg_sum_s = 0.0f;
    float sumQi = 0.0f, sumSk = 0.0f;
    // Algebraic factoring: accumulate rj_weighted[d] = Σⱼ grad_A_j * bf2f(R[j,d])
    // Then gradQ[d] = rj_weighted[d] * bf2f(S[k,d]), gradS[d] = rj_weighted[d] * bf2f(Q[i,d])
    // This replaces two D-sized accumulators with one, saving 64 registers and
    // reducing the hot inner loop from 3 shmem loads/d to 1 shmem load/d.
    float rj_weighted[REG_CAP];
    if constexpr (!CORRECTION_ONLY) {
        if (valid) {
            sumQi = sum_qBH[i0];
            sumSk = sum_sBH[k0];
        }
        for (int d = 0; d < REG_CAP; ++d) rj_weighted[d] = 0.0f;
    }

    const int sh_i_off = threadIdx.x * D_PAD;
    const int sh_k_off = threadIdx.y * D_PAD;

    for (int jBase = 0; jBase < N; jBase += BLOCK_J) {
        if constexpr (CORRECTION_ONLY) {
            const int lid = threadIdx.x + threadIdx.y * BLOCK_I;
            if (lid < BLOCK_J) sh_sumr[lid] = 0.0f;
        }

        const int tid_l      = threadIdx.x + threadIdx.y * BLOCK_I;
        const int nThreads_l = BLOCK_I * BLOCK_K;
        for (int idx = tid_l; idx < BLOCK_J * D_CONST; idx += nThreads_l) {
            const int jj = idx / D_CONST;
            const int dd = idx % D_CONST;
            const int jGlob = jBase + jj;
            if (jGlob < N) {
                sh_R  [jj*D_CONST + dd] = bf2f(Rbh[jGlob*D_CONST + dd]);
                sh_Vr1[jj*D_CONST + dd] = bf2f(Vr1bh[jGlob*D_CONST + dd]);
                sh_Vr2[jj*D_CONST + dd] = bf2f(Vr2bh[jGlob*D_CONST + dd]);
                sh_gYj[jj*D_CONST + dd] = bf2f(gYrbh[jGlob*D_CONST + dd]);
                sh_gYj2[jj*D_CONST + dd] = bf2f(gYr2bh[jGlob*D_CONST + dd]);
            } else {
                sh_R  [jj*D_CONST + dd] = 0.0f;
                sh_Vr1[jj*D_CONST + dd] = 0.0f;
                sh_Vr2[jj*D_CONST + dd] = 0.0f;
                sh_gYj[jj*D_CONST + dd] = 0.0f;
                sh_gYj2[jj*D_CONST + dd] = 0.0f;
            }
        }
        if (tid_l < BLOCK_J) {
            const int jGlob = jBase + tid_l;
            if (jGlob < N) {
                sh_mj[tid_l] = mjBH[jGlob];
                sh_lj[tid_l] = ljBH[jGlob];
                if constexpr (!CORRECTION_ONLY) {
                    sh_sumr[tid_l] = (sum_r + (int64_t)bh * N)[jGlob];
                }
            } else {
                sh_mj[tid_l] = 0.0f;
                sh_lj[tid_l] = 1.0f;
                if constexpr (!CORRECTION_ONLY) {
                    sh_sumr[tid_l] = 0.0f;
                }
            }
        }
        __syncthreads();

        // ============================================================
        // D-tiled dot products with j sub-tiling.
        // Precomputes i/k pairwise products per D_TILE, then sweeps j.
        // i/k values are loaded once per D_TILE instead of once per
        // (j, d) pair, eliminating 50% of shmem loads.
        //
        // Shmem loads per j-tile: 6,144 (was 12,288).
        // Register cost: 7 × J_SUB = 28 per-j accumulators (was 7 scalar).
        // ============================================================
        constexpr int J_SUB  = 4;  // j sub-tile size
        constexpr int D_TILE = 4;  // d tile size

        for (int jSub = 0; jSub < BLOCK_J && (jBase + jSub) < N; jSub += J_SUB) {
            // Per-j accumulators for this sub-tile
            float dot_j[J_SUB], d1_j[J_SUB], d2_j[J_SUB], d3_j[J_SUB];
            float d4_j[J_SUB], d5_j[J_SUB], d6_j[J_SUB];
            #pragma unroll
            for (int jj = 0; jj < J_SUB; jj++) {
                dot_j[jj] = 0.f; d1_j[jj] = 0.f; d2_j[jj] = 0.f; d3_j[jj] = 0.f;
                d4_j[jj] = 0.f; d5_j[jj] = 0.f; d6_j[jj] = 0.f;
            }

            // D-tiled precomputation: d-outer, j-inner
            for (int d_base = 0; d_base < D_CONST; d_base += D_TILE) {
                // Precompute 7 i/k pairwise products (8 shmem loads × D_TILE)
                float p_dot[D_TILE], p_d1[D_TILE], p_d2[D_TILE], p_d3[D_TILE];
                float p_d4[D_TILE], p_d5[D_TILE], p_d6[D_TILE];
                #pragma unroll
                for (int dd = 0; dd < D_TILE; dd++) {
                    const int d = d_base + dd;
                    const float qi   = sh_Qi  [sh_i_off + d];
                    const float sk   = sh_Sk  [sh_k_off + d];
                    const float vq1i = sh_Vq1i[sh_i_off + d];
                    const float vq2i = sh_Vq2i[sh_i_off + d];
                    const float vs1k = sh_Vs1k[sh_k_off + d];
                    const float vs2k = sh_Vs2k[sh_k_off + d];
                    const float dyi  = sh_dYi [sh_i_off + d];
                    const float dyi2 = sh_dYi2[sh_i_off + d];
                    const float dyk  = sh_dYk [sh_k_off + d];
                    const float dyk2 = sh_dYk2[sh_k_off + d];

                    p_dot[dd] = qi * sk;
                    p_d1[dd]  = dyi * vs1k;
                    p_d2[dd]  = vq1i * vs1k;
                    p_d3[dd]  = vq1i * dyk;
                    p_d4[dd]  = dyi2 * vs2k;
                    p_d5[dd]  = vq2i * vs2k;
                    p_d6[dd]  = vq2i * dyk2;
                }

                // Accumulate over j sub-tile. j-arrays have stride D_CONST
                // (no padding) and d_base is a multiple of D_TILE=4, so each
                // 4-float slice is 16-byte aligned → one LDS.128 per array.
                #pragma unroll
                for (int jj = 0; jj < J_SUB; jj++) {
                    const int jOff = jSub + jj;
                    if (jBase + jOff >= N) break;
                    const int rowOff = jOff * D_CONST + d_base;
                    const float4 rj4  = *reinterpret_cast<const float4*>(&sh_R  [rowOff]);
                    const float4 vr14 = *reinterpret_cast<const float4*>(&sh_Vr1[rowOff]);
                    const float4 vr24 = *reinterpret_cast<const float4*>(&sh_Vr2[rowOff]);
                    const float4 gyj4 = *reinterpret_cast<const float4*>(&sh_gYj[rowOff]);
                    const float4 gyj24 = *reinterpret_cast<const float4*>(&sh_gYj2[rowOff]);
                    const float rj[4]  = { rj4.x,  rj4.y,  rj4.z,  rj4.w  };
                    const float vr1[4] = { vr14.x, vr14.y, vr14.z, vr14.w };
                    const float vr2[4] = { vr24.x, vr24.y, vr24.z, vr24.w };
                    const float gyj[4] = { gyj4.x, gyj4.y, gyj4.z, gyj4.w };
                    const float gyj2[4] = { gyj24.x, gyj24.y, gyj24.z, gyj24.w };
                    #pragma unroll
                    for (int dd = 0; dd < D_TILE; dd++) {
                        dot_j[jj] += p_dot[dd] * rj[dd];
                        d1_j[jj]  += p_d1[dd]  * vr1[dd];
                        d2_j[jj]  += p_d2[dd]  * gyj[dd];
                        d3_j[jj]  += p_d3[dd]  * vr1[dd];
                        d4_j[jj]  += p_d4[dd]  * vr2[dd];
                        d5_j[jj]  += p_d5[dd]  * gyj2[dd];
                        d6_j[jj]  += p_d6[dd]  * vr2[dd];
                    }
                }
            }

            // Process accumulated dot products for this sub-tile
            #pragma unroll
            for (int jj = 0; jj < J_SUB; jj++) {
                const int jOff = jSub + jj;
                if (jBase + jOff >= N) break;
                const int jGlob = jBase + jOff;

                const float logits = dot_j[jj] * scale;
                const bool aq_valid = valid && mask_pair_allowed(mask, N, b, i0, jGlob, k0);
                const bool ar_valid = valid && mask_pair_allowed(mask, N, b, jGlob, i0, k0);
                const bool as_valid = valid && mask_pair_allowed(mask, N, b, k0, i0, jGlob);
                const float Aq = aq_valid ? (__expf(fminf(logits - mi, EXP_CLIP)) / fmaxf(li, DENOM_EPS)) : 0.0f;
                const float Ar = ar_valid ? (__expf(fminf(logits - sh_mj[jOff], EXP_CLIP)) / fmaxf(sh_lj[jOff], DENOM_EPS)) : 0.0f;
                const float As = as_valid ? (__expf(fminf(logits - mk, EXP_CLIP)) / fmaxf(lk, DENOM_EPS)) : 0.0f;

                const float gAq = d1_j[jj] + d5_j[jj] * As + d6_j[jj] * Ar;
                const float gAr = d2_j[jj] + d4_j[jj] * As + d6_j[jj] * Aq;
                const float gAs = d3_j[jj] + d4_j[jj] * Ar + d5_j[jj] * Aq;

                if constexpr (CORRECTION_ONLY) {
                    reg_sum_q += gAq * Aq;
                    reg_sum_s += gAs * As;
                    if (valid) atomicAdd(&sh_sumr[jOff], gAr * Ar);
                } else {
                    const float grad_A = (gAq - sumQi) * Aq
                                       + (gAr - sh_sumr[jOff]) * Ar
                                       + (gAs - sumSk) * As;
                    // Factored accumulation: float4 load per 4 d's (LDS.128).
                    #pragma unroll
                    for (int d = 0; d < D_CONST; d += 4) {
                        const float4 rj4 = *reinterpret_cast<const float4*>(&sh_R[jOff*D_CONST + d]);
                        rj_weighted[d+0] += grad_A * rj4.x;
                        rj_weighted[d+1] += grad_A * rj4.y;
                        rj_weighted[d+2] += grad_A * rj4.z;
                        rj_weighted[d+3] += grad_A * rj4.w;
                    }
                }
            }
        }
        __syncthreads();

        if constexpr (CORRECTION_ONLY) {
            const int lid = threadIdx.x + threadIdx.y * BLOCK_I;
            if (lid < BLOCK_J && (jBase + lid) < N)
                atomicAdd(&(sum_r + (int64_t)bh * N)[jBase + lid], sh_sumr[lid]);
            __syncthreads();
        }
    }

    if constexpr (CORRECTION_ONLY) {
        float* reduce_buf = shmem;
        reduce_buf[threadIdx.x * BLOCK_K + threadIdx.y] = valid ? reg_sum_q : 0.0f;
        __syncthreads();
        for (int s = BLOCK_K / 2; s > 0; s >>= 1) {
            if (threadIdx.y < s)
                reduce_buf[threadIdx.x * BLOCK_K + threadIdx.y] +=
                    reduce_buf[threadIdx.x * BLOCK_K + threadIdx.y + s];
            __syncthreads();
        }
        if (threadIdx.y == 0 && i0 < N)
            atomicAdd(&sum_qBH[i0], reduce_buf[threadIdx.x * BLOCK_K]);

        reduce_buf[threadIdx.x * BLOCK_K + threadIdx.y] = valid ? reg_sum_s : 0.0f;
        __syncthreads();
        for (int s = BLOCK_I / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s)
                reduce_buf[threadIdx.x * BLOCK_K + threadIdx.y] +=
                    reduce_buf[(threadIdx.x + s) * BLOCK_K + threadIdx.y];
            __syncthreads();
        }
        if (threadIdx.x == 0 && k0 < N)
            atomicAdd(&sum_sBH[k0], reduce_buf[threadIdx.y]);
    } else {
        // Algebraic factoring epilogue:
        //   gradQ[i,d] = scale * rj_weighted[d] * bf2f(S[k,d])
        //   gradS[k,d] = scale * rj_weighted[d] * bf2f(Q[i,d])
        // bf2f(S[k,d]) and bf2f(Q[i,d]) are still in shared memory (loaded before j-loop).
        float* gQbh = gradQ + bh * stride_BH;
        float* gSbh = gradS + bh * stride_BH;
        if (valid) {
            for (int d = 0; d < D_CONST; ++d) {
                const float rw = scale * rj_weighted[d];
                atomicAdd(&gQbh[i0*D_CONST + d], rw * sh_Sk[sh_k_off + d]);
                atomicAdd(&gSbh[k0*D_CONST + d], rw * sh_Qi[sh_i_off + d]);
            }
        }
    }
}

/**
 * R_grad_kernel - Computes gradR with Jacobian corrections.
 *
 * CORRECTION_ONLY=true:  Computes correction sum sum_r[j]
 * CORRECTION_ONLY=false: Computes gradR using precomputed corrections
 */
template<bool CORRECTION_ONLY, int BLOCK_J, int BLOCK_I, int BLOCK_K, int D_CONST, int REG_CAP = D_CONST>
__global__ void __launch_bounds__(256, 1) R_grad_kernel(
    const bf16* __restrict__ Q, const bf16* __restrict__ R, const bf16* __restrict__ S,
    const bf16* __restrict__ Vq1, const bf16* __restrict__ Vq2,
    const bf16* __restrict__ Vr1, const bf16* __restrict__ Vr2,
    const bf16* __restrict__ Vs1, const bf16* __restrict__ Vs2,
    const bf16* __restrict__ grad_Yq,
    const bf16* __restrict__ grad_Yr,
    const bf16* __restrict__ grad_Ys,
    const bf16* __restrict__ grad_Yq_,
    const bf16* __restrict__ grad_Yr_,
    const bf16* __restrict__ grad_Ys_,
    const float* __restrict__ m_i, const float* __restrict__ l_i,
    const float* __restrict__ m_j, const float* __restrict__ l_j,
    const float* __restrict__ m_k, const float* __restrict__ l_k,
    float* __restrict__ sum_q, float* __restrict__ sum_r, float* __restrict__ sum_s,
    float* __restrict__ gradR,
    const bool* __restrict__ mask,
    int N, int H, float scale)
{
    const int j0 = blockIdx.x * BLOCK_J + threadIdx.x;
    const int k0 = blockIdx.y * BLOCK_K + threadIdx.y;
    const int bh = blockIdx.z;
    const int b = bh / H;
    const bool valid = (j0 < N && k0 < N);

    // Per (B,H) base pointers
    const int64_t stride_BH = (int64_t)N * D_CONST;
    const bf16* Qbh    = Q   + bh * stride_BH;
    const bf16* Rbh    = R   + bh * stride_BH;
    const bf16* Sbh    = S   + bh * stride_BH;
    const bf16* Vq1bh  = Vq1 + bh * stride_BH;
    const bf16* Vq2bh  = Vq2 + bh * stride_BH;
    const bf16* Vr1bh  = Vr1 + bh * stride_BH;
    const bf16* Vr2bh  = Vr2 + bh * stride_BH;
    const bf16* Vs1bh  = Vs1 + bh * stride_BH;
    const bf16* Vs2bh  = Vs2 + bh * stride_BH;
    const bf16* gYqbh  = grad_Yq + bh * stride_BH;
    const bf16* gYrbh  = grad_Yr + bh * stride_BH;
    const bf16* gYsbh  = grad_Ys + bh * stride_BH;
    const bf16* gYq2bh = grad_Yq_ + bh * stride_BH;
    const bf16* gYr2bh = grad_Yr_ + bh * stride_BH;
    const bf16* gYs2bh = grad_Ys_ + bh * stride_BH;
    const float* miBH  = m_i + bh * N;
    const float* liBH  = l_i + bh * N;
    const float* mjBH  = m_j + bh * N;
    const float* ljBH  = l_j + bh * N;
    const float* mkBH  = m_k + bh * N;
    const float* lkBH  = l_k + bh * N;
    float* sum_rBH     = sum_r + bh * N;

    constexpr int D_PAD = D_CONST + 1;  // bank-conflict-free stride
    extern __shared__ float shmem[];

    // Persistent j/k data (padded stride)
    float* sh_Rj   = shmem;
    float* sh_Vr1j = sh_Rj   + BLOCK_J * D_PAD;
    float* sh_Vr2j = sh_Vr1j + BLOCK_J * D_PAD;
    float* sh_dYj  = sh_Vr2j + BLOCK_J * D_PAD;
    float* sh_dYj2 = sh_dYj  + BLOCK_J * D_PAD;

    float* sh_Sk   = sh_dYj2 + BLOCK_J * D_PAD;
    float* sh_Vs1k = sh_Sk   + BLOCK_K * D_PAD;
    float* sh_Vs2k = sh_Vs1k + BLOCK_K * D_PAD;
    float* sh_dYk  = sh_Vs2k + BLOCK_K * D_PAD;
    float* sh_dYk2 = sh_dYk  + BLOCK_K * D_PAD;

    // I-tile data (streamed). Stride is D_CONST (not D_PAD) so that
    // 4-element rows are 16-byte aligned and inner-loop reads can use
    // float4 LDS.128. Bank conflicts are handled at the cooperative-store
    // site by the linear-tid mapping below (warp covers one row × 32 cols).

    float* sh_Q    = sh_dYk2 + BLOCK_K * D_PAD;
    float* sh_Vq1  = sh_Q    + BLOCK_I * D_CONST;
    float* sh_Vq2  = sh_Vq1  + BLOCK_I * D_CONST;
    float* sh_dYi  = sh_Vq2  + BLOCK_I * D_CONST;
    float* sh_dYi2 = sh_dYi  + BLOCK_I * D_CONST;

    float* sh_mi   = sh_dYi2 + BLOCK_I * D_CONST;
    float* sh_li   = sh_mi   + BLOCK_I;
    float* sh_sumq = sh_li   + BLOCK_I;

    // Cooperative load of j-indexed and k-indexed data into shared memory
    {
        const int tid = threadIdx.x + threadIdx.y * BLOCK_J;
        const int nThreads = BLOCK_J * BLOCK_K;
        for (int idx = tid; idx < BLOCK_J * D_CONST; idx += nThreads) {
            const int jj = idx / D_CONST;
            const int dd = idx % D_CONST;
            const int jGlob = blockIdx.x * BLOCK_J + jj;
            if (jGlob < N) {
                sh_Rj  [jj * D_PAD + dd] = bf2f(Rbh[jGlob * D_CONST + dd]);
                sh_Vr1j[jj * D_PAD + dd] = bf2f(Vr1bh[jGlob * D_CONST + dd]);
                sh_Vr2j[jj * D_PAD + dd] = bf2f(Vr2bh[jGlob * D_CONST + dd]);
                sh_dYj [jj * D_PAD + dd] = bf2f(gYrbh[jGlob * D_CONST + dd]);
                sh_dYj2[jj * D_PAD + dd] = bf2f(gYr2bh[jGlob * D_CONST + dd]);
            } else {
                sh_Rj  [jj * D_PAD + dd] = 0.0f;
                sh_Vr1j[jj * D_PAD + dd] = 0.0f;
                sh_Vr2j[jj * D_PAD + dd] = 0.0f;
                sh_dYj [jj * D_PAD + dd] = 0.0f;
                sh_dYj2[jj * D_PAD + dd] = 0.0f;
            }
        }
        for (int idx = tid; idx < BLOCK_K * D_CONST; idx += nThreads) {
            const int kk = idx / D_CONST;
            const int dd = idx % D_CONST;
            const int kGlob = blockIdx.y * BLOCK_K + kk;
            if (kGlob < N) {
                sh_Sk  [kk * D_PAD + dd] = bf2f(Sbh[kGlob * D_CONST + dd]);
                sh_Vs1k[kk * D_PAD + dd] = bf2f(Vs1bh[kGlob * D_CONST + dd]);
                sh_Vs2k[kk * D_PAD + dd] = bf2f(Vs2bh[kGlob * D_CONST + dd]);
                sh_dYk [kk * D_PAD + dd] = bf2f(gYsbh[kGlob * D_CONST + dd]);
                sh_dYk2[kk * D_PAD + dd] = bf2f(gYs2bh[kGlob * D_CONST + dd]);
            } else {
                sh_Sk  [kk * D_PAD + dd] = 0.0f;
                sh_Vs1k[kk * D_PAD + dd] = 0.0f;
                sh_Vs2k[kk * D_PAD + dd] = 0.0f;
                sh_dYk [kk * D_PAD + dd] = 0.0f;
                sh_dYk2[kk * D_PAD + dd] = 0.0f;
            }
        }
    }

    float mj = 0.0f, lj = 1.0f, mk = 0.0f, lk = 1.0f;
    if (valid) {
        mj = mjBH[j0];  lj = ljBH[j0];
        mk = mkBH[k0];  lk = lkBH[k0];
    }

    __syncthreads();

    float reg_sum_r = 0.0f;
    float sumRj = 0.0f, sumSk = 0.0f;
    float grad_acc[REG_CAP];
    if constexpr (!CORRECTION_ONLY) {
        if (valid) {
            sumRj = sum_rBH[j0];
            sumSk = (sum_s + (int64_t)bh * N)[k0];
        }
        for (int d = 0; d < REG_CAP; ++d) grad_acc[d] = 0.0f;
    }

    const int sh_j_off = threadIdx.x * D_PAD;
    const int sh_k_off = threadIdx.y * D_PAD;

    // Stream i-tiles through shared memory
    const int tid_l       = threadIdx.x + threadIdx.y * BLOCK_J;
    const int nThreads_l  = BLOCK_J * BLOCK_K;
    for (int iBase = 0; iBase < N; iBase += BLOCK_I) {
        // Cooperative load of i-tile. We keep the linear tid mapping and pack
        // stores as float4 to lower shared-store instruction pressure.
        constexpr int D_VEC = 4;
        const int nVecPerRow = D_CONST / D_VEC;
        for (int idx4 = tid_l; idx4 < BLOCK_I * nVecPerRow; idx4 += nThreads_l) {
            const int ii    = idx4 / nVecPerRow;
            const int vv    = idx4 % nVecPerRow;
            const int dd    = vv * D_VEC;
            const int iGlob = iBase + ii;
            float4 q4, vq14, vq24, dyi4;
            if (iGlob < N) {
                q4   = make_float4(
                    bf2f(Qbh[iGlob*D_CONST + dd + 0]),
                    bf2f(Qbh[iGlob*D_CONST + dd + 1]),
                    bf2f(Qbh[iGlob*D_CONST + dd + 2]),
                    bf2f(Qbh[iGlob*D_CONST + dd + 3]));
                vq14 = make_float4(
                    bf2f(Vq1bh[iGlob*D_CONST + dd + 0]),
                    bf2f(Vq1bh[iGlob*D_CONST + dd + 1]),
                    bf2f(Vq1bh[iGlob*D_CONST + dd + 2]),
                    bf2f(Vq1bh[iGlob*D_CONST + dd + 3]));
                vq24 = make_float4(
                    bf2f(Vq2bh[iGlob*D_CONST + dd + 0]),
                    bf2f(Vq2bh[iGlob*D_CONST + dd + 1]),
                    bf2f(Vq2bh[iGlob*D_CONST + dd + 2]),
                    bf2f(Vq2bh[iGlob*D_CONST + dd + 3]));
                dyi4 = make_float4(
                    bf2f(gYqbh[iGlob*D_CONST + dd + 0]),
                    bf2f(gYqbh[iGlob*D_CONST + dd + 1]),
                    bf2f(gYqbh[iGlob*D_CONST + dd + 2]),
                    bf2f(gYqbh[iGlob*D_CONST + dd + 3]));
                const float4 dyi24 = make_float4(
                    bf2f(gYq2bh[iGlob*D_CONST + dd + 0]),
                    bf2f(gYq2bh[iGlob*D_CONST + dd + 1]),
                    bf2f(gYq2bh[iGlob*D_CONST + dd + 2]),
                    bf2f(gYq2bh[iGlob*D_CONST + dd + 3]));
                *reinterpret_cast<float4*>(&sh_dYi2[ii*D_CONST + dd]) = dyi24;
            } else {
                q4   = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                vq14 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                vq24 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                dyi4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                *reinterpret_cast<float4*>(&sh_dYi2[ii*D_CONST + dd]) = dyi4;
            }
            *reinterpret_cast<float4*>(&sh_Q  [ii*D_CONST + dd]) = q4;
            *reinterpret_cast<float4*>(&sh_Vq1[ii*D_CONST + dd]) = vq14;
            *reinterpret_cast<float4*>(&sh_Vq2[ii*D_CONST + dd]) = vq24;
            *reinterpret_cast<float4*>(&sh_dYi[ii*D_CONST + dd]) = dyi4;
        }
        if (tid_l < BLOCK_I) {
            const int iGlob = iBase + tid_l;
            if (iGlob < N) {
                sh_mi[tid_l] = miBH[iGlob];
                sh_li[tid_l] = liBH[iGlob];
                if constexpr (!CORRECTION_ONLY) {
                    sh_sumq[tid_l] = (sum_q + (int64_t)bh * N)[iGlob];
                }
            } else {
                sh_mi[tid_l] = 0.0f;
                sh_li[tid_l] = 1.0f;  // avoid div-by-zero in OOB rows
                if constexpr (!CORRECTION_ONLY) {
                    sh_sumq[tid_l] = 0.0f;
                }
            }
        }
        __syncthreads();

        // D-tiled dot products with i sub-tiling.
        // Mirrors the QS kernel strategy: precompute thread-invariant j/k products
        // once per D_TILE and reuse them across several i rows.
        constexpr int I_SUB  = 4;
        constexpr int D_TILE = 4;
        for (int iSub = 0; iSub < BLOCK_I && (iBase + iSub) < N; iSub += I_SUB) {
            float dot_i[I_SUB], d1_i[I_SUB], d2_i[I_SUB], d3_i[I_SUB];
            float d4_i[I_SUB], d5_i[I_SUB], d6_i[I_SUB];
            #pragma unroll
            for (int ii = 0; ii < I_SUB; ++ii) {
                dot_i[ii] = 0.f; d1_i[ii] = 0.f; d2_i[ii] = 0.f; d3_i[ii] = 0.f;
                d4_i[ii] = 0.f; d5_i[ii] = 0.f; d6_i[ii] = 0.f;
            }

            for (int d_base = 0; d_base < D_CONST; d_base += D_TILE) {
                float p_dot[D_TILE], p_d1[D_TILE], p_d2[D_TILE], p_d3[D_TILE];
                float p_d4[D_TILE], p_d5[D_TILE], p_d6[D_TILE];
                #pragma unroll
                for (int dd = 0; dd < D_TILE; ++dd) {
                    const int d = d_base + dd;
                    const float rj   = sh_Rj  [sh_j_off + d];
                    const float sk   = sh_Sk  [sh_k_off + d];
                    const float vr1j = sh_Vr1j[sh_j_off + d];
                    const float vr2j = sh_Vr2j[sh_j_off + d];
                    const float vs1k = sh_Vs1k[sh_k_off + d];
                    const float vs2k = sh_Vs2k[sh_k_off + d];
                    const float dyj  = sh_dYj [sh_j_off + d];
                    const float dyj2 = sh_dYj2[sh_j_off + d];
                    const float dyk  = sh_dYk [sh_k_off + d];
                    const float dyk2 = sh_dYk2[sh_k_off + d];

                    p_dot[dd] = rj * sk;
                    p_d1[dd]  = vr1j * vs1k;
                    p_d2[dd]  = dyj * vs1k;
                    p_d3[dd]  = dyk * vr1j;
                    p_d4[dd]  = vr2j * vs2k;
                    p_d5[dd]  = dyj2 * vs2k;
                    p_d6[dd]  = dyk2 * vr2j;
                }

                #pragma unroll
                for (int ii = 0; ii < I_SUB; ++ii) {
                    const int iOff = iSub + ii;
                    if (iBase + iOff >= N) break;
                    // i-tile is stored at stride D_CONST (not D_PAD) so each
                    // 4-float slice is 16-byte aligned: one LDS.128 per array.
                    const int iRow = iOff * D_CONST + d_base;
                    const float4 qi4  = *reinterpret_cast<const float4*>(&sh_Q  [iRow]);
                    const float4 vq14 = *reinterpret_cast<const float4*>(&sh_Vq1[iRow]);
                    const float4 vq24 = *reinterpret_cast<const float4*>(&sh_Vq2[iRow]);
                    const float4 dyi4 = *reinterpret_cast<const float4*>(&sh_dYi[iRow]);
                    const float4 dyi24 = *reinterpret_cast<const float4*>(&sh_dYi2[iRow]);
                    const float qi[4]  = { qi4.x,  qi4.y,  qi4.z,  qi4.w  };
                    const float vq1[4] = { vq14.x, vq14.y, vq14.z, vq14.w };
                    const float vq2[4] = { vq24.x, vq24.y, vq24.z, vq24.w };
                    const float dyi[4] = { dyi4.x, dyi4.y, dyi4.z, dyi4.w };
                    const float dyi2[4] = { dyi24.x, dyi24.y, dyi24.z, dyi24.w };
                    #pragma unroll
                    for (int dd = 0; dd < D_TILE; ++dd) {
                        dot_i[ii] += qi[dd]  * p_dot[dd];
                        if constexpr (!CORRECTION_ONLY) d1_i[ii] += dyi[dd] * p_d1[dd];
                        d2_i[ii] += vq1[dd] * p_d2[dd];
                        if constexpr (!CORRECTION_ONLY) d3_i[ii] += vq1[dd] * p_d3[dd];
                        d4_i[ii] += dyi2[dd] * p_d4[dd];
                        if constexpr (!CORRECTION_ONLY) d5_i[ii] += vq2[dd] * p_d5[dd];
                        d6_i[ii] += vq2[dd] * p_d6[dd];
                    }
                }
            }

            #pragma unroll
            for (int ii = 0; ii < I_SUB; ++ii) {
                const int iOff = iSub + ii;
                if (iBase + iOff >= N) break;
                const int iGlob = iBase + iOff;
                const float mi = sh_mi[iOff];
                const float li = sh_li[iOff];

                const float logits = dot_i[ii] * scale;
                const bool aq_valid = valid && mask_pair_allowed(mask, N, b, iGlob, j0, k0);
                const bool ar_valid = valid && mask_pair_allowed(mask, N, b, j0, iGlob, k0);
                const bool as_valid = valid && mask_pair_allowed(mask, N, b, k0, iGlob, j0);
                const float Aq = aq_valid ? (__expf(fminf(logits - mi, EXP_CLIP)) / fmaxf(li, DENOM_EPS)) : 0.0f;
                const float Ar = ar_valid ? (__expf(fminf(logits - mj, EXP_CLIP)) / fmaxf(lj, DENOM_EPS)) : 0.0f;
                const float As = as_valid ? (__expf(fminf(logits - mk, EXP_CLIP)) / fmaxf(lk, DENOM_EPS)) : 0.0f;
                const float gAr = d2_i[ii] + d4_i[ii] * As + d6_i[ii] * Aq;

                if constexpr (CORRECTION_ONLY) {
                    reg_sum_r += gAr * Ar;
                } else {
                    const float sumQi = sh_sumq[iOff];
                    const float gAq = d1_i[ii] + d5_i[ii] * As + d6_i[ii] * Ar;
                    const float gAs = d3_i[ii] + d4_i[ii] * Ar + d5_i[ii] * Aq;
                    const float grad_A = (gAq - sumQi) * Aq
                                       + (gAr - sumRj) * Ar
                                       + (gAs - sumSk) * As;
                    // sh_Q stride D_CONST → float4-aligned; sh_Sk stride D_PAD
                    // (=D+1) is not 16-byte aligned at sh_k_off for ty>0, so
                    // it stays scalar.
                    const int iRow = iOff * D_CONST;
                    #pragma unroll
                    for (int d = 0; d < D_CONST; d += 4) {
                        const float4 qi4 = *reinterpret_cast<const float4*>(&sh_Q[iRow + d]);
                        grad_acc[d+0] += grad_A * qi4.x * sh_Sk[sh_k_off + d + 0];
                        grad_acc[d+1] += grad_A * qi4.y * sh_Sk[sh_k_off + d + 1];
                        grad_acc[d+2] += grad_A * qi4.z * sh_Sk[sh_k_off + d + 2];
                        grad_acc[d+3] += grad_A * qi4.w * sh_Sk[sh_k_off + d + 3];
                    }
                }
            }
        }
        __syncthreads();
    }

    // ======== EPILOGUE ========
    if constexpr (CORRECTION_ONLY) {
        // Block reduction for sum_r[j]: reduce reg_sum_r across k-dim (threadIdx.y)
        float* reduce_buf = shmem;  // reuse shared memory (i-tile data done)

        // Transposed [k][j] layout makes warp-contiguous x-lanes hit distinct banks.
        const int reduce_idx = threadIdx.y * BLOCK_J + threadIdx.x;
        reduce_buf[reduce_idx] = valid ? reg_sum_r : 0.0f;
        __syncthreads();
        for (int s = BLOCK_K / 2; s > 0; s >>= 1) {
            if (threadIdx.y < s) {
                reduce_buf[reduce_idx] +=
                    reduce_buf[(threadIdx.y + s) * BLOCK_J + threadIdx.x];
            }
            __syncthreads();
        }
        if (threadIdx.y == 0 && j0 < N)
            atomicAdd(&sum_rBH[j0], reduce_buf[threadIdx.x]);
    } else {
        // Write result (atomic due to k-dimension overlap)
        float* gRbh = gradR + bh * stride_BH;
        if (valid) {
            for (int d = 0; d < D_CONST; ++d)
                atomicAdd(&gRbh[j0*D_CONST + d], scale * grad_acc[d]);
        }
    }
}



// =============================================================================
// D-dispatch: routes to D_CONST=32 or D_CONST=64 template instantiation
// =============================================================================
#define DISPATCH_D(D_VAL, ...) \
  [&] { \
    if ((D_VAL) == 16)      { constexpr int D_TMPL = 16; __VA_ARGS__; } \
    else if ((D_VAL) == 32) { constexpr int D_TMPL = 32; __VA_ARGS__; } \
    else if ((D_VAL) == 64) { constexpr int D_TMPL = 64; __VA_ARGS__; } \
    else { TORCH_CHECK(false, "backward: unsupported D=", (D_VAL), ". Supported: 16, 32, 64"); } \
  }()

// =============================================================================
// Internal implementation that uses pre-computed softmax stats
// =============================================================================
static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
backward_impl(torch::Tensor grad_Y_q,
              torch::Tensor grad_Y_r,
              torch::Tensor grad_Y_s,
              torch::Tensor grad_Y_q_,
              torch::Tensor grad_Y_r_,
              torch::Tensor grad_Y_s_,
              torch::Tensor Q,
              torch::Tensor R,
              torch::Tensor S,
              torch::Tensor Vq_1,
              torch::Tensor Vq_2,
              torch::Tensor Vr_1,
              torch::Tensor Vr_2,
              torch::Tensor Vs_1,
              torch::Tensor Vs_2,
              torch::Tensor m_i,
              torch::Tensor l_i,
              torch::Tensor m_j,
              torch::Tensor l_j,
              torch::Tensor m_k,
              torch::Tensor l_k,
              torch::Tensor mask,
              double dropout_rate,
              torch::Tensor Y_q,
              torch::Tensor Y_r,
              torch::Tensor Y_s) {

  // ============================================================================
  // 1. EXTRACT DIMENSIONS AND CONSTANTS
  // ============================================================================
  TORCH_CHECK(Q.scalar_type() == at::kBFloat16, "backward expects bfloat16 activations.");
  TORCH_CHECK(R.scalar_type() == at::kBFloat16, "backward expects bfloat16 activations.");
  TORCH_CHECK(S.scalar_type() == at::kBFloat16, "backward expects bfloat16 activations.");
  TORCH_CHECK(Vq_1.scalar_type() == at::kBFloat16, "backward expects bfloat16 activations.");
  TORCH_CHECK(Vq_2.scalar_type() == at::kBFloat16, "backward expects bfloat16 activations.");
  TORCH_CHECK(Vr_1.scalar_type() == at::kBFloat16, "backward expects bfloat16 activations.");
  TORCH_CHECK(Vr_2.scalar_type() == at::kBFloat16, "backward expects bfloat16 activations.");
  TORCH_CHECK(Vs_1.scalar_type() == at::kBFloat16, "backward expects bfloat16 activations.");
  TORCH_CHECK(Vs_2.scalar_type() == at::kBFloat16, "backward expects bfloat16 activations.");
  TORCH_CHECK(grad_Y_q.scalar_type() == at::kBFloat16, "backward expects bfloat16 grad_Y_q.");
  TORCH_CHECK(grad_Y_r.scalar_type() == at::kBFloat16, "backward expects bfloat16 grad_Y_r.");
  TORCH_CHECK(grad_Y_s.scalar_type() == at::kBFloat16, "backward expects bfloat16 grad_Y_s.");
  TORCH_CHECK(grad_Y_q_.scalar_type() == at::kBFloat16, "backward expects bfloat16 grad_Y_q_.");
  TORCH_CHECK(grad_Y_r_.scalar_type() == at::kBFloat16, "backward expects bfloat16 grad_Y_r_.");
  TORCH_CHECK(grad_Y_s_.scalar_type() == at::kBFloat16, "backward expects bfloat16 grad_Y_s_.");
  TORCH_CHECK(m_i.scalar_type() == at::kFloat && l_i.scalar_type() == at::kFloat &&
              m_j.scalar_type() == at::kFloat && l_j.scalar_type() == at::kFloat &&
              m_k.scalar_type() == at::kFloat && l_k.scalar_type() == at::kFloat,
              "backward expects FP32 softmax stats.");

  const int B = Q.size(0);
  const int H = Q.size(1);
  const int N = Q.size(2); //i think N and I/J/K are aliases, deal with later
  const int I = Q.size(2);
  const int J = R.size(2);
  const int K = S.size(2);
  const int D = Q.size(3);
  const float scale = 1.0f / sqrtf(static_cast<float>(D));
  const bool use_mask = mask.defined() && mask.numel() > 0;
  if (use_mask) {
      TORCH_CHECK(mask.scalar_type() == at::kBool, "backward mask must be bool");
      TORCH_CHECK(mask.is_cuda(), "backward mask must be on CUDA device");
      TORCH_CHECK(mask.dim() == 3, "backward mask must have shape [B, N, N]");
      TORCH_CHECK(mask.size(0) == B, "backward mask batch dim mismatch");
      TORCH_CHECK(mask.size(1) == N && mask.size(2) == N,
          "backward mask shape must be [B, N, N] with N matching padded sequence length");
  }
  const bool* mask_ptr = use_mask ? mask.data_ptr<bool>() : nullptr;

  // ============================================================================
  // 2. ALLOCATE GRADIENT TENSORS
  // ============================================================================
  auto options_fp32 = Q.options().dtype(at::kFloat);
  auto grad_Q = torch::zeros({B, H, I, D}, options_fp32);
  auto grad_R = torch::zeros({B, H, J, D}, options_fp32);
  auto grad_S = torch::zeros({B, H, K, D}, options_fp32);
  auto grad_Vq_1 = torch::zeros({B, H, I, D}, options_fp32);
  auto grad_Vq_2 = torch::zeros({B, H, I, D}, options_fp32);
  auto grad_Vr_1 = torch::zeros({B, H, J, D}, options_fp32);
  auto grad_Vr_2 = torch::zeros({B, H, J, D}, options_fp32);
  auto grad_Vs_1 = torch::zeros({B, H, K, D}, options_fp32);
  auto grad_Vs_2 = torch::zeros({B, H, K, D}, options_fp32);

  auto sum_q = torch::zeros({B, H, N}, options_fp32);
  auto sum_r = torch::zeros({B, H, N}, options_fp32);
  auto sum_s = torch::zeros({B, H, N}, options_fp32);

  // ============================================================================
  // 2b. TENSOR-CORE FAST PATH (gather-only; see Bwd_gather_tc above)
  // ============================================================================
  // Requires the forward outputs Y_q/Y_r/Y_s (for the collapsed correction
  // sums) and all-zero scatter cotangents. Anything else takes the scalar
  // path below unchanged. Disable with ATT3_BWD_TC=0.
  if (D == 64 && I == J && J == K && (N % 16 == 0)
      && Y_q.defined() && Y_r.defined() && Y_s.defined()) {
    static const int max_smem_optin = []() {
        int dev = 0, major = 0, v = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        if (major < 8) return 0;
        cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        return v;
    }();
    const int K_pad = ceil_div(N, BTC_BK) * BTC_BK;
    constexpr int TC_DPAD = 64 + 8;
    // Masked runs add two resident bit-packed [K_pad+1][K_pad/32] mask tiles
    // (row-major and bit-transposed, one spare zero row each for pad rows) plus
    // the per-col anchor inv-l floats — 17.4 KB at N=256.
    const size_t smem_mask = use_mask
        ? sizeof(uint32_t) * 2 * (size_t)(K_pad + 1) * (K_pad / 32)
          + sizeof(float) * ((size_t)K_pad + BTC_BJ)
        : 0;
    const size_t smem_tc =
        sizeof(bf16) * ((size_t)3 * BTC_BJ * TC_DPAD + (size_t)3 * K_pad * TC_DPAD) +
        sizeof(float) * (3 * 64 + (size_t)3 * K_pad + 3 * BTC_BJ
                         + BTC_WARPS * 2 * 64 + 2 * 64)
        + smem_mask;
    if (att3_tc::state().bwd_enabled && smem_tc <= (size_t)max_smem_optin) {
      // Single host round-trip for the gate.
      const bool scatter_active =
          (grad_Y_q_.ne(0).any() | grad_Y_r_.ne(0).any() | grad_Y_s_.ne(0).any())
              .item<bool>();
      if (!scatter_active) {
        // With no scatter cross terms every correction sum collapses to
        // rowsum(dY o Y) — the FA2 shortcut (probe-validated).
        sum_q = (grad_Y_q.to(at::kFloat) * Y_q.to(at::kFloat)).sum(-1).contiguous();
        sum_r = (grad_Y_r.to(at::kFloat) * Y_r.to(at::kFloat)).sum(-1).contiguous();
        sum_s = (grad_Y_s.to(at::kFloat) * Y_s.to(at::kFloat)).sum(-1).contiguous();

        auto* tc_kernel = use_mask ? Bwd_gather_tc<64, true>
                                   : Bwd_gather_tc<64, false>;
        static size_t smem_attr_set[2] = {0, 0};
        const int variant = use_mask ? 1 : 0;
        if (smem_tc > smem_attr_set[variant]) {
            AT_CUDA_CHECK(cudaFuncSetAttribute(
                tc_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_tc));
            smem_attr_set[variant] = smem_tc;
        }
        auto stream = at::cuda::getCurrentCUDAStream();
        const dim3 grid_tc(N, H, B), block_tc(BTC_WARPS * 32);
        auto bp = [](const torch::Tensor& t) {
            return reinterpret_cast<const bf16*>(t.data_ptr<at::BFloat16>());
        };
        auto launch = [&](const torch::Tensor& Xa, const torch::Tensor& Va, const torch::Tensor& gYa,
                          const torch::Tensor& Xr, const torch::Tensor& Vr, const torch::Tensor& gYr,
                          const torch::Tensor& Xc, const torch::Tensor& Vc, const torch::Tensor& gYc,
                          const torch::Tensor& ma, const torch::Tensor& la, const torch::Tensor& sa,
                          const torch::Tensor& mr, const torch::Tensor& lr, const torch::Tensor& sr,
                          const torch::Tensor& mc, const torch::Tensor& lc, const torch::Tensor& sc,
                          torch::Tensor& gX, torch::Tensor& gV) {
            tc_kernel<<<grid_tc, block_tc, smem_tc, stream>>>(
                bp(Xa), bp(Va), bp(gYa), bp(Xr), bp(Vr), bp(gYr), bp(Xc), bp(Vc), bp(gYc),
                ma.data_ptr<float>(), la.data_ptr<float>(), sa.data_ptr<float>(),
                mr.data_ptr<float>(), lr.data_ptr<float>(), sr.data_ptr<float>(),
                mc.data_ptr<float>(), lc.data_ptr<float>(), sc.data_ptr<float>(),
                gX.data_ptr<float>(), gV.data_ptr<float>(),
                mask_ptr, H, N, K_pad, scale);
            ++att3_tc::state().bwd_launches;
        };
        // Role table (anchor / rows / cols), one launch per gradient family:
        launch(Q, Vq_1, grad_Y_q,  R, Vr_1, grad_Y_r,  S, Vs_1, grad_Y_s,
               m_i, l_i, sum_q,  m_j, l_j, sum_r,  m_k, l_k, sum_s,
               grad_Q, grad_Vq_1);
        launch(R, Vr_1, grad_Y_r,  Q, Vq_1, grad_Y_q,  S, Vs_1, grad_Y_s,
               m_j, l_j, sum_r,  m_i, l_i, sum_q,  m_k, l_k, sum_s,
               grad_R, grad_Vr_1);
        launch(S, Vs_1, grad_Y_s,  Q, Vq_1, grad_Y_q,  R, Vr_1, grad_Y_r,
               m_k, l_k, sum_s,  m_i, l_i, sum_q,  m_j, l_j, sum_r,
               grad_S, grad_Vs_1);
        AT_CUDA_CHECK(cudaGetLastError());
        // No device sync: everything above is ordered on the current stream,
        // so the usual PyTorch stream semantics cover consumers.

        // Scatter value grads are identically zero here (their cotangents are).
        return std::make_tuple(
            grad_Q.to(at::kBFloat16),
            grad_R.to(at::kBFloat16),
            grad_S.to(at::kBFloat16),
            grad_Vq_1.to(at::kBFloat16),
            grad_Vq_2.to(at::kBFloat16),
            grad_Vr_1.to(at::kBFloat16),
            grad_Vr_2.to(at::kBFloat16),
            grad_Vs_1.to(at::kBFloat16),
            grad_Vs_2.to(at::kBFloat16));
      }
    }
  }

  // ============================================================================
  // 3. COMPUTE grad_{Vq,Vr,Vs}_1 (GATHER-GRAD KERNELS)
  // ============================================================================
  DISPATCH_D(D, {
    constexpr int TI = T_I;
    constexpr int TK = T_K;
    dim3 block_dim(TI, TK);
    dim3 grid_dim((N + TI - 1) / TI, (N + TK - 1) / TK, B * H);

    // Role table (out / reg / loop), one launch per V_1 gradient; the last arg
    // puts out on thread y, which the grad_Vs permutation wants. Gather kernels
    // use static shared memory sized by D_TMPL.
    auto launch = [&](const at::Tensor& X_out, const at::Tensor& X_reg,
                      const at::Tensor& X_loop, const at::Tensor& V_reg,
                      const at::Tensor& V_loop, const at::Tensor& gY_loop,
                      const at::Tensor& gY_reg, const at::Tensor& m_loop,
                      const at::Tensor& l_loop, const at::Tensor& m_reg,
                      const at::Tensor& l_reg, at::Tensor& gradV_out,
                      bool out_is_y) {
      auto* kernel = out_is_y ? V_gather_grad<D_TMPL, true>
                              : V_gather_grad<D_TMPL, false>;
      kernel<<<grid_dim, block_dim, 0, at::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<const bf16*>(X_out.data_ptr<at::BFloat16>()),
          reinterpret_cast<const bf16*>(X_reg.data_ptr<at::BFloat16>()),
          reinterpret_cast<const bf16*>(X_loop.data_ptr<at::BFloat16>()),
          reinterpret_cast<const bf16*>(V_reg.data_ptr<at::BFloat16>()),
          reinterpret_cast<const bf16*>(V_loop.data_ptr<at::BFloat16>()),
          reinterpret_cast<const bf16*>(gY_loop.data_ptr<at::BFloat16>()),
          reinterpret_cast<const bf16*>(gY_reg.data_ptr<at::BFloat16>()),
          m_loop.data_ptr<float>(), l_loop.data_ptr<float>(),
          m_reg.data_ptr<float>(), l_reg.data_ptr<float>(),
          gradV_out.data_ptr<float>(),
          mask_ptr, N, H, scale);
    };

    launch(Q, S, R,  Vs_1, Vr_1,  grad_Y_r, grad_Y_s,  m_j, l_j,  m_k, l_k,  grad_Vq_1, false);
    launch(R, S, Q,  Vs_1, Vq_1,  grad_Y_q, grad_Y_s,  m_i, l_i,  m_k, l_k,  grad_Vr_1, false);
    launch(S, Q, R,  Vq_1, Vr_1,  grad_Y_r, grad_Y_q,  m_j, l_j,  m_i, l_i,  grad_Vs_1, true);
  });

  // ============================================================================
  // 4. COMPUTE grad_{Vq,Vr,Vs}_2 (SCATTER-GRAD KERNELS)
  // ============================================================================
  DISPATCH_D(D, {
    constexpr int TI = T_I;
    constexpr int TK = T_K;
    dim3 block_dim(TI, TK);
    dim3 grid_dim((N + TI - 1) / TI, (N + TK - 1) / TK, B * H);

    const size_t shmem_scatter = 3 * T_J * D_TMPL * sizeof(float) + 2 * T_J * sizeof(float);

    // Same role table as the gather grads, but the out mode's stats are consumed
    // as well, so all six m/l tensors are passed.
    auto launch = [&](const at::Tensor& X_out, const at::Tensor& X_reg,
                      const at::Tensor& X_loop, const at::Tensor& V_reg,
                      const at::Tensor& V_loop, const at::Tensor& gY_loop,
                      const at::Tensor& gY_reg, const at::Tensor& m_out,
                      const at::Tensor& l_out, const at::Tensor& m_loop,
                      const at::Tensor& l_loop, const at::Tensor& m_reg,
                      const at::Tensor& l_reg, at::Tensor& gradV_out,
                      bool out_is_y) {
      auto* kernel = out_is_y ? V_scatter_grad<D_TMPL, true>
                              : V_scatter_grad<D_TMPL, false>;
      kernel<<<grid_dim, block_dim, shmem_scatter, at::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<const bf16*>(X_out.data_ptr<at::BFloat16>()),
          reinterpret_cast<const bf16*>(X_reg.data_ptr<at::BFloat16>()),
          reinterpret_cast<const bf16*>(X_loop.data_ptr<at::BFloat16>()),
          reinterpret_cast<const bf16*>(V_reg.data_ptr<at::BFloat16>()),
          reinterpret_cast<const bf16*>(V_loop.data_ptr<at::BFloat16>()),
          reinterpret_cast<const bf16*>(gY_loop.data_ptr<at::BFloat16>()),
          reinterpret_cast<const bf16*>(gY_reg.data_ptr<at::BFloat16>()),
          m_out.data_ptr<float>(), l_out.data_ptr<float>(),
          m_loop.data_ptr<float>(), l_loop.data_ptr<float>(),
          m_reg.data_ptr<float>(), l_reg.data_ptr<float>(),
          gradV_out.data_ptr<float>(),
          mask_ptr, N, H, scale);
    };

    launch(Q, S, R,  Vs_2, Vr_2,  grad_Y_r_, grad_Y_s_,
           m_i, l_i,  m_j, l_j,  m_k, l_k,  grad_Vq_2, false);
    launch(R, S, Q,  Vs_2, Vq_2,  grad_Y_q_, grad_Y_s_,
           m_j, l_j,  m_i, l_i,  m_k, l_k,  grad_Vr_2, false);
    launch(S, Q, R,  Vq_2, Vr_2,  grad_Y_r_, grad_Y_q_,
           m_k, l_k,  m_j, l_j,  m_i, l_i,  grad_Vs_2, true);
  });
  AT_CUDA_CHECK(cudaGetLastError());


  // ===========================================================================
  // 5. JACOBIAN CORRECTIONS + 6. GRAD Q/S/R
  //    All dispatched through D template
  // ============================================================================
  DISPATCH_D(D, {
    // Phase 1: Correction sums: sum_q, sum_r, sum_s
    {
      constexpr int corrI = 8;
      constexpr int corrK = 8;
      constexpr int corrJ = 16;

      dim3 block_qs(corrI, corrK);
      dim3 grid_qs((N + corrI - 1) / corrI,
                   (N + corrK - 1) / corrK,
                   B * H);

      constexpr int D_PAD_c = D_TMPL + 1;
      const size_t shmem_corr_qs =
          5 * corrI * D_PAD_c * sizeof(float) +
          5 * corrK * D_PAD_c * sizeof(float) +
          5 * corrJ * D_TMPL * sizeof(float) +
          3 * corrJ * sizeof(float);

      cudaFuncSetAttribute(
          QS_grad_kernel<true, corrI, corrJ, corrK, D_TMPL>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          shmem_corr_qs);

      QS_grad_kernel<true, corrI, corrJ, corrK, D_TMPL>
          <<<grid_qs, block_qs, shmem_corr_qs, at::cuda::getCurrentCUDAStream()>>>(
              reinterpret_cast<const bf16*>(Q.data_ptr<at::BFloat16>()),
              reinterpret_cast<const bf16*>(R.data_ptr<at::BFloat16>()),
              reinterpret_cast<const bf16*>(S.data_ptr<at::BFloat16>()),
              reinterpret_cast<const bf16*>(Vq_1.data_ptr<at::BFloat16>()),
              reinterpret_cast<const bf16*>(Vq_2.data_ptr<at::BFloat16>()),
              reinterpret_cast<const bf16*>(Vr_1.data_ptr<at::BFloat16>()),
              reinterpret_cast<const bf16*>(Vr_2.data_ptr<at::BFloat16>()),
              reinterpret_cast<const bf16*>(Vs_1.data_ptr<at::BFloat16>()),
              reinterpret_cast<const bf16*>(Vs_2.data_ptr<at::BFloat16>()),
              reinterpret_cast<const bf16*>(grad_Y_q.data_ptr<at::BFloat16>()),
              reinterpret_cast<const bf16*>(grad_Y_r.data_ptr<at::BFloat16>()),
              reinterpret_cast<const bf16*>(grad_Y_s.data_ptr<at::BFloat16>()),
              reinterpret_cast<const bf16*>(grad_Y_q_.data_ptr<at::BFloat16>()),
              reinterpret_cast<const bf16*>(grad_Y_r_.data_ptr<at::BFloat16>()),
              reinterpret_cast<const bf16*>(grad_Y_s_.data_ptr<at::BFloat16>()),
              m_i.data_ptr<float>(),
              l_i.data_ptr<float>(),
              m_j.data_ptr<float>(),
              l_j.data_ptr<float>(),
              m_k.data_ptr<float>(),
              l_k.data_ptr<float>(),
              sum_q.data_ptr<float>(),
              sum_r.data_ptr<float>(),
              sum_s.data_ptr<float>(),
              nullptr,  // gradQ not used in correction mode
              nullptr,  // gradS not used in correction mode
              mask_ptr, N, H, scale);

      AT_CUDA_CHECK(cudaGetLastError());
    }

    // Phase 2: grad_Q + grad_S (fused)
    {
      constexpr int tileI = TILE_I;
      constexpr int tileK = TILE_K;
      constexpr int tileJ = 16;

      dim3 block_dim(tileI, tileK);
      dim3 grid_dim((N + tileI - 1) / tileI,
                    (N + tileK - 1) / tileK,
                    B * H);

      constexpr int D_PAD_g = D_TMPL + 1;
      const size_t shmem_bytes =
          5 * tileI * D_PAD_g * sizeof(float) +
          5 * tileK * D_PAD_g * sizeof(float) +
          5 * tileJ * D_TMPL * sizeof(float) +
          3 * tileJ * sizeof(float);

      cudaFuncSetAttribute(
          QS_grad_kernel<false, tileI, tileJ, tileK, D_TMPL>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          shmem_bytes);

      QS_grad_kernel<false, tileI, tileJ, tileK, D_TMPL>
          <<<grid_dim, block_dim, shmem_bytes, at::cuda::getCurrentCUDAStream()>>>(
              reinterpret_cast<const bf16*>(Q.data_ptr<at::BFloat16>()),
              reinterpret_cast<const bf16*>(R.data_ptr<at::BFloat16>()),
              reinterpret_cast<const bf16*>(S.data_ptr<at::BFloat16>()),
              reinterpret_cast<const bf16*>(Vq_1.data_ptr<at::BFloat16>()),
              reinterpret_cast<const bf16*>(Vq_2.data_ptr<at::BFloat16>()),
              reinterpret_cast<const bf16*>(Vr_1.data_ptr<at::BFloat16>()),
              reinterpret_cast<const bf16*>(Vr_2.data_ptr<at::BFloat16>()),
              reinterpret_cast<const bf16*>(Vs_1.data_ptr<at::BFloat16>()),
              reinterpret_cast<const bf16*>(Vs_2.data_ptr<at::BFloat16>()),
              reinterpret_cast<const bf16*>(grad_Y_q.data_ptr<at::BFloat16>()),
              reinterpret_cast<const bf16*>(grad_Y_r.data_ptr<at::BFloat16>()),
              reinterpret_cast<const bf16*>(grad_Y_s.data_ptr<at::BFloat16>()),
              reinterpret_cast<const bf16*>(grad_Y_q_.data_ptr<at::BFloat16>()),
              reinterpret_cast<const bf16*>(grad_Y_r_.data_ptr<at::BFloat16>()),
              reinterpret_cast<const bf16*>(grad_Y_s_.data_ptr<at::BFloat16>()),
              m_i.data_ptr<float>(),
              l_i.data_ptr<float>(),
              m_j.data_ptr<float>(),
              l_j.data_ptr<float>(),
              m_k.data_ptr<float>(),
              l_k.data_ptr<float>(),
              sum_q.data_ptr<float>(),
              sum_r.data_ptr<float>(),
              sum_s.data_ptr<float>(),
              grad_Q.data_ptr<float>(),
              grad_S.data_ptr<float>(),
              mask_ptr, N, H, scale);

      AT_CUDA_CHECK(cudaGetLastError());
    }

    // Phase 3: grad_R
    {
      constexpr int tileJ = TILE_J;
      constexpr int tileK = TILE_K;
      constexpr int tileI = 16;

      dim3 block_dim(tileJ, tileK);
      dim3 grid_dim((N + tileJ - 1) / tileJ,
                    (N + tileK - 1) / tileK,
                    B * H);

      constexpr int D_PAD_r = D_TMPL + 1;
      const size_t shmem_bytes =
          5 * tileJ * D_PAD_r * sizeof(float) +
          5 * tileK * D_PAD_r * sizeof(float) +
          5 * tileI * D_TMPL * sizeof(float) +
          3 * tileI * sizeof(float);

      cudaFuncSetAttribute(
          R_grad_kernel<false, tileJ, tileI, tileK, D_TMPL>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          shmem_bytes);

      R_grad_kernel<false, tileJ, tileI, tileK, D_TMPL>
          <<<grid_dim, block_dim, shmem_bytes, at::cuda::getCurrentCUDAStream()>>>(

             // the input tensors
              reinterpret_cast<const bf16*>(Q.data_ptr<at::BFloat16>()),
              reinterpret_cast<const bf16*>(R.data_ptr<at::BFloat16>()),
              reinterpret_cast<const bf16*>(S.data_ptr<at::BFloat16>()),
              reinterpret_cast<const bf16*>(Vq_1.data_ptr<at::BFloat16>()),
              reinterpret_cast<const bf16*>(Vq_2.data_ptr<at::BFloat16>()),
              reinterpret_cast<const bf16*>(Vr_1.data_ptr<at::BFloat16>()),
              reinterpret_cast<const bf16*>(Vr_2.data_ptr<at::BFloat16>()),
              reinterpret_cast<const bf16*>(Vs_1.data_ptr<at::BFloat16>()),
              reinterpret_cast<const bf16*>(Vs_2.data_ptr<at::BFloat16>()),

              // the gradient of the output 
              reinterpret_cast<const bf16*>(grad_Y_q.data_ptr<at::BFloat16>()),
              reinterpret_cast<const bf16*>(grad_Y_r.data_ptr<at::BFloat16>()),
              reinterpret_cast<const bf16*>(grad_Y_s.data_ptr<at::BFloat16>()),
              reinterpret_cast<const bf16*>(grad_Y_q_.data_ptr<at::BFloat16>()),
              reinterpret_cast<const bf16*>(grad_Y_r_.data_ptr<at::BFloat16>()),
              reinterpret_cast<const bf16*>(grad_Y_s_.data_ptr<at::BFloat16>()),

              // softmax stats
              m_i.data_ptr<float>(),
              l_i.data_ptr<float>(),
              m_j.data_ptr<float>(),
              l_j.data_ptr<float>(),
              m_k.data_ptr<float>(),
              l_k.data_ptr<float>(),

              // jacobian correction sums
              sum_q.data_ptr<float>(),
              sum_r.data_ptr<float>(),
              sum_s.data_ptr<float>(),

              // the gradient we want to compute
              grad_R.data_ptr<float>(),

              mask_ptr, N, H, scale);

      AT_CUDA_CHECK(cudaGetLastError());
    }
  });

  cudaDeviceSynchronize();

  return std::make_tuple(
      grad_Q.to(at::kBFloat16),
      grad_R.to(at::kBFloat16),
      grad_S.to(at::kBFloat16),
      grad_Vq_1.to(at::kBFloat16),
      grad_Vq_2.to(at::kBFloat16),
      grad_Vr_1.to(at::kBFloat16),
      grad_Vr_2.to(at::kBFloat16),
      grad_Vs_1.to(at::kBFloat16),
      grad_Vs_2.to(at::kBFloat16));
}

// =============================================================================
// Public API: backward_cuda (uses pre-computed softmax stats from forward pass)
// =============================================================================
// NOTE: The forward pass computes and returns softmax stats (m_i, l_i, m_j, l_j, m_k, l_k).

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
backward_cuda(torch::Tensor grad_Y_q,
              torch::Tensor grad_Y_r,
              torch::Tensor grad_Y_s,
              torch::Tensor grad_Y_q_,
              torch::Tensor grad_Y_r_,
              torch::Tensor grad_Y_s_,
              torch::Tensor Q,
              torch::Tensor R,
              torch::Tensor S,
              torch::Tensor Vq_1,
              torch::Tensor Vq_2,
              torch::Tensor Vr_1,
              torch::Tensor Vr_2,
              torch::Tensor Vs_1,
              torch::Tensor Vs_2,
              torch::Tensor m_i,
              torch::Tensor l_i,
              torch::Tensor m_j,
              torch::Tensor l_j,
              torch::Tensor m_k,
              torch::Tensor l_k,
              torch::Tensor mask,
              double dropout_rate,
              torch::Tensor Y_q,
              torch::Tensor Y_r,
              torch::Tensor Y_s) {

  // Ensure all tensors are contiguous
  grad_Y_q = grad_Y_q.contiguous();
  grad_Y_r = grad_Y_r.contiguous();
  grad_Y_s = grad_Y_s.contiguous();
  grad_Y_q_ = grad_Y_q_.contiguous();
  grad_Y_r_ = grad_Y_r_.contiguous();
  grad_Y_s_ = grad_Y_s_.contiguous();
  Q = Q.contiguous();
  R = R.contiguous();
  S = S.contiguous();
  Vq_1 = Vq_1.contiguous();
  Vq_2 = Vq_2.contiguous();
  Vr_1 = Vr_1.contiguous();
  Vr_2 = Vr_2.contiguous();
  Vs_1 = Vs_1.contiguous();
  Vs_2 = Vs_2.contiguous();
  m_i = m_i.contiguous();
  l_i = l_i.contiguous();
  m_j = m_j.contiguous();
  l_j = l_j.contiguous();
  m_k = m_k.contiguous();
  l_k = l_k.contiguous();
  if (mask.defined()) {
    mask = mask.contiguous();
  }
  if (Y_q.defined()) Y_q = Y_q.contiguous();
  if (Y_r.defined()) Y_r = Y_r.contiguous();
  if (Y_s.defined()) Y_s = Y_s.contiguous();

  // Call the internal implementation directly with provided stats
  return backward_impl(
      grad_Y_q, grad_Y_r, grad_Y_s, grad_Y_q_, grad_Y_r_, grad_Y_s_,
      Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2,
      m_i, l_i, m_j, l_j, m_k, l_k, mask, dropout_rate, Y_q, Y_r, Y_s);
}

