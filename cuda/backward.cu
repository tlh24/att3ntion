/**
 * @file backward.cu
 * @brief Backward kernels for hypergraph and single-gather attention, using
 * the softmax stats saved by the forward. Copyright (c) 2026 Springtail AI. MIT License.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <tuple>

#include "common.cuh"
#include "../cpp/cuda_bindings.h"

#ifndef T_I
#define T_I TILE_I
#endif
#ifndef T_J
#define T_J TILE_J
#endif
#ifndef T_K
#define T_K TILE_K
#endif

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

    const float m_reg_val = m_regBH[reg_safe];
    const float inv_l_reg = 1.0f / fmaxf(l_regBH[reg_safe], DENOM_EPS);

    __shared__ float sh_X [T_J][D_CONST];
    __shared__ float sh_V [T_J][D_CONST];
    __shared__ float sh_gY[T_J][D_CONST];
    __shared__ float sh_m[T_J];
    __shared__ float sh_l_inv[T_J];      // pre-inverted, multiply not divide

    for (int lBase=0; lBase<N; lBase+=T_J){
        // Row per thread y, D strided over thread x; needs T_K >= T_J.
        int lt = threadIdx.y;
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
    const int tx = blockIdx.x * T_I + threadIdx.x;
    const int ty = blockIdx.y * T_K + threadIdx.y;
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
    // gY_reg row: on thread y it is warp-uniform, so read it from global
    // (broadcast, L1-hot) rather than a fourth D-long register array that
    // spills. On thread x each lane wants its own row, so cache it.
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
        // Row per thread y, D strided over thread x; needs T_K >= T_J.
        const int lt = threadIdx.y;
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
// Tensor-core backward (gather-only path)
// =============================================================================
// Used when the scatter cotangents are all zero: the cross terms d4/d5/d6
// vanish, so every correction sum collapses FlashAttention-style to
// rowsum(dY o Y), computed host-side. What remains is one cube pass, run three
// times with permuted roles like Y_gather_tc:
//
//   anchor a (one CTA per (b,h,a)) / rows r (16-row warp tiles) / cols c
//   (double-buffered BTC_BK-column stage, so smem is flat in N). Score-shaped
//   GEMMs per (r,c) tile, D contracted:
//
//     x   = scale * sum_d Xa[d]  * Xr[r,d]  * Xc[c,d]      (logits)
//     d_a =         sum_d gYa[d] * Vr[r,d]  * Vc[c,d]
//     d_r =         sum_d Va[d]  * gYr[r,d] * Vc[c,d]
//     d_c =         sum_d Va[d]  * Vr[r,d]  * gYc[c,d]
//
//   The anchor vector is folded into the A operands as a diagonal rescale of
//   the raw rows, staged once per row block into four shared-memory tiles.
//   Softmax weights come straight from the forward stats:
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
// Four of the six factors are separable and fold into scales the cell loop
// already multiplies by: mask[r][a] into P_r's inv-l and mask[a][r] as a 0/1
// float per row, mask[c][a] into P_c's inv-l and mask[a][c] into ilac_sm, the
// anchor's inv-l staged per column. Folding needs exp to stay finite on dead
// cells (a fully masked anchor carries m = NEG_INF from the forward, so
// exp(x - m) is inf and a zero gate gives NaN), hence the exponent is clamped
// at 0. Live cells are untouched since the forward guarantees x <= m, and the
// clamp also subsumes the pad test: pad rows and cols carry a zero inv-l and
// read zero mask bits.
//
// The other two factors are 2-D, but a (r,c) tile only touches a [BJ][BK]
// mask rectangle, staged per tile as two bit-packed windows: msk_sm packs
// mask[r][c] along c (one word per row), mskT_sm packs mask[c][r] along r.
// Both pack along the mask's own fast axis, so both load straight from global
// with no transpose. Pad rows and cols pack to zero bits.
//
// The collapsed correction sums are unaffected by masking: sum = rowsum(dY o Y)
// holds for any weight matrix the forward actually used, and the forward
// zeroes masked cells, so a fully masked row gets Y = 0 and sum = 0.

// Tile shape: WARPS x 16 rows per block iteration, BK cols per inner iteration.
// The host picks the largest shape whose smem fits the device (8 warps on an
// H100; 4 at D=64 and 2 at D=128 inside the 99 KB of sm_86/89).
constexpr int BTC_WARPS = 8;
constexpr int BTC_BK = 32;
constexpr int BTC_BJ = BTC_WARPS * 16;

// Which softmaxes a launch differentiates. BWD_ALL is the three-gather pass
// above. The single-gather backward has one softmax, normalized over the
// query, so it specializes on where the query sits in the pass: the anchor
// for the Q-owned pass, the rows for the R/S-owned passes. Everything belonging
// to the two absent softmaxes (operand tiles, score GEMMs, weights, the value
// accumulator when the anchor has no value) drops out at compile time.
enum BwdRole { BWD_ALL = 0, BWD_QUERY_ANCHOR = 1, BWD_QUERY_ROWS = 2 };

// single_col=true drops the second column buffer: the caller guarantees every
// row block visits exactly one column tile, so the cp.async pipeline's second
// buffer would go unused.
constexpr size_t btc_smem_bytes(int D, int warps, int bk, bool masked, int role = BWD_ALL, bool single_col = false) {
    const int bj = warps * 16, dpad = D + 8;
    const int pa = role != BWD_QUERY_ROWS, pr = role != BWD_QUERY_ANCHOR, pc = role == BWD_ALL;
    const int a_tiles = 1 + pa + pr + pc, c_tiles = 2 + pc, outs = 1 + pr;
    const int cb = single_col ? 1 : 2;
    size_t b = sizeof(bf16) * ((size_t)a_tiles * bj * dpad + (size_t)cb * c_tiles * bk * dpad)
             + sizeof(float) * ((size_t)3 * D + cb * 3 * bk + 3 * bj + warps * outs * D + outs * D);
    if (masked) {
        b += sizeof(uint32_t) * cb * (size_t)(pr * bj + pc * bk * (bj / 32))
           + sizeof(float) * (cb * (size_t)bk + bj) * pa;
    }
    return b;
}

template<int D_CONST, bool MASKED, int WARPS, int BK, int ROLE = BWD_ALL, int DHT = 64, bool SINGLE_COL = false>
__global__ __launch_bounds__(WARPS * 32, 1)
void Bwd_gather_tc(
    const bf16* __restrict__ Xa_bf,  // anchor side [B,H,N,D]
    const bf16* __restrict__ Va_bf,
    const bf16* __restrict__ gYa_bf,
    const bf16* __restrict__ Xr_bf,  // row side
    const bf16* __restrict__ Vr_bf,
    const bf16* __restrict__ gYr_bf,
    const bf16* __restrict__ Xc_bf,  // col side (streamed per k tile)
    const bf16* __restrict__ Vc_bf,
    const bf16* __restrict__ gYc_bf,
    const float* __restrict__ m_a, const float* __restrict__ l_a, const float* __restrict__ sum_a,
    const float* __restrict__ m_r, const float* __restrict__ l_r, const float* __restrict__ sum_r,
    const float* __restrict__ m_c, const float* __restrict__ l_c, const float* __restrict__ sum_c,
    float* __restrict__ gradXa,      // [B,H,N,D] fp32, direct store
    float* __restrict__ gradVa,      // [B,H,N,D] fp32, direct store
    const bool* __restrict__ mask,   // [B,N,N] or null (MASKED only)
    int H, int N, int win, float scale, int Hkv)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    static_assert(D_CONST == 64 || D_CONST == 128, "Bwd_gather_tc supports D=64/128");
    static_assert(DHT == 64 || DHT == D_CONST, "output slice is 64 channels or all of D");
    constexpr int D = D_CONST;
    constexpr int DPAD = D + 8;
    constexpr int BJ = WARPS * 16;
    constexpr int MRW = BJ / 32;    // words per col of a transposed mask window
    constexpr int KS = D / 16;      // score GEMM k-steps (D contracted)
    // The output accumulators cover DH cols per pass over the col side. At
    // D=128, DHT=64 makes two passes that recompute the scores rather than
    // doubling the accumulator registers; DHT=128 is one pass.
    constexpr int DH = DHT;
    constexpr int NPASS = D / DH;
    constexpr bool PA = ROLE != BWD_QUERY_ROWS;    // anchor-normalized softmax
    constexpr bool PR = ROLE != BWD_QUERY_ANCHOR;  // row-normalized softmax
    constexpr bool PC = ROLE == BWD_ALL;           // col-normalized softmax
    constexpr int NOUT = PR ? 2 : 1;               // gradXa (+ gradVa)

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

    constexpr int CB = SINGLE_COL ? 1 : 2;  // column buffer count

    extern __shared__ char smem_raw[];
    // The four A operands, anchor already folded in: scale*Xa o Xr, gYa o Vr,
    // Va o gYr, Va o Vr. Staged per row block; the epilogue re-reads the raw
    // rows from global instead of keeping a second copy here.
    bf16* a0_sm   = reinterpret_cast<bf16*>(smem_raw);            // [BJ][DPAD]
    bf16* a1_sm   = a0_sm + BJ * DPAD;                            // PA
    bf16* a2_sm   = a1_sm + (PA ? BJ * DPAD : 0);                 // PR
    bf16* a3_sm   = a2_sm + (PR ? BJ * DPAD : 0);                 // PC
    bf16* xc_sm   = a3_sm + (PC ? BJ * DPAD : 0);                 // [CB][BK][DPAD]
    bf16* vc_sm   = xc_sm + CB * BK * DPAD;
    bf16* gyc_sm  = vc_sm + CB * BK * DPAD;                       // PC
    float* anchX  = reinterpret_cast<float*>(gyc_sm + (PC ? CB * BK * DPAD : 0));  // [D] scale*Xa
    float* anchV  = anchX + D;                                    // [D]
    float* anchG  = anchV + D;                                    // [D]
    float* mc_sm  = anchG + D;                                    // [CB][BK]
    float* ilc_sm = mc_sm + CB * BK;
    float* sc_sm  = ilc_sm + CB * BK;
    float* mr_sm  = sc_sm + CB * BK;                             // [BJ]
    float* ilr_sm = mr_sm + BJ;
    float* sr_sm  = ilr_sm + BJ;
    float* wOut   = sr_sm + BJ;                                  // [WARPS][NOUT*D]
    float* redOut = wOut + WARPS * NOUT * D;                     // [NOUT*D]
    // MASKED only (host omits these bytes): the tile in flight's mask windows.
    uint32_t* msk_sm  = reinterpret_cast<uint32_t*>(redOut + NOUT * D);   // [CB][BJ] PR
    uint32_t* mskT_sm = msk_sm + (PR ? CB * BJ : 0);                     // [CB][BK][MRW] PC
    float* ilac_sm = reinterpret_cast<float*>(mskT_sm + (PC ? CB * BK * MRW : 0));  // [CB][BK] PA
    float* par_sm  = ilac_sm + CB * BK;                                  // [BJ] PA

    // Offsets by operand role. Query-head operands (Q, dY, the forward stats and
    // every output, including the per-head gradient partials) live at (b*H + h);
    // key/value operands at the KV head (b*Hkv + h / (H/Hkv)), identical unless
    // the caller shares KV across query heads. With the query as anchor the
    // anchor side is query-indexed and rows/cols are KV; with the queries as rows
    // the anchor (R or S) is KV, the rows (Q, dY) are query-indexed, cols are KV.
    // BWD_ALL always runs with Hkv == H.
    const int64_t bh = (int64_t)b * H + h;
    const int64_t kvh = (int64_t)b * Hkv + h / (H / Hkv);
    const int64_t q_off = bh * N * D, kv_off = kvh * N * D;
    const int64_t anc_off  = (ROLE == BWD_QUERY_ROWS ? kv_off : q_off) + (int64_t)a * D;  // Xa / Va / gYa reads
    const int64_t a_off    = q_off + (int64_t)a * D;                                      // gradXa / gradVa stores
    const int64_t rows_off = (ROLE == BWD_QUERY_ROWS ? q_off : kv_off);                   // Xr / Vr / gYr
    const int64_t nd_off   = kv_off;                                                      // Xc / Vc / gYc
    const int64_t st_off = bh * N;

    // ---- one-time loads: anchor only ----
    constexpr int DV = D / 8;
    for (int d = tid; d < D; d += blockDim.x) {
        anchX[d] = scale * bf2f(Xa_bf[anc_off + d]);
        if constexpr (PR || PC) anchV[d] = bf2f(Va_bf[anc_off + d]);
        if constexpr (PA) anchG[d] = bf2f(gYa_bf[anc_off + d]);
    }
    for (int d = tid; d < NOUT * D; d += blockDim.x) redOut[d] = 0.0f;

    const float ma  = PA ? m_a[st_off + a] : 0.0f;
    const float ila = PA ? 1.0f / fmaxf(l_a[st_off + a], DENOM_EPS) : 0.0f;
    const float sa  = PA ? sum_a[st_off + a] : 0.0f;
    const bool* mb  = MASKED ? mask + (int64_t)b * N * N : nullptr;

    // ---- row blocks of BJ rows, one 16-row tile per warp ----
    // win > 0 restricts the row blocks and, per block, the col tiles to those
    // that can hold visible pairs (sg_row_bounds / sg_col_bounds, keyed on
    // which side of the pass the query sits); the mask still decides every
    // cell. BWD_ALL passes 0 and stays dense.
    constexpr int SIDE = (ROLE == BWD_QUERY_ROWS) ? SG_QUERY_ROWS : SG_QUERY_ANCHOR;
    int j_lo, j_hi;
    sg_row_bounds(SIDE, a, N, win, BJ, j_lo, j_hi);
    for (int j0 = j_lo; j0 < j_hi; j0 += BJ) {
        __syncthreads();  // previous iteration's smem reads (and anchor) done
        int k_lo, k_hi;
        sg_col_bounds(SIDE, a, j0, BJ, N, win, BK, k_lo, k_hi);

        // Stage col tile k0 into buffer `buf`: matrices, forward stats and
        // (masked) mask windows. Pad cols carry a zero inv-l and zero mask bits.
        auto stage_cols = [&](int k0, int buf) {
            bf16* xs = xc_sm + buf * BK * DPAD;
            bf16* vs = vc_sm + buf * BK * DPAD;
            bf16* gs = gyc_sm + buf * BK * DPAD;
            for (int idx = tid; idx < BK * DV; idx += blockDim.x) {
                const int kl = idx / DV, dv = (idx % DV) * 8;
                const int k = k0 + kl;
                if (k < N) {
                    const int64_t off = nd_off + (int64_t)k * D + dv;
                    cp_async16(xs + kl * DPAD + dv, Xc_bf + off);
                    cp_async16(vs + kl * DPAD + dv, Vc_bf + off);
                    if constexpr (PC) cp_async16(gs + kl * DPAD + dv, gYc_bf + off);
                } else {
                    const uint4 z = make_uint4(0, 0, 0, 0);
                    *reinterpret_cast<uint4*>(xs + kl * DPAD + dv) = z;
                    *reinterpret_cast<uint4*>(vs + kl * DPAD + dv) = z;
                    if constexpr (PC) *reinterpret_cast<uint4*>(gs + kl * DPAD + dv) = z;
                }
            }
            for (int kl = tid; kl < BK; kl += blockDim.x) {
                const int k = k0 + kl;
                float mc = 0.0f, ilc = 0.0f, sc = 0.0f;
                if (PC && k < N) {
                    mc  = m_c[st_off + k];
                    ilc = 1.0f / fmaxf(l_c[st_off + k], DENOM_EPS);
                    sc  = sum_c[st_off + k];
                }
                if constexpr (MASKED) {
                    // Separable column factors: mask[c][a] zeroes P_c's
                    // inv-l, mask[a][c] rides in ilac_sm.
                    if (PC && (k >= N || !mb[(int64_t)k * N + a])) ilc = 0.0f;
                    if constexpr (PA) {
                        ilac_sm[buf * BK + kl] =
                            (k < N && mb[(int64_t)a * N + k]) ? ila : 0.0f;
                    }
                }
                mc_sm[buf * BK + kl]  = mc;
                ilc_sm[buf * BK + kl] = ilc;
                sc_sm[buf * BK + kl]  = sc;
            }
            if constexpr (MASKED && PR) {
                for (int jl = tid; jl < BJ; jl += blockDim.x) {
                    const int j = j0 + jl;
                    msk_sm[buf * BJ + jl] = (j < N)
                        ? sg_pack_mask32(mb + (int64_t)j * N + k0, min(BK, N - k0))
                        : 0u;
                }
            }
            if constexpr (MASKED && PC) {
                for (int idx = tid; idx < BK * MRW; idx += blockDim.x) {
                    const int kl = idx / MRW, w = idx - kl * MRW;
                    const int k = k0 + kl, jb = j0 + w * 32;
                    mskT_sm[buf * BK * MRW + idx] = (k < N && jb < N)
                        ? sg_pack_mask32(mb + (int64_t)k * N + jb, min(32, N - jb))
                        : 0u;
                }
            }
        };

        // A operands: the anchor rescale is done in fp32 with a single bf16
        // rounding, matching Y_gather_tc's Qp precision. Pad rows stage as zeros.
        for (int idx = tid; idx < BJ * DV; idx += blockDim.x) {
            const int jl = idx / DV, dv = (idx % DV) * 8;
            const int j = j0 + jl;
            uint4 xq = make_uint4(0, 0, 0, 0), vq = xq, gq = xq;
            if (j < N) {
                const int64_t off = rows_off + (int64_t)j * D + dv;
                xq = *reinterpret_cast<const uint4*>(Xr_bf + off);
                if constexpr (PA || PC) vq = *reinterpret_cast<const uint4*>(Vr_bf + off);
                if constexpr (PR) gq = *reinterpret_cast<const uint4*>(gYr_bf + off);
            }
            const __nv_bfloat162* xp = reinterpret_cast<const __nv_bfloat162*>(&xq);
            const __nv_bfloat162* vp = reinterpret_cast<const __nv_bfloat162*>(&vq);
            const __nv_bfloat162* gp = reinterpret_cast<const __nv_bfloat162*>(&gq);
            uint4 a0, a1, a2, a3;
            __nv_bfloat162* p0 = reinterpret_cast<__nv_bfloat162*>(&a0);
            __nv_bfloat162* p1 = reinterpret_cast<__nv_bfloat162*>(&a1);
            __nv_bfloat162* p2 = reinterpret_cast<__nv_bfloat162*>(&a2);
            __nv_bfloat162* p3 = reinterpret_cast<__nv_bfloat162*>(&a3);
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                const int d = dv + 2 * e;
                const float2 x = __bfloat1622float2(xp[e]);
                const float2 v = __bfloat1622float2(vp[e]);
                const float2 gy = __bfloat1622float2(gp[e]);
                p0[e] = __floats2bfloat162_rn(x.x * anchX[d], x.y * anchX[d + 1]);
                p1[e] = __floats2bfloat162_rn(v.x * anchG[d], v.y * anchG[d + 1]);
                p2[e] = __floats2bfloat162_rn(gy.x * anchV[d], gy.y * anchV[d + 1]);
                p3[e] = __floats2bfloat162_rn(v.x * anchV[d], v.y * anchV[d + 1]);
            }
            *reinterpret_cast<uint4*>(a0_sm + jl * DPAD + dv) = a0;
            if constexpr (PA) *reinterpret_cast<uint4*>(a1_sm + jl * DPAD + dv) = a1;
            if constexpr (PR) *reinterpret_cast<uint4*>(a2_sm + jl * DPAD + dv) = a2;
            if constexpr (PC) *reinterpret_cast<uint4*>(a3_sm + jl * DPAD + dv) = a3;
        }
        for (int jl = tid; jl < BJ; jl += blockDim.x) {
            const int j = j0 + jl;
            if (j < N) {
                float il = 0.0f;
                if constexpr (PR) {
                    mr_sm[jl] = m_r[st_off + j];
                    il        = 1.0f / fmaxf(l_r[st_off + j], DENOM_EPS);
                    sr_sm[jl] = sum_r[st_off + j];
                }
                if constexpr (MASKED) {
                    // Row-side factors go through smem rather than registers:
                    // a rematerializable smem read is cheaper than a live register.
                    if (PR && !mb[(int64_t)j * N + a]) il = 0.0f;          // mask[r][a]
                    if constexpr (PA) par_sm[jl] = mb[(int64_t)a * N + j] ? 1.0f : 0.0f;  // mask[a][r]
                }
                ilr_sm[jl] = il;
            } else {
                mr_sm[jl] = 0.0f; ilr_sm[jl] = 0.0f; sr_sm[jl] = 0.0f;
                if constexpr (MASKED && PA) par_sm[jl] = 0.0f;
            }
        }
        __syncthreads();

        const int jw = warp * 16;
        const float mr0 = PR ? mr_sm[jw + g]     : 0.0f, sr0 = PR ? sr_sm[jw + g]     : 0.0f;
        const float mr1 = PR ? mr_sm[jw + g + 8] : 0.0f, sr1 = PR ? sr_sm[jw + g + 8] : 0.0f;
        const float ilr0 = PR ? ilr_sm[jw + g] : 0.0f, ilr1 = PR ? ilr_sm[jw + g + 8] : 0.0f;
        const float par0 = (MASKED && PA) ? par_sm[jw + g]     : 1.0f;
        const float par1 = (MASKED && PA) ? par_sm[jw + g + 8] : 1.0f;
        // Zero-filled pad rows produce x = 0, which with extreme stats can
        // push exp(x - m)/l past bf16 range (inf -> 0*inf = NaN downstream).
        // Forcing pad cells to NEG_INF zeroes all three weights exactly.
        const bool rpad0 = (j0 + jw + g)     >= N;
        const bool rpad1 = (j0 + jw + g + 8) >= N;

        // Rows jw+g and jw+g+8 share one word of the transposed window (a
        // 16-row tile never straddles a 32-row boundary), so one load covers
        // both at shifts 0 and 8.
        const int rwd = (jw + g) >> 5;
        const int rb  = (jw + g) & 31;

        for (int pass = 0; pass < NPASS; pass++) {
            const int d0 = pass * DH;   // this pass's output col slice
            stage_cols(k_lo, 0);
            asm volatile("cp.async.wait_all;\n" ::);
            __syncthreads();

            float Ug[DH / 8][4], U1[PR ? DH / 8 : 1][4], U2[PC ? DH / 8 : 1][4];
            #pragma unroll
            for (int nt = 0; nt < DH / 8; nt++) {
                #pragma unroll
                for (int e = 0; e < 4; e++) {
                    Ug[nt][e] = 0.0f;
                    if constexpr (PR) U1[nt][e] = 0.0f;
                    if constexpr (PC) U2[nt][e] = 0.0f;
                }
            }

            int cur = 0;
            for (int k0 = k_lo; k0 < k_hi; k0 += BK) {
                // Prefetch the next tile; the closing barrier publishes it and
                // frees `cur`. SINGLE_COL has a single trip, nothing to prefetch.
                int nxt = cur;
                if constexpr (!SINGLE_COL) {
                    nxt = cur ^ 1;
                    if (k0 + BK < k_hi) stage_cols(k0 + BK, nxt);
                }
                const bf16* xc_cur  = xc_sm + cur * BK * DPAD;
                const bf16* vc_cur  = vc_sm + cur * BK * DPAD;
                const bf16* gyc_cur = gyc_sm + cur * BK * DPAD;
                const float* mc_cur  = mc_sm + cur * BK;
                const float* ilc_cur = ilc_sm + cur * BK;
                const float* sc_cur  = sc_sm + cur * BK;

                // 16 cols at a time: score GEMMs, elementwise, output GEMMs.
                #pragma unroll
                for (int s2 = 0; s2 < BK / 16; s2++) {
                    float ax[2][4], ada[2][4], adr[2][4], adc[2][4];
                    #pragma unroll
                    for (int nt = 0; nt < 2; nt++) {
                        #pragma unroll
                        for (int e = 0; e < 4; e++) {
                            ax[nt][e] = 0.0f; ada[nt][e] = 0.0f; adr[nt][e] = 0.0f; adc[nt][e] = 0.0f;
                        }
                    }
                    const bf16* bx = xc_cur + (s2 * 16 + brow) * DPAD + bcol8;
                    const bf16* bv = vc_cur + (s2 * 16 + brow) * DPAD + bcol8;
                    const bf16* bg = gyc_cur + (s2 * 16 + brow) * DPAD + bcol8;
                    #pragma unroll
                    for (int ks = 0; ks < KS; ks++) {
                        uint32_t A0[4], A1[4], A2[4], A3[4], bfr[4];
                        const int roff = (jw + lrow) * DPAD + ks * 16 + lcol8;
                        ldmatrix_x4(A0, a0_sm + roff);
                        if constexpr (PA) ldmatrix_x4(A1, a1_sm + roff);
                        if constexpr (PR) ldmatrix_x4(A2, a2_sm + roff);
                        if constexpr (PC) ldmatrix_x4(A3, a3_sm + roff);
                        ldmatrix_x4(bfr, bx + ks * 16);
                        mma_bf16_m16n8k16(ax[0], A0, bfr);
                        mma_bf16_m16n8k16(ax[1], A0, bfr + 2);
                        ldmatrix_x4(bfr, bv + ks * 16);   // shared by d_a and d_r
                        if constexpr (PA) {
                            mma_bf16_m16n8k16(ada[0], A1, bfr);
                            mma_bf16_m16n8k16(ada[1], A1, bfr + 2);
                        }
                        if constexpr (PR) {
                            mma_bf16_m16n8k16(adr[0], A2, bfr);
                            mma_bf16_m16n8k16(adr[1], A2, bfr + 2);
                        }
                        if constexpr (PC) {
                            ldmatrix_x4(bfr, bg + ks * 16);
                            mma_bf16_m16n8k16(adc[0], A3, bfr);
                            mma_bf16_m16n8k16(adc[1], A3, bfr + 2);
                        }
                    }

                    // Elementwise: weights from forward stats, Jacobian-corrected
                    // grad_A; repack C-fragments as output-GEMM A-fragments.
                    uint32_t gAf[4], Prf[4], Pcf[4];
                    #pragma unroll
                    for (int hf = 0; hf < 2; hf++) {
                        const int cl = s2 * 16 + hf * 8 + 2 * tig;   // col within the staged tile
                        const int c  = k0 + cl;
                        const float mc0 = PC ? mc_cur[cl]     : 0.0f, ilc0 = PC ? ilc_cur[cl]     : 0.0f;
                        const float mc1 = PC ? mc_cur[cl + 1] : 0.0f, ilc1 = PC ? ilc_cur[cl + 1] : 0.0f;
                        const float sc0 = PC ? sc_cur[cl] : 0.0f, sc1 = PC ? sc_cur[cl + 1] : 0.0f;
                        // The two 2-D factors, one aligned load per pair in either
                        // window: bits 0/1 of rc* are mask[r][c] and mask[r][c+1] (c
                        // even, so the pair shares a word), bits 0/8 of cr* are
                        // mask[c][r] and mask[c][r+8].
                        uint32_t rc0 = ~0u, rc1 = ~0u, cr0 = ~0u, cr1 = ~0u;
                        float ilac0 = 0.0f, ilac1 = 0.0f;
                        if constexpr (MASKED && PR) {
                            const uint32_t* mw  = msk_sm + cur * BJ;
                            rc0 = mw[jw + g]     >> cl;
                            rc1 = mw[jw + g + 8] >> cl;
                        }
                        if constexpr (MASKED && PC) {
                            const uint32_t* mtw = mskT_sm + cur * BK * MRW;
                            cr0 = mtw[cl * MRW + rwd]       >> rb;
                            cr1 = mtw[(cl + 1) * MRW + rwd] >> rb;
                        }
                        if constexpr (MASKED && PA) {
                            ilac0 = ilac_sm[cur * BK + cl];
                            ilac1 = ilac_sm[cur * BK + cl + 1];
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
                            float Pa = 0.0f;
                            Pr[e] = 0.0f; Pc[e] = 0.0f;
                            if constexpr (MASKED) {
                                // Exponent clamped at 0 so every gate can be a plain
                                // multiply or select (see the MASKED notes above).
                                const float x = ax[hf][e];
                                const float ilac = c1 ? ilac1 : ilac0;
                                const float par  = hi ? par1 : par0;
                                // Gate the inv-l, never the exp: guarding the whole
                                // expression lets nvcc branch around the MUFU, and a
                                // scattered mask then diverges inside the warp.
                                const float ilrg = (((hi ? rc1 : rc0) >> c1) & 1u) ? ilr : 0.0f;
                                const float ilcg = (((c1 ? cr1 : cr0) >> (hi ? 8 : 0)) & 1u)
                                                 ? ilc : 0.0f;
                                if constexpr (PA) Pa    = __expf(fminf(x - ma,  0.0f)) * ilac * par;
                                if constexpr (PR) Pr[e] = __expf(fminf(x - mrr, 0.0f)) * ilrg;
                                if constexpr (PC) Pc[e] = __expf(fminf(x - mcc, 0.0f)) * ilcg;
                            } else {
                                const bool pad = (hi ? rpad1 : rpad0) | (c + c1 >= N);
                                const float x = pad ? NEG_INF : ax[hf][e];
                                if constexpr (PA) Pa    = __expf(x - ma)  * ila;
                                if constexpr (PR) Pr[e] = __expf(x - mrr) * ilr;
                                if constexpr (PC) Pc[e] = __expf(x - mcc) * ilc;
                            }
                            if constexpr (ROLE == BWD_ALL) {
                                gA[e] = (ada[hf][e] - sa) * Pa
                                      + (adr[hf][e] - srr) * Pr[e]
                                      + (adc[hf][e] - scc) * Pc[e];
                            } else if constexpr (PA) {
                                gA[e] = (ada[hf][e] - sa) * Pa;
                            } else {
                                gA[e] = (adr[hf][e] - srr) * Pr[e];
                            }
                        }
                        gAf[2 * hf + 0] = pack_bf162(gA[0], gA[1]);
                        gAf[2 * hf + 1] = pack_bf162(gA[2], gA[3]);
                        if constexpr (PR) {
                            Prf[2 * hf + 0] = pack_bf162(Pr[0], Pr[1]);
                            Prf[2 * hf + 1] = pack_bf162(Pr[2], Pr[3]);
                        }
                        if constexpr (PC) {
                            Pcf[2 * hf + 0] = pack_bf162(Pc[0], Pc[1]);
                            Pcf[2 * hf + 1] = pack_bf162(Pc[2], Pc[3]);
                        }
                    }

                    // Output GEMMs: contract cols (B fragments via ldmatrix.trans).
                    const bf16* px = xc_cur + (s2 * 16 + lrow) * DPAD + lcol8 + d0;
                    const bf16* pv = vc_cur + (s2 * 16 + lrow) * DPAD + lcol8 + d0;
                    const bf16* pg = gyc_cur + (s2 * 16 + lrow) * DPAD + lcol8 + d0;
                    #pragma unroll
                    for (int np = 0; np < DH / 16; np++) {
                        uint32_t bfr[4];
                        ldmatrix_x4_trans(bfr, px + np * 16);
                        mma_bf16_m16n8k16(Ug[2 * np],     gAf, bfr);
                        mma_bf16_m16n8k16(Ug[2 * np + 1], gAf, bfr + 2);
                        if constexpr (PR) {
                            ldmatrix_x4_trans(bfr, pv + np * 16);
                            mma_bf16_m16n8k16(U1[2 * np],     Prf, bfr);
                            mma_bf16_m16n8k16(U1[2 * np + 1], Prf, bfr + 2);
                        }
                        if constexpr (PC) {
                            ldmatrix_x4_trans(bfr, pg + np * 16);
                            mma_bf16_m16n8k16(U2[2 * np],     Pcf, bfr);
                            mma_bf16_m16n8k16(U2[2 * np + 1], Pcf, bfr + 2);
                        }
                    }
                }

                // SINGLE_COL still needs the barrier: with NPASS > 1 the next
                // pass re-stages this buffer. Only the wait and toggle are dead.
                if constexpr (!SINGLE_COL) {
                    asm volatile("cp.async.wait_all;\n" ::);
                    cur = nxt;
                }
                __syncthreads();
            }

            // ---- epilogue: Hadamard row-collapse of this warp's 16 rows ----
            // Raw rows come from global (L2-hot); pad rows contribute zeros.
            float ng[DH / 4], nv[PR ? DH / 4 : 1];
            const int64_t r0 = rows_off + (int64_t)min(j0 + jw + g, N - 1) * D + d0 + 2 * tig;
            const int64_t r1 = rows_off + (int64_t)min(j0 + jw + g + 8, N - 1) * D + d0 + 2 * tig;
            auto ld2 = [](const bf16* p, bool pad) {
                return pad ? make_float2(0.0f, 0.0f)
                           : __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(p));
            };
            #pragma unroll
            for (int nt = 0; nt < DH / 8; nt++) {
                const float2 x0 = ld2(Xr_bf + r0 + nt * 8, rpad0),  x1 = ld2(Xr_bf + r1 + nt * 8, rpad1);
                ng[2 * nt + 0] = x0.x * Ug[nt][0] + x1.x * Ug[nt][2];
                ng[2 * nt + 1] = x0.y * Ug[nt][1] + x1.y * Ug[nt][3];
                if constexpr (PC) {
                    const float2 v0 = ld2(Vr_bf + r0 + nt * 8, rpad0),  v1 = ld2(Vr_bf + r1 + nt * 8, rpad1);
                    const float2 g0 = ld2(gYr_bf + r0 + nt * 8, rpad0), g1 = ld2(gYr_bf + r1 + nt * 8, rpad1);
                    nv[2 * nt + 0] = g0.x * U1[nt][0] + g1.x * U1[nt][2]
                                   + v0.x * U2[nt][0] + v1.x * U2[nt][2];
                    nv[2 * nt + 1] = g0.y * U1[nt][1] + g1.y * U1[nt][3]
                                   + v0.y * U2[nt][1] + v1.y * U2[nt][3];
                } else if constexpr (PR) {
                    const float2 g0 = ld2(gYr_bf + r0 + nt * 8, rpad0), g1 = ld2(gYr_bf + r1 + nt * 8, rpad1);
                    nv[2 * nt + 0] = g0.x * U1[nt][0] + g1.x * U1[nt][2];
                    nv[2 * nt + 1] = g0.y * U1[nt][1] + g1.y * U1[nt][3];
                }
            }
            #pragma unroll
            for (int off = 4; off <= 16; off <<= 1) {
                #pragma unroll
                for (int e = 0; e < DH / 4; e++) {
                    ng[e] += __shfl_xor_sync(0xFFFFFFFF, ng[e], off);
                    if constexpr (PR) nv[e] += __shfl_xor_sync(0xFFFFFFFF, nv[e], off);
                }
            }
            if (lane < 4) {
                float* wo = wOut + warp * NOUT * D + d0;
                #pragma unroll
                for (int nt = 0; nt < DH / 8; nt++) {
                    wo[nt * 8 + 2 * lane]         = ng[2 * nt + 0];
                    wo[nt * 8 + 2 * lane + 1]     = ng[2 * nt + 1];
                    if constexpr (PR) {
                        wo[D + nt * 8 + 2 * lane]     = nv[2 * nt + 0];
                        wo[D + nt * 8 + 2 * lane + 1] = nv[2 * nt + 1];
                    }
                }
            }
        }
        __syncthreads();
        for (int t = tid; t < NOUT * D; t += blockDim.x) {
            float acc = 0.0f;
            #pragma unroll
            for (int w = 0; w < WARPS; w++) acc += wOut[w * NOUT * D + t];
            redOut[t] += acc;
        }
    }
    __syncthreads();

    // ---- direct stores: this CTA exclusively owns row a of both outputs ----
    for (int t = tid; t < D; t += blockDim.x) {
        gradXa[a_off + t] = scale * redOut[t];
        if constexpr (PR) gradVa[a_off + t] = redOut[D + t];
    }
#endif  // __CUDA_ARCH__ >= 800
}

// =============================================================================
// Gradient Kernels for Q, R, S (with Jacobian corrections)
// =============================================================================
// Two passes over 2D (i,k) or (j,k) tiles, streaming the third mode:
//   1. QS_grad_kernel<true>  -> correction sums sum_q, sum_r, sum_s
//   2. QS_grad_kernel<false> -> gradQ, gradS;  R_grad_kernel<false> -> gradR

/**
 * QS_grad_kernel - gradQ and gradS over (i,k) tiles, streaming j.
 *
 * CORRECTION_ONLY=true:  correction sums (sum_q, sum_r, sum_s)
 * CORRECTION_ONLY=false: gradQ and gradS from precomputed corrections
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
    // Factored accumulation: rj_weighted[d] = sum_j grad_A_j * R[j,d], then
    // gradQ[i,d] = rj_weighted[d] * S[k,d] and gradS[k,d] = rj_weighted[d] * Q[i,d]
    // in the epilogue. One D-long accumulator instead of two, one shmem load per d.
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

        // D-tiled dot products with j sub-tiling: the i/k pairwise products are
        // formed once per D_TILE and reused across J_SUB rows of j, at the cost
        // of 7 x J_SUB per-j accumulators.
        constexpr int J_SUB  = 4;
        constexpr int D_TILE = 4;

        for (int jSub = 0; jSub < BLOCK_J && (jBase + jSub) < N; jSub += J_SUB) {
            float dot_j[J_SUB], d1_j[J_SUB], d2_j[J_SUB], d3_j[J_SUB];
            float d4_j[J_SUB], d5_j[J_SUB], d6_j[J_SUB];
            #pragma unroll
            for (int jj = 0; jj < J_SUB; jj++) {
                dot_j[jj] = 0.f; d1_j[jj] = 0.f; d2_j[jj] = 0.f; d3_j[jj] = 0.f;
                d4_j[jj] = 0.f; d5_j[jj] = 0.f; d6_j[jj] = 0.f;
            }

            for (int d_base = 0; d_base < D_CONST; d_base += D_TILE) {
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

                // j-arrays have stride D_CONST (no padding) and d_base is a
                // multiple of 4, so each 4-float slice is 16-byte aligned: one
                // LDS.128 per array.
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
        // S[k] and Q[i] rows are still in shared memory from before the j loop.
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
 * R_grad_kernel - gradR over (j,k) tiles, streaming i.
 *
 * CORRECTION_ONLY=true:  correction sum sum_r[j]
 * CORRECTION_ONLY=false: gradR from precomputed corrections
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

    // Streamed i-tile, stride D_CONST (not D_PAD) so 4-float slices are
    // 16-byte aligned for LDS.128.
    float* sh_Q    = sh_dYk2 + BLOCK_K * D_PAD;
    float* sh_Vq1  = sh_Q    + BLOCK_I * D_CONST;
    float* sh_Vq2  = sh_Vq1  + BLOCK_I * D_CONST;
    float* sh_dYi  = sh_Vq2  + BLOCK_I * D_CONST;
    float* sh_dYi2 = sh_dYi  + BLOCK_I * D_CONST;

    float* sh_mi   = sh_dYi2 + BLOCK_I * D_CONST;
    float* sh_li   = sh_mi   + BLOCK_I;
    float* sh_sumq = sh_li   + BLOCK_I;

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

    const int tid_l       = threadIdx.x + threadIdx.y * BLOCK_J;
    const int nThreads_l  = BLOCK_J * BLOCK_K;
    for (int iBase = 0; iBase < N; iBase += BLOCK_I) {
        // float4 stores to cut shared-store instruction count.
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

        // Same D-tiling as QS_grad_kernel, with the j/k products reused across
        // I_SUB rows of i.
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
                    // sh_Sk has stride D_PAD, not 16-byte aligned for ty > 0,
                    // so it stays scalar.
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

    if constexpr (CORRECTION_ONLY) {
        // Reduce reg_sum_r across k (threadIdx.y), reusing shared memory.
        float* reduce_buf = shmem;

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
        // Atomic: every k tile contributes to the same gradR rows.
        float* gRbh = gradR + bh * stride_BH;
        if (valid) {
            for (int d = 0; d < D_CONST; ++d)
                atomicAdd(&gRbh[j0*D_CONST + d], scale * grad_acc[d]);
        }
    }
}



// WARPS if its smem fits the device, else the minimum shape (4 warps at D=64,
// 2 at D=128), else nullptr.
template<int D, int WARPS, int BK, int ROLE = BWD_ALL>
static decltype(&Bwd_gather_tc<64, false, 8, 32>) pick_bwd_tc(
    bool use_mask, int max_smem_optin, size_t& smem, int& threads)
{
    smem = btc_smem_bytes(D, WARPS, BK, use_mask, ROLE);
    if (smem > (size_t)max_smem_optin) {
        constexpr int MIN_WARPS = (D == 128) ? 2 : 4;
        if constexpr (WARPS > MIN_WARPS) {
            return pick_bwd_tc<D, MIN_WARPS, BK, ROLE>(use_mask, max_smem_optin, smem, threads);
        }
        return nullptr;
    }
    threads = WARPS * 32;
    auto* k = use_mask ? Bwd_gather_tc<D, true, WARPS, BK, ROLE>
                       : Bwd_gather_tc<D, false, WARPS, BK, ROLE>;
    int attribute_current_device=0; AT_CUDA_CHECK(cudaGetDevice(&attribute_current_device));
    static thread_local int attr_device[2]={-1,-1};
    if (attr_device[use_mask] != attribute_current_device) {
        AT_CUDA_CHECK(cudaFuncSetAttribute(
            k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        attr_device[use_mask] = attribute_current_device;
    }
    return k;
}

// Exact tile shape for the single-gather roles, no fallback: a requested shape
// that does not fit returns nullptr instead of silently running another one.
// Legal set: 2/4/8 warps x BK 16/32, and at D=128 also DHT=128.
using BwdTcKernel = decltype(&Bwd_gather_tc<64, false, 8, 32>);

template<int D, int WARPS, int BK, int ROLE, int DHT, bool SINGLE_COL = false>
static BwdTcKernel sg_bwd_variant(bool use_mask, int max_smem_optin, size_t& smem, int& threads)
{
    smem = btc_smem_bytes(D, WARPS, BK, use_mask, ROLE, SINGLE_COL);
    if (smem > (size_t)max_smem_optin) return nullptr;
    threads = WARPS * 32;
    auto* k = use_mask ? Bwd_gather_tc<D, true, WARPS, BK, ROLE, DHT, SINGLE_COL>
                       : Bwd_gather_tc<D, false, WARPS, BK, ROLE, DHT, SINGLE_COL>;
    int attribute_current_device=0; AT_CUDA_CHECK(cudaGetDevice(&attribute_current_device));
    static thread_local int attr_device[2]={-1,-1};
    if (attr_device[use_mask] != attribute_current_device) {
        AT_CUDA_CHECK(cudaFuncSetAttribute(
            k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        attr_device[use_mask] = attribute_current_device;
    }
    return k;
}

// SINGLE_COL is only compiled for WARPS=2, the shape used for win <= 32;
// other shapes ignore single_col.
template<int D, int WARPS, int BK, int ROLE, int DHT>
static BwdTcKernel sg_bwd_pick_variant(bool use_mask, bool single_col, int max_smem_optin,
                                       size_t& smem, int& threads)
{
    if constexpr (WARPS == 2) {
        if (single_col) return sg_bwd_variant<D, WARPS, BK, ROLE, DHT, true>(use_mask, max_smem_optin, smem, threads);
    }
    return sg_bwd_variant<D, WARPS, BK, ROLE, DHT, false>(use_mask, max_smem_optin, smem, threads);
}

static BwdTcKernel sg_pick_bwd(int D, int warps, int bk, int dh, int role, bool use_mask,
                               int max_smem_optin, size_t& smem, int& threads, bool& legal,
                               bool single_col = false)
{
    legal = false;
#define SG_BWD_CASE(DD, W, K, DHV)                                                        \
    if (D == DD && warps == W && bk == K && dh == DHV) {                                  \
        legal = true;                                                                    \
        return role == BWD_QUERY_ANCHOR                                                  \
            ? sg_bwd_pick_variant<DD, W, K, BWD_QUERY_ANCHOR, DHV>(use_mask, single_col, max_smem_optin, smem, threads) \
            : sg_bwd_pick_variant<DD, W, K, BWD_QUERY_ROWS, DHV>(use_mask, single_col, max_smem_optin, smem, threads);  \
    }
    SG_BWD_CASE(64, 2, 16, 64)  SG_BWD_CASE(64, 2, 32, 64)
    SG_BWD_CASE(64, 4, 16, 64)  SG_BWD_CASE(64, 4, 32, 64)
    SG_BWD_CASE(64, 8, 16, 64)  SG_BWD_CASE(64, 8, 32, 64)
    SG_BWD_CASE(128, 2, 16, 64) SG_BWD_CASE(128, 2, 32, 64)
    SG_BWD_CASE(128, 4, 16, 64) SG_BWD_CASE(128, 4, 32, 64)
    SG_BWD_CASE(128, 8, 16, 64) SG_BWD_CASE(128, 8, 32, 64)
    SG_BWD_CASE(128, 2, 16, 128) SG_BWD_CASE(128, 2, 32, 128)
    SG_BWD_CASE(128, 4, 16, 128) SG_BWD_CASE(128, 4, 32, 128)
    SG_BWD_CASE(128, 8, 16, 128) SG_BWD_CASE(128, 8, 32, 128)
#undef SG_BWD_CASE
    return nullptr;
}

// =============================================================================
// D-dispatch: routes to the D_CONST template instantiation
// =============================================================================
#define DISPATCH_D(D_VAL, ...) \
  [&] { \
    if ((D_VAL) == 16)      { constexpr int D_TMPL = 16; __VA_ARGS__; } \
    else if ((D_VAL) == 32) { constexpr int D_TMPL = 32; __VA_ARGS__; } \
    else if ((D_VAL) == 64) { constexpr int D_TMPL = 64; __VA_ARGS__; } \
    else if ((D_VAL) == 128) { constexpr int D_TMPL = 128; __VA_ARGS__; } \
    else { TORCH_CHECK(false, "backward: unsupported D=", (D_VAL), ". Supported: 16, 32, 64, 128"); } \
  }()

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
  const int N = Q.size(2);
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
  // 2b. TENSOR-CORE FAST PATH (gather-only; see Bwd_gather_tc)
  // ============================================================================
  // Requires the forward outputs Y_q/Y_r/Y_s (for the collapsed correction
  // sums) and all-zero scatter cotangents; otherwise falls through to the
  // scalar path. Disable with ATT3_BWD_TC=0.
  if ((D == 64 || D == 128) && I == J && J == K && (N % 16 == 0)
      && Y_q.defined() && Y_r.defined() && Y_s.defined()) {
    static const int max_smem_optin = []() {
        int dev = 0, major = 0, v = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        if (major < 8) return 0;
        cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        return v;
    }();
    size_t smem_tc = 0;
    int threads_tc = 0;
    auto* tc_kernel = (D == 64)
        ? pick_bwd_tc<64, BTC_WARPS, BTC_BK>(use_mask, max_smem_optin, smem_tc, threads_tc)
        : pick_bwd_tc<128, BTC_WARPS, BTC_BK>(use_mask, max_smem_optin, smem_tc, threads_tc);
    if (att3_tc::state().bwd_enabled && tc_kernel != nullptr) {
      // Single host round-trip for the gate.
      const bool scatter_active =
          (grad_Y_q_.ne(0).any() | grad_Y_r_.ne(0).any() | grad_Y_s_.ne(0).any())
              .item<bool>();
      if (!scatter_active) {
        // With no scatter cross terms every correction sum collapses to
        // rowsum(dY o Y), as in FA2.
        sum_q = (grad_Y_q.to(at::kFloat) * Y_q.to(at::kFloat)).sum(-1).contiguous();
        sum_r = (grad_Y_r.to(at::kFloat) * Y_r.to(at::kFloat)).sum(-1).contiguous();
        sum_s = (grad_Y_s.to(at::kFloat) * Y_s.to(at::kFloat)).sum(-1).contiguous();

        auto stream = at::cuda::getCurrentCUDAStream();
        const dim3 grid_tc(N, H, B), block_tc(threads_tc);
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
                mask_ptr, H, N, 0, scale, H);
            ++att3_tc::state().bwd_launches;
        };
        // Role table (anchor / rows / cols), one launch per gradient family.
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
    // puts out on thread y, which the grad_Vs permutation wants.
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


  // ============================================================================
  // 5. JACOBIAN CORRECTIONS + 6. GRAD Q/S/R
  // ============================================================================
  DISPATCH_D(D, {
    // Correction sums sum_q, sum_r, sum_s
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

    // grad_Q + grad_S
    {
      constexpr int tileI = D_TMPL == 128 ? 8 : TILE_I;   // 16x16 tiles need 124 KB smem at D=128
      constexpr int tileK = D_TMPL == 128 ? 8 : TILE_K;
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

    // grad_R
    {
      constexpr int tileJ = D_TMPL == 128 ? 8 : TILE_J;
      constexpr int tileK = D_TMPL == 128 ? 8 : TILE_K;
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
// Public API: backward_cuda
// =============================================================================

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

  return backward_impl(
      grad_Y_q, grad_Y_r, grad_Y_s, grad_Y_q_, grad_Y_r_, grad_Y_s_,
      Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2,
      m_i, l_i, m_j, l_j, m_k, l_k, mask, dropout_rate, Y_q, Y_r, Y_s);
}



// =============================================================================
// Single query-gather backward (see cuda_bindings.h)
// =============================================================================
// Three Bwd_gather_tc passes with permuted roles, each owning one input's
// gradient by direct store. The one softmax is normalized over the query i,
// so the Q pass runs BWD_QUERY_ANCHOR and the R/S passes BWD_QUERY_ROWS; the
// Jacobian correction is the FA2 delta = rowsum(dY o Y), computed once here.
//
//   pass  anchor/rows/cols   query   stores
//   Q     Q / R / S          anchor  dQ
//   R     R / Q / S          rows    dR, dVr
//   S     S / Q / R          rows    dS, dVs
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
single_gather_backward_cuda(
    at::Tensor dY, at::Tensor Q, at::Tensor R, at::Tensor S, at::Tensor Vr,
    at::Tensor Vs, at::Tensor Y, at::Tensor m, at::Tensor l, at::Tensor mask,
    int64_t window)
{
  single_gather_check({Q, R, S, Vr, Vs, dY, Y}, mask);
  TORCH_CHECK(window <= 0 || mask.defined(),
              "single_gather: window metadata needs the mask it describes");
  const int B = Q.size(0), H = Q.size(1), N = Q.size(2), D = Q.size(3);
  const int win = (int)std::max<int64_t>(window, 0);
  for (const auto& s : {m, l}) {
    TORCH_CHECK(s.defined() && s.is_cuda() && s.device() == Q.device()
                && s.scalar_type() == at::kFloat && s.is_contiguous()
                && s.sizes() == at::IntArrayRef({B, H, N}),
                "single_gather_backward: m/l must be contiguous fp32 [B,H,N] on the input device");
  }
  const float scale = 1.0f / sqrtf((float)D);
  const bool use_mask = mask.defined();
  const bool* mask_ptr = use_mask ? mask.data_ptr<bool>() : nullptr;

  c10::cuda::CUDAGuard guard(Q.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto delta = (dY.to(at::kFloat) * Y.to(at::kFloat)).sum(-1).contiguous();
  auto fp32 = Q.options().dtype(at::kFloat);
  auto gQ = torch::empty({B, H, N, D}, fp32), gR = torch::empty({B, H, N, D}, fp32),
       gS = torch::empty({B, H, N, D}, fp32), gVr = torch::empty({B, H, N, D}, fp32),
       gVs = torch::empty({B, H, N, D}, fp32);

  int major = 0, optin = 0;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, Q.device().index());
  if (major >= 8) cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, Q.device().index());
  // Tile shape: sg_cfg overrides if set, else 2 warps for win <= 32 (at most
  // 32 rows are visible in either pass), else 8 warps x BK 32, falling back to
  // 4 warps at D=64 / 2 at D=128 when smem does not fit. The anchor (Q) and
  // rows (R/S) passes are tuned separately.
  auto& st = att3_tc::state();
  const int dh = st.sg_cfg.bwd_dh ? st.sg_cfg.bwd_dh : 64;
  size_t smem_a = 0, smem_r = 0;
  int thr_a = 0, thr_r = 0;
  bool legal_a = false, legal_r = false;
  const int default_warps = (win > 0 && win <= 32) ? 2 : BTC_WARPS;
  // SINGLE_COL needs the worst-case column-tile width across every row tile
  // the kernel will visit, which sg_col_bounds bounds differently by role: the
  // anchor side ignores the row tile entirely (width <= win), but the rows
  // side's opposite-key window widens toward the first row tile, up to
  // min(bj, win) + win - 1 (see sg_col_bounds' SG_QUERY_ROWS branch).
  auto single_col_for = [&](int role, int rw, int rbk) {
    if (win <= 0) return false;
    if (role == BWD_QUERY_ANCHOR) return win <= rbk;
    return std::min(rw * 16, win) + win - 1 <= rbk;
  };
  auto pick = [&](int role, int cw, int cbk, size_t& smem, int& thr, bool& legal) -> BwdTcKernel {
    if (cw || cbk) {
      const int rw = cw ? cw : default_warps, rbk = cbk ? cbk : BTC_BK;
      return sg_pick_bwd(D, rw, rbk, dh, role, use_mask,
                         optin, smem, thr, legal, single_col_for(role, rw, rbk));
    }
    const bool single_col = single_col_for(role, default_warps, BTC_BK);
    BwdTcKernel k = sg_pick_bwd(D, default_warps, BTC_BK, dh, role, use_mask, optin, smem, thr, legal, single_col);
    if (k == nullptr && legal) {
      const int rw = D == 128 ? 2 : 4;
      k = sg_pick_bwd(D, rw, BTC_BK, dh, role, use_mask, optin, smem, thr, legal, single_col_for(role, rw, BTC_BK));
    }
    return k;
  };
  auto* ka = pick(BWD_QUERY_ANCHOR, st.sg_cfg.bwd_a_warps, st.sg_cfg.bwd_a_bk, smem_a, thr_a, legal_a);
  auto* kr = pick(BWD_QUERY_ROWS, st.sg_cfg.bwd_r_warps, st.sg_cfg.bwd_r_bk, smem_r, thr_r, legal_r);
  TORCH_CHECK(legal_a && legal_r,
              "single_gather_backward: no kernel for warps=", st.sg_cfg.bwd_a_warps, "/",
              st.sg_cfg.bwd_r_warps, " bk=", st.sg_cfg.bwd_a_bk, "/", st.sg_cfg.bwd_r_bk, " dh=", dh,
              " (warps in {2,4,8}, bk in {16,32}, dh 64 or D)");
  TORCH_CHECK(ka != nullptr && kr != nullptr,
              "single_gather_backward: tensor-core path unavailable on this device "
              "(needs sm_80+ and ", std::max(smem_a, smem_r), " B opt-in shared memory, have ", optin, ")");

  const dim3 grid(N, H, B);
  auto bp = [](const at::Tensor& t) {
    return t.defined() ? reinterpret_cast<const bf16*>(t.data_ptr<at::BFloat16>()) : nullptr;
  };
  auto fp = [](const at::Tensor& t) { return t.defined() ? t.data_ptr<float>() : nullptr; };
  ka<<<grid, thr_a, smem_a, stream>>>(
      bp(Q), nullptr, bp(dY),  bp(R), bp(Vr), nullptr,  bp(S), bp(Vs), nullptr,
      fp(m), fp(l), fp(delta),  nullptr, nullptr, nullptr,  nullptr, nullptr, nullptr,
      gQ.data_ptr<float>(), nullptr, mask_ptr, H, N, win, scale, H);
  kr<<<grid, thr_r, smem_r, stream>>>(
      bp(R), bp(Vr), nullptr,  bp(Q), nullptr, bp(dY),  bp(S), bp(Vs), nullptr,
      nullptr, nullptr, nullptr,  fp(m), fp(l), fp(delta),  nullptr, nullptr, nullptr,
      gR.data_ptr<float>(), gVr.data_ptr<float>(), mask_ptr, H, N, win, scale, H);
  kr<<<grid, thr_r, smem_r, stream>>>(
      bp(S), bp(Vs), nullptr,  bp(Q), nullptr, bp(dY),  bp(R), bp(Vr), nullptr,
      nullptr, nullptr, nullptr,  fp(m), fp(l), fp(delta),  nullptr, nullptr, nullptr,
      gS.data_ptr<float>(), gVs.data_ptr<float>(), mask_ptr, H, N, win, scale, H);
  AT_CUDA_CHECK(cudaGetLastError());

  ++st.sg_bwd_anchor_launches;
  st.sg_bwd_rows_launches += 2;
  st.sg_last_bwd = "D=" + std::to_string(D) + " warps=" + std::to_string(thr_a / 32) + "/"
                 + std::to_string(thr_r / 32) + " bk="
                 + std::to_string(st.sg_cfg.bwd_a_bk ? st.sg_cfg.bwd_a_bk : BTC_BK) + "/"
                 + std::to_string(st.sg_cfg.bwd_r_bk ? st.sg_cfg.bwd_r_bk : BTC_BK)
                 + " dh=" + std::to_string(dh) + " masked=" + std::to_string(use_mask)
                 + " win=" + std::to_string(win) + " roles=anchor,rows,rows";
  return std::make_tuple(gQ.to(at::kBFloat16), gR.to(at::kBFloat16), gS.to(at::kBFloat16),
                         gVr.to(at::kBFloat16), gVs.to(at::kBFloat16));
}


// =============================================================================
// Shared-KV backward (Hkv = 1, D = 128)
// =============================================================================
// Same three passes as single_gather_backward_cuda with the KV operands read in
// place through the kernel's KV-head offset. The R/S passes write per-query-head
// partials [B,Hq,N,D] (each CTA owns its row, so no atomics); the head
// reduction rounds each partial to bf16, sums the heads in fp32, casts once.
#include "shared_reduction.cuh"
#include "shared_hopper.h"
#ifdef ATT3NTION_WITH_HOPPER
#include "shared_hopper_rs.cuh"
#include "shared_hopper_rs64.cuh"
#endif
#include "shared_retained_rs.cuh"
#include "shared_retained_dq.cuh"
#include "shared_mask_metadata.cuh"

#include "shared_wide_dq.cuh"

// Validate every raw-pointer operand, including its device, before padding or
// any launch.
static void shared_backward_check_state(
    const at::Tensor& Q, const at::Tensor& dY, const at::Tensor& Y,
    const at::Tensor& m, const at::Tensor& l) {
  for (const auto& x : {dY, Y}) {
    TORCH_CHECK(x.defined() && x.is_cuda() && x.device() == Q.device()
                && x.scalar_type() == at::kBFloat16 && x.is_contiguous()
                && x.sizes() == Q.sizes(),
                "single_gather_shared_backward: dY and Y must be contiguous bf16 matching Q on Q's CUDA device");
  }
  for (const auto& s : {m, l}) {
    TORCH_CHECK(s.defined() && s.is_cuda() && s.device() == Q.device()
                && s.scalar_type() == at::kFloat && s.is_contiguous()
                && s.sizes() == at::IntArrayRef({Q.size(0), Q.size(1), Q.size(2)}),
                "single_gather_shared_backward: m/l must be contiguous fp32 [B,Hq,N] on Q's CUDA device");
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
single_gather_shared_backward_auto(
    at::Tensor dY, at::Tensor Q, at::Tensor R, at::Tensor S, at::Tensor Vr,
    at::Tensor Vs, at::Tensor Y, at::Tensor m, at::Tensor l, at::Tensor mask,
    int64_t window, int64_t rs_group)
{
  single_gather_shared_check(Q, {R, S, Vr, Vs}, mask, window, 1);
  shared_backward_check_state(Q,dY,Y,m,l);
  rs_group = 4;
  if (Q.size(1) % 2) {
    const int64_t heads = Q.size(1);
    auto extra = torch::zeros({Q.size(0),1,Q.size(2),Q.size(3)},Q.options());
    auto extra_stats = torch::zeros({Q.size(0),1,Q.size(2)},m.options());
    auto result = single_gather_shared_backward_auto(
        torch::cat({dY,extra},1),torch::cat({Q,extra},1),R,S,Vr,Vs,
        torch::cat({Y,extra},1),torch::cat({m,extra_stats},1),
        torch::cat({l,torch::ones_like(extra_stats)},1),mask,window,4);
    std::get<0>(result) = std::get<0>(result).narrow(1,0,heads).contiguous();
    return result;
  }
  const int B = Q.size(0), H = Q.size(1), N = Q.size(2), D = Q.size(3);
  const float scale = 1.0f / sqrtf((float)D);
  const int win = (int)window;
  const bool* mask_ptr = mask.data_ptr<bool>();
  c10::cuda::CUDAGuard guard(Q.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int mask_words = (N+31)/32;
  auto packed_mask = torch::empty({B,N,mask_words}, Q.options().dtype(at::kInt));
  auto support = torch::empty({B,N}, Q.options().dtype(at::kByte));
  auto* packed_ptr = reinterpret_cast<uint32_t*>(packed_mask.data_ptr<int32_t>());
  auto* support_ptr = support.data_ptr<uint8_t>();
  AT_CUDA_CHECK(att3_mask_metadata::launch(mask_ptr,packed_ptr,support_ptr,B,N,win,stream));
  auto delta = torch::empty({B,H,N}, Q.options().dtype(at::kFloat));
  AT_CUDA_CHECK((cudaError_t)fusion_delta(
      reinterpret_cast<const bf16*>(dY.data_ptr<at::BFloat16>()),
      reinterpret_cast<const bf16*>(Y.data_ptr<at::BFloat16>()),
      delta.data_ptr<float>(), B*H*N, stream));
  auto fp32 = Q.options().dtype(at::kFloat);
  // R/S partials are stored as bf16 (fp32 accumulation, one rounding), which is
  // exactly the per-head value the reduction consumes.
  auto gQ = torch::empty({B,H,N,D},fp32);
  auto pR = torch::empty_like(Q), pS = torch::empty_like(Q),
       pVr = torch::empty_like(Q), pVs = torch::empty_like(Q);
  int major = 0, minor = 0, optin = 0;
  AT_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, Q.device().index()));
  AT_CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, Q.device().index()));
  if (major >= 8) cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, Q.device().index());
  // Fallback Q pass: 2 warps x 16 rows cover a w32 window, BK 32, one-pass
  // D=128 output; single column tile when win <= 32.
  size_t smem_a = 0; int thr_a = 0; bool legal = false;
  auto* ka = sg_pick_bwd(128, 2, 32, 128, BWD_QUERY_ANCHOR, true, optin, smem_a, thr_a, legal, win <= 32);
  TORCH_CHECK(ka != nullptr, "single_gather_shared_backward: Q pass unavailable on this device");
  auto bp = [](const at::Tensor& t) { return reinterpret_cast<const bf16*>(t.data_ptr<at::BFloat16>()); };
  auto fp = [](const at::Tensor& t) { return t.data_ptr<float>(); };
  const bool used_retained_dq = launch_att3_shared_hopper_dq(
      bp(Q),bp(dY),bp(R),bp(Vr),bp(S),bp(Vs),fp(m),fp(l),fp(delta),
      gQ.data_ptr<float>(),mask_ptr,B,H,N,win,scale,stream,support_ptr) || sg_wide_dq::launch_wide_dq(
      bp(Q),bp(dY),bp(R),bp(Vr),bp(S),bp(Vs),fp(m),fp(l),fp(delta),
      gQ.data_ptr<float>(),mask_ptr,B,H,N,win,scale,stream,support_ptr) || sg_retained_dq::launch_retained_dq(
      bp(Q),bp(dY),bp(R),bp(Vr),bp(S),bp(Vs),fp(m),fp(l),fp(delta),
      gQ.data_ptr<float>(),mask_ptr,B,H,N,win,scale,stream,support_ptr);
  if (!used_retained_dq) {
  ka<<<dim3(N, H, B), thr_a, smem_a, stream>>>(
      bp(Q), nullptr, bp(dY),  bp(R), bp(Vr), nullptr,  bp(S), bp(Vs), nullptr,
      fp(m), fp(l), fp(delta),  nullptr, nullptr, nullptr,  nullptr, nullptr, nullptr,
      gQ.data_ptr<float>(), nullptr, mask_ptr, H, N, win, scale, 1);
  }
  bool used_retained = false;
  bool used_hopper_rs = false;
#ifdef ATT3NTION_WITH_HOPPER
  // Hopper wgmma R/S kernels for w32 / w64; other shapes use the retained-MMA
  // kernels. All write the same bf16 partials.
  if (major == 9 && minor == 0 && win == 32 && H >= 16 && H % 4 == 0 && N >= 96) {
    AT_CUDA_CHECK((cudaError_t)att3_shared_rs_wgmma_w32(
        bp(R),bp(Vr),bp(Q),bp(dY),bp(S),bp(Vs),fp(m),fp(l),fp(delta),
        pR.data_ptr<at::BFloat16>(),pVr.data_ptr<at::BFloat16>(),
        pS.data_ptr<at::BFloat16>(),pVs.data_ptr<at::BFloat16>(),
        B,H,N,win,scale,stream,support_ptr,packed_ptr,true));
    used_hopper_rs = used_retained = true;
  }
  if (!used_retained && major == 9 && minor == 0 && win == 64 && H % 4 == 0
      && N >= 128 && int64_t(B)*H*N >= 6144) {
    AT_CUDA_CHECK((cudaError_t)att3_shared_rs_wgmma64_w64(
        bp(R),bp(Vr),bp(Q),bp(dY),bp(S),bp(Vs),fp(m),fp(l),fp(delta),
        pR.data_ptr<at::BFloat16>(),pVr.data_ptr<at::BFloat16>(),
        pS.data_ptr<at::BFloat16>(),pVs.data_ptr<at::BFloat16>(),
        B,H,N,win,scale,stream,support_ptr,packed_ptr,true));
    used_hopper_rs = used_retained = true;
  }
#endif
  if (!used_retained) {
    used_retained = att3_shared_rs::launch_retained_rs(
        bp(R),bp(Vr),bp(Q),bp(dY),bp(S),bp(Vs),fp(m),fp(l),fp(delta),
        reinterpret_cast<float*>(pR.data_ptr<at::BFloat16>()),
        reinterpret_cast<float*>(pVr.data_ptr<at::BFloat16>()),
        reinterpret_cast<float*>(pS.data_ptr<at::BFloat16>()),
        reinterpret_cast<float*>(pVs.data_ptr<at::BFloat16>()),
        mask_ptr,B,H,N,win,scale,optin,stream,support_ptr,packed_ptr,mask_words);
    TORCH_CHECK(used_retained,"single_gather_shared_backward: packed R/S schedule unavailable on this device");
  }
  AT_CUDA_CHECK(cudaGetLastError());
  // Head reduction: bf16 per head, fp32 sum, bf16 once.
  auto dQ = torch::empty_like(Q);
  auto dR = torch::empty_like(R), dS = torch::empty_like(S),
       dVr = torch::empty_like(Vr), dVs = torch::empty_like(Vs);
  auto outp = [](at::Tensor& t) { return reinterpret_cast<bf16*>(t.data_ptr<at::BFloat16>()); };
  AT_CUDA_CHECK((cudaError_t)fusion_reduce_split(8,
      fp(gQ), reinterpret_cast<float*>(pR.data_ptr<at::BFloat16>()), reinterpret_cast<float*>(pS.data_ptr<at::BFloat16>()), reinterpret_cast<float*>(pVr.data_ptr<at::BFloat16>()), reinterpret_cast<float*>(pVs.data_ptr<at::BFloat16>()),
      outp(dQ), outp(dR), outp(dS), outp(dVr), outp(dVs), B,H,N*D,stream));
  auto& st = att3_tc::state();
  ++st.sg_shared_bwd_launches;
  st.sg_last_shared_bwd = std::string("auto D=128 Hkv=1 R/S=")
      + (used_hopper_rs ? (win == 64 ? "hopper-shared-projection-w64" : "hopper-register-A-moving48") : "retained-mma")
      + (used_retained_dq ? " dQ=retained" : " dQ=legacy")
      + " win=" + std::to_string(win)
      + " partials=bf16 reduce=fused-bf16-per-head/fp32-split4-sum delta=warp128";
  return std::make_tuple(dQ,dR,dS,dVr,dVs);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
single_gather_shared_backward_cuda(
    at::Tensor dY, at::Tensor Q, at::Tensor R, at::Tensor S, at::Tensor Vr,
    at::Tensor Vs, at::Tensor Y, at::Tensor m, at::Tensor l, at::Tensor mask,
    int64_t window, int64_t rs_group)
{
  if (rs_group == 0) {
    single_gather_shared_check(Q, {R,S,Vr,Vs}, mask, window, 1);
    c10::cuda::CUDAGuard auto_guard(Q.device());
    int major = 0;
    AT_CUDA_CHECK(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,Q.device().index()));
    if (major == 9) return single_gather_shared_backward_auto(dY,Q,R,S,Vr,Vs,Y,m,l,mask,window,4);
    rs_group = 1;
  }
  single_gather_shared_check(Q, {R, S, Vr, Vs}, mask, window, rs_group);
  shared_backward_check_state(Q,dY,Y,m,l);
  const int B = Q.size(0), H = Q.size(1), N = Q.size(2), D = Q.size(3);
  const float scale = 1.0f / sqrtf((float)D);
  const int win = (int)window;
  const bool* mask_ptr = mask.data_ptr<bool>();
  c10::cuda::CUDAGuard guard(Q.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto delta = (dY.to(at::kFloat) * Y.to(at::kFloat)).sum(-1).contiguous();
  auto fp32 = Q.options().dtype(at::kFloat);
  auto gQ = torch::empty({B, H, N, D}, fp32), pR = torch::empty({B, H, N, D}, fp32),
       pS = torch::empty({B, H, N, D}, fp32), pVr = torch::empty({B, H, N, D}, fp32),
       pVs = torch::empty({B, H, N, D}, fp32);
  int major = 0, optin = 0;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, Q.device().index());
  if (major >= 8) cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, Q.device().index());
  // Q pass: 2 warps x 16 rows cover a w32 window, BK 32, one-pass D=128
  // output; single column tile when win <= 32.
  size_t smem_a = 0; int thr_a = 0; bool legal = false;
  auto* ka = sg_pick_bwd(128, 2, 32, 128, BWD_QUERY_ANCHOR, true, optin, smem_a, thr_a, legal, win <= 32);
  TORCH_CHECK(ka != nullptr, "single_gather_shared_backward: Q pass unavailable on this device");
  auto bp = [](const at::Tensor& t) { return reinterpret_cast<const bf16*>(t.data_ptr<at::BFloat16>()); };
  auto fp = [](const at::Tensor& t) { return t.data_ptr<float>(); };
  ka<<<dim3(N, H, B), thr_a, smem_a, stream>>>(
      bp(Q), nullptr, bp(dY),  bp(R), bp(Vr), nullptr,  bp(S), bp(Vs), nullptr,
      fp(m), fp(l), fp(delta),  nullptr, nullptr, nullptr,  nullptr, nullptr, nullptr,
      gQ.data_ptr<float>(), nullptr, mask_ptr, H, N, win, scale, 1);
  if (rs_group == 1) {
    // R/S-owned passes: 2 warps, BK 16, one-pass output.
    size_t smem_r = 0; int thr_r = 0;
    auto* kr = sg_pick_bwd(128, 2, 16, 128, BWD_QUERY_ROWS, true, optin, smem_r, thr_r, legal);
    TORCH_CHECK(kr != nullptr, "single_gather_shared_backward: R/S pass unavailable on this device");
    kr<<<dim3(N, H, B), thr_r, smem_r, stream>>>(
        bp(R), bp(Vr), nullptr,  bp(Q), nullptr, bp(dY),  bp(S), bp(Vs), nullptr,
        nullptr, nullptr, nullptr,  fp(m), fp(l), fp(delta),  nullptr, nullptr, nullptr,
        pR.data_ptr<float>(), pVr.data_ptr<float>(), mask_ptr, H, N, win, scale, 1);
    kr<<<dim3(N, H, B), thr_r, smem_r, stream>>>(
        bp(S), bp(Vs), nullptr,  bp(Q), nullptr, bp(dY),  bp(R), bp(Vr), nullptr,
        nullptr, nullptr, nullptr,  fp(m), fp(l), fp(delta),  nullptr, nullptr, nullptr,
        pS.data_ptr<float>(), pVs.data_ptr<float>(), mask_ptr, H, N, win, scale, 1);
  } else {
    TORCH_CHECK(launch_bwd_rows_shared_grouped(Q, dY, R, Vr, S, Vs, m, l, delta, pR, pVr, pS, pVs, mask_ptr,
                                               B, H, N, scale, optin, stream, win, (int)rs_group),
                "single_gather_shared_backward: grouped R/S pass unavailable on this device");
  }
  AT_CUDA_CHECK(cudaGetLastError());
  // Head reduction: bf16 per head, fp32 sum, bf16 once.
  auto reduce = [&](const at::Tensor& part) {
    return part.to(at::kBFloat16).to(at::kFloat).view({B, 1, H, N, D}).sum(2).to(at::kBFloat16);
  };
  auto& st = att3_tc::state();
  ++st.sg_shared_bwd_launches;
  st.sg_last_shared_bwd = "D=128 Hkv=1 Q=Bwd_gather_tc<128,masked,2,32,anchor,dh128> rs_group=" + std::to_string(rs_group)
                        + (rs_group == 1 ? " R/S=Bwd_gather_tc<128,masked,2,16,rows,dh128>" : (rs_group == 4 ? " R/S=dual<2heads,2directions,WPH1,BK16,rawQD>" : " R/S=grouped<G2,WPH2,BK32,rawQD>"))
                        + " win=" + std::to_string(win) + " reduce=bf16-per-head/fp32-sum";
  return std::make_tuple(gQ.to(at::kBFloat16), reduce(pR), reduce(pS), reduce(pVr), reduce(pVs));
}
