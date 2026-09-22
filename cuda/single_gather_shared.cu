/**
 * @file single_gather_shared.cu
 * @brief Grouped shared-KV kernels (D=128, Hkv=1): one CTA serves G consecutive
 * query heads of one (batch, KV head) and loads each raw KV tile once for all
 * of them. Everything head-dependent stays private to the head.
 *
 * Copyright (c) 2026 Springtail AI. MIT License.
 */
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "common.cuh"
#include "../cpp/cuda_bindings.h"

#include "shared_retained_fwd.cuh"
#include "shared_hopper.h"

namespace {

constexpr int SG_D = 128;
constexpr int SG_DPAD = SG_D + 8;
constexpr int SG_WPH = 2;                 // warps per head
constexpr int SG_BJ = SG_WPH * 16;        // rows per head tile
constexpr int SG_BK = 16;                 // cols per tile
constexpr float SG_MASKED_THRESH = -5e29f;

// BK stays 16 even at w32: two double-buffered tiles let the second tile's
// cp.async overlap the first tile's MMAs, which beats one BK=32 tile.
constexpr size_t fwd_grouped_smem(int G) {
    // rowp[G] + raw rows + V rows + 2 x (cols, V cols) in bf16; col_mul, row_mul,
    // anchor[G], wN[G][WPH], wML[G][WPH], redN[G], redML[G] in fp32.
    return sizeof(bf16) * ((size_t)(G + 2) * SG_BJ * SG_DPAD + (size_t)4 * SG_BK * SG_DPAD)
         + sizeof(float) * ((size_t)2 * SG_BK + SG_BJ + (size_t)G * (SG_D + SG_WPH * SG_D + SG_WPH * 2 + SG_D + 2));
}

// Grouped forward: block = (query i, G heads, batch b).
template<int G>
__global__ __launch_bounds__(G * SG_WPH * 32)
void Y_gather_tc_grouped(
    const bf16* __restrict__ Q, const bf16* __restrict__ R, const bf16* __restrict__ S,
    const bf16* __restrict__ Vr, const bf16* __restrict__ Vs,
    bf16* __restrict__ Y, float* __restrict__ m_out, float* __restrict__ l_out,
    const bool* __restrict__ mask, int H, int N, float scale, int win)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int D = SG_D, DPAD = SG_DPAD, BJ = SG_BJ, BK = SG_BK, WPH = SG_WPH;
    constexpr int KS = D / 16, NT = D / 8, CT = BK / 8, DV = D / 8;
    constexpr int NTHR = G * WPH * 32;

    const int i = blockIdx.x;
    const int h0 = blockIdx.y * G;
    const int b = blockIdx.z;
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int lh = warp / WPH;            // local head
    const int rw = (warp % WPH) * 16;     // row tile within the head's BJ rows
    const int tid_h = tid % (WPH * 32);   // thread index within the head's warps
    const int g = lane / 4, tig = lane % 4;
    const int lrow = (lane & 7) + ((lane >> 3) & 1) * 8, lcol8 = (lane >> 4) * 8;
    const int brow = (lane & 7) + ((lane >> 4) & 1) * 8, bcol8 = ((lane >> 3) & 1) * 8;

    extern __shared__ char smem_raw[];
    bf16* rowp_sm  = reinterpret_cast<bf16*>(smem_raw);                 // [G][BJ][DPAD] scale*Q_h o R
    bf16* rows_sm  = rowp_sm + (size_t)G * BJ * DPAD;                   // [BJ][DPAD] raw R (shared)
    bf16* v_rows_sm = rows_sm + BJ * DPAD;                              // [BJ][DPAD] Vr (shared)
    bf16* cols_sm  = v_rows_sm + BJ * DPAD;                             // [2][BK][DPAD] S (shared)
    bf16* v_cols_sm = cols_sm + 2 * BK * DPAD;                          // [2][BK][DPAD] Vs (shared)
    float* col_mul = reinterpret_cast<float*>(v_cols_sm + 2 * BK * DPAD);   // [2][BK]
    float* row_mul = col_mul + 2 * BK;                                  // [BJ]
    float* anchor_sm = row_mul + BJ;                                    // [G][D]
    float* wN  = anchor_sm + G * D;                                     // [G][WPH][D]
    float* wML = wN + G * WPH * D;                                      // [G][WPH][2]
    float* redN = wML + G * WPH * 2;                                    // [G][D]
    float* redML = redN + G * D;                                        // [G][2]

    const int64_t kv_off = (int64_t)b * N * D;                          // Hkv = 1
    const bool* mrow = mask + ((int64_t)b * N + i) * N;

    for (int t = tid; t < G * D; t += NTHR) {
        const int hh = t / D, d = t % D;
        anchor_sm[t] = scale * bf2f(Q[(((int64_t)b * H + h0 + hh) * N + i) * D + d]);
    }

    auto stage_cols = [&](int k0, int buf) {
        bf16* cs = cols_sm + buf * BK * DPAD;
        bf16* vs = v_cols_sm + buf * BK * DPAD;
        for (int idx = tid; idx < BK * DV; idx += NTHR) {
            const int kl = idx / DV, dv = (idx % DV) * 8;
            const int k = k0 + kl;
            if (k < N) {
                cp_async16(cs + kl * DPAD + dv, S + kv_off + (int64_t)k * D + dv);
                cp_async16(vs + kl * DPAD + dv, Vs + kv_off + (int64_t)k * D + dv);
            } else {
                const uint4 z = make_uint4(0, 0, 0, 0);
                *reinterpret_cast<uint4*>(cs + kl * DPAD + dv) = z;
                *reinterpret_cast<uint4*>(vs + kl * DPAD + dv) = z;
            }
        }
        for (int kl = tid; kl < BK; kl += NTHR) {
            const int k = k0 + kl;
            col_mul[buf * BK + kl] = (k < N && mrow[k]) ? 1.0f : 0.0f;
        }
    };

    int j_lo, j_hi;
    sg_row_bounds(SG_QUERY_ANCHOR, i, N, win, BJ, j_lo, j_hi);
    for (int j0 = j_lo; j0 < j_hi; j0 += BJ) {
        __syncthreads();
        int k_lo, k_hi;
        sg_col_bounds(SG_QUERY_ANCHOR, i, j0, BJ, N, win, BK, k_lo, k_hi);

        // Raw R / Vr rows once per block (shared by every head).
        for (int idx = tid; idx < BJ * DV; idx += NTHR) {
            const int jl = idx / DV, dv = (idx % DV) * 8;
            const int j = j0 + jl;
            uint4 rp = make_uint4(0, 0, 0, 0), vp = rp;
            if (j < N) {
                const int64_t off = kv_off + (int64_t)j * D + dv;
                rp = *reinterpret_cast<const uint4*>(R + off);
                vp = *reinterpret_cast<const uint4*>(Vr + off);
            }
            *reinterpret_cast<uint4*>(rows_sm + jl * DPAD + dv) = rp;
            *reinterpret_cast<uint4*>(v_rows_sm + jl * DPAD + dv) = vp;
        }
        for (int jl = tid; jl < BJ; jl += NTHR) {
            const int j = j0 + jl;
            row_mul[jl] = (j < N && mrow[j]) ? 1.0f : 0.0f;
        }
        stage_cols(k_lo, 0);
        asm volatile("cp.async.wait_all;\n" ::);
        __syncthreads();

        // Head-scaled row tile: same fp32 product and single bf16 rounding as
        // the per-head kernel.
        {
            const float* anc = anchor_sm + lh * D;
            bf16* rp_h = rowp_sm + (size_t)lh * BJ * DPAD;
            for (int idx = tid_h; idx < BJ * DV; idx += WPH * 32) {
                const int jl = idx / DV, dv = (idx % DV) * 8;
                uint4 pack = *reinterpret_cast<const uint4*>(rows_sm + jl * DPAD + dv);
                __nv_bfloat162* pairs = reinterpret_cast<__nv_bfloat162*>(&pack);
                #pragma unroll
                for (int e = 0; e < 4; e++) {
                    const float2 rf = __bfloat1622float2(pairs[e]);
                    pairs[e] = __floats2bfloat162_rn(anc[dv + 2 * e] * rf.x, anc[dv + 2 * e + 1] * rf.y);
                }
                *reinterpret_cast<uint4*>(rp_h + jl * DPAD + dv) = pack;
            }
        }
        __syncthreads();

        const bf16* rowp_h = rowp_sm + (size_t)lh * BJ * DPAD;
        const float rm0 = row_mul[rw + g], rm1 = row_mul[rw + g + 8];
        uint32_t a_rowp[KS][4];
        #pragma unroll
        for (int ks = 0; ks < KS; ks++) {
            ldmatrix_x4(a_rowp[ks], rowp_h + (rw + lrow) * DPAD + ks * 16 + lcol8);
        }
        float m0 = NEG_INF, m1 = NEG_INF, l0 = 0.0f, l1 = 0.0f;
        float U[NT][4];
        #pragma unroll
        for (int nt = 0; nt < NT; nt++) { U[nt][0] = U[nt][1] = U[nt][2] = U[nt][3] = 0.0f; }

        int cur = 0;
        for (int k0 = k_lo; k0 < k_hi; k0 += BK) {
            const int nxt = cur ^ 1;
            if (k0 + BK < k_hi) stage_cols(k0 + BK, nxt);
            const bf16* cols_cur = cols_sm + cur * BK * DPAD;
            const bf16* v_cols_cur = v_cols_sm + cur * BK * DPAD;

            float acc[CT][4];
            #pragma unroll
            for (int nt = 0; nt < CT; nt++) { acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.0f; }
            #pragma unroll
            for (int p = 0; p < BK / 16; p++) {
                const bf16* bp = cols_cur + (p * 16 + brow) * DPAD + bcol8;
                #pragma unroll
                for (int ks = 0; ks < KS; ks++) {
                    uint32_t bfr[4];
                    ldmatrix_x4(bfr, bp + ks * 16);
                    mma_bf16_m16n8k16(acc[2 * p], a_rowp[ks], bfr);
                    mma_bf16_m16n8k16(acc[2 * p + 1], a_rowp[ks], bfr + 2);
                }
            }
            float mt0 = NEG_INF, mt1 = NEG_INF;
            #pragma unroll
            for (int nt = 0; nt < CT; nt++) {
                const int kc = cur * BK + nt * 8 + 2 * tig;
                const float k0f = col_mul[kc], k1f = col_mul[kc + 1];
                acc[nt][0] = (rm0 * k0f > 0.5f) ? acc[nt][0] : NEG_INF;
                acc[nt][1] = (rm0 * k1f > 0.5f) ? acc[nt][1] : NEG_INF;
                acc[nt][2] = (rm1 * k0f > 0.5f) ? acc[nt][2] : NEG_INF;
                acc[nt][3] = (rm1 * k1f > 0.5f) ? acc[nt][3] : NEG_INF;
                mt0 = fmaxf(mt0, fmaxf(acc[nt][0], acc[nt][1]));
                mt1 = fmaxf(mt1, fmaxf(acc[nt][2], acc[nt][3]));
            }
            #pragma unroll
            for (int off = 1; off <= 2; off <<= 1) {
                mt0 = fmaxf(mt0, __shfl_xor_sync(0xFFFFFFFF, mt0, off));
                mt1 = fmaxf(mt1, __shfl_xor_sync(0xFFFFFFFF, mt1, off));
            }
            const float mn0 = fmaxf(m0, mt0), mn1 = fmaxf(m1, mt1);
            const float a0 = __expf(m0 - mn0), a1 = __expf(m1 - mn1);
            l0 *= a0; l1 *= a1;
            #pragma unroll
            for (int nt = 0; nt < NT; nt++) { U[nt][0] *= a0; U[nt][1] *= a0; U[nt][2] *= a1; U[nt][3] *= a1; }
            m0 = mn0; m1 = mn1;

            uint32_t pfr[BK / 16][4];
            #pragma unroll
            for (int s2 = 0; s2 < BK / 16; s2++) {
                #pragma unroll
                for (int half = 0; half < 2; half++) {
                    const int nt = 2 * s2 + half;
                    const float p0 = (acc[nt][0] < SG_MASKED_THRESH) ? 0.0f : __expf(acc[nt][0] - mn0);
                    const float p1 = (acc[nt][1] < SG_MASKED_THRESH) ? 0.0f : __expf(acc[nt][1] - mn0);
                    const float p2 = (acc[nt][2] < SG_MASKED_THRESH) ? 0.0f : __expf(acc[nt][2] - mn1);
                    const float p3 = (acc[nt][3] < SG_MASKED_THRESH) ? 0.0f : __expf(acc[nt][3] - mn1);
                    l0 += p0 + p1; l1 += p2 + p3;
                    pfr[s2][2 * half + 0] = pack_bf162(p0, p1);
                    pfr[s2][2 * half + 1] = pack_bf162(p2, p3);
                }
            }
            #pragma unroll
            for (int s2 = 0; s2 < BK / 16; s2++) {
                const bf16* bp = v_cols_cur + (s2 * 16 + lrow) * DPAD + lcol8;
                #pragma unroll
                for (int np = 0; np < NT / 2; np++) {
                    uint32_t bfr[4];
                    ldmatrix_x4_trans(bfr, bp + np * 16);
                    mma_bf16_m16n8k16(U[2 * np], pfr[s2], bfr);
                    mma_bf16_m16n8k16(U[2 * np + 1], pfr[s2], bfr + 2);
                }
            }
            asm volatile("cp.async.wait_all;\n" ::);
            __syncthreads();
            cur = nxt;
        }

        // ---- epilogue: Vr-weighted row collapse of this warp's 16 rows ----
        #pragma unroll
        for (int off = 1; off <= 2; off <<= 1) {
            l0 += __shfl_xor_sync(0xFFFFFFFF, l0, off);
            l1 += __shfl_xor_sync(0xFFFFFFFF, l1, off);
        }
        float Mw = fmaxf(m0, m1);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) Mw = fmaxf(Mw, __shfl_xor_sync(0xFFFFFFFF, Mw, off));
        const float w0 = __expf(m0 - Mw), w1 = __expf(m1 - Mw);
        float Lw = w0 * l0 + w1 * l1;
        #pragma unroll
        for (int off = 4; off <= 16; off <<= 1) Lw += __shfl_xor_sync(0xFFFFFFFF, Lw, off);

        float nacc[2 * NT];
        const bf16* v1r0 = v_rows_sm + (rw + g) * DPAD + 2 * tig;
        const bf16* v1r1 = v_rows_sm + (rw + g + 8) * DPAD + 2 * tig;
        #pragma unroll
        for (int nt = 0; nt < NT; nt++) {
            const float2 v10 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(v1r0 + nt * 8));
            const float2 v11 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(v1r1 + nt * 8));
            nacc[2 * nt + 0] = w0 * v10.x * U[nt][0] + w1 * v11.x * U[nt][2];
            nacc[2 * nt + 1] = w0 * v10.y * U[nt][1] + w1 * v11.y * U[nt][3];
        }
        #pragma unroll
        for (int off = 4; off <= 16; off <<= 1) {
            #pragma unroll
            for (int e = 0; e < 2 * NT; e++) nacc[e] += __shfl_xor_sync(0xFFFFFFFF, nacc[e], off);
        }
        float* wN_h = wN + (lh * WPH + warp % WPH) * D;
        if (lane < 4) {
            #pragma unroll
            for (int nt = 0; nt < NT; nt++) {
                wN_h[nt * 8 + 2 * lane] = nacc[2 * nt + 0];
                wN_h[nt * 8 + 2 * lane + 1] = nacc[2 * nt + 1];
            }
        }
        if (lane == 0) { wML[(lh * WPH + warp % WPH) * 2] = Mw; wML[(lh * WPH + warp % WPH) * 2 + 1] = Lw; }
        __syncthreads();

        // ---- combine the head's warps into (M, L, N). The grouped path runs
        // only for win <= 32 == BJ, so this is the only row tile and the result
        // is written outright instead of merged into a running state.
        {
            const float* wML_h = wML + lh * WPH * 2;
            const float* wN_hh = wN + lh * WPH * D;
            float* redN_h = redN + lh * D;
            float* redML_h = redML + lh * 2;
            float Mnew = NEG_INF;
            #pragma unroll
            for (int wd = 0; wd < WPH; wd++) Mnew = fmaxf(Mnew, wML_h[wd * 2]);
            float nNew0 = 0.0f, nNew1 = 0.0f;       // two channels per thread (64 threads, D = 128)
            const int c0 = tid_h, c1 = tid_h + WPH * 32;
            #pragma unroll
            for (int wd = 0; wd < WPH; wd++) {
                const float ew = __expf(wML_h[wd * 2] - Mnew);
                nNew0 += ew * wN_hh[wd * D + c0];
                nNew1 += ew * wN_hh[wd * D + c1];
            }
            float lNew = 0.0f;
            #pragma unroll
            for (int wd = 0; wd < WPH; wd++) lNew += __expf(wML_h[wd * 2] - Mnew) * wML_h[wd * 2 + 1];
            __syncthreads();
            redN_h[c0] = nNew0;
            redN_h[c1] = nNew1;
            if (tid_h == 0) { redML_h[0] = Mnew; redML_h[1] = lNew; }
        }
    }
    __syncthreads();

    {
        const float* redN_h = redN + lh * D;
        const float Lfin = redML[lh * 2 + 1];
        const float inv = (Lfin > 1e-20f) ? (1.0f / Lfin) : 0.0f;
        const int64_t ybase = ((int64_t)b * H + h0 + lh) * N + i;
        for (int d = tid_h; d < D; d += WPH * 32) Y[ybase * D + d] = f2bf(redN_h[d] * inv);
        if (tid_h == 0) { m_out[ybase] = redML[lh * 2]; l_out[ybase] = Lfin; }
    }
#endif
}

// Grouped R/S-owned backward pass: block = (KV position a, G query heads, batch b).
// G2 keeps the per-head kernel's BJ32 summation order; G4 uses BJ16, which
// changes it. With RAW, raw Q/dY of the current row tile stay in shared memory
// for the row collapse.
constexpr size_t bwd_grouped_smem(int G) {
    const int WPH = G == 4 ? 1 : 2, BK = G == 4 ? 16 : 32;
    const int BJ = WPH * 16, D = 128, DPAD = 136;
    return sizeof(bf16) * ((size_t)4 * G * BJ * DPAD + (size_t)4 * BK * DPAD)
         + sizeof(float) * ((size_t)2 * D + (size_t)G * 3 * BJ + (size_t)G * WPH * 2 * D + (size_t)G * 2 * D)
         + sizeof(uint32_t) * 2 * BJ;
}

template<int G, int WPH, int BK, bool RAW=false>
__global__ __launch_bounds__(G * WPH * 32, 1)
void Bwd_rows_tc_grouped(
    const bf16* __restrict__ Xa_bf,   // anchor stream (R or S), [B,1,N,D]
    const bf16* __restrict__ Va_bf,   // its values (Vr or Vs)
    const bf16* __restrict__ Xr_bf,   // Q [B,H,N,D]
    const bf16* __restrict__ gYr_bf,  // dY [B,H,N,D]
    const bf16* __restrict__ Xc_bf,   // other key stream (S or R), [B,1,N,D]
    const bf16* __restrict__ Vc_bf,   // its values
    const float* __restrict__ m_r, const float* __restrict__ l_r, const float* __restrict__ sum_r,   // [B,H,N]
    float* __restrict__ gradXa, float* __restrict__ gradVa,   // per-head partials [B,H,N,D]
    const bool* __restrict__ mask, int H, int N, float scale, int win)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int D = 128, DPAD = 136, BJ = WPH * 16;
    constexpr int KS = D / 16, DH = D, DV = D / 8, NOUT = 2;
    constexpr int NTHR = G * WPH * 32;

    const int a = blockIdx.x;
    const int h0 = blockIdx.y * G;
    const int b = blockIdx.z;
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int lh = warp / WPH;
    const int jw = (warp % WPH) * 16;
    const int tid_h = tid % (WPH * 32);
    const int g = lane / 4, tig = lane % 4;
    const int lrow = (lane & 7) + ((lane >> 3) & 1) * 8, lcol8 = (lane >> 4) * 8;
    const int brow = (lane & 7) + ((lane >> 4) & 1) * 8, bcol8 = ((lane >> 3) & 1) * 8;

    extern __shared__ char smem_raw[];
    bf16* a0_sm = reinterpret_cast<bf16*>(smem_raw);            // [G][BJ][DPAD] scale*Q_h o Xa
    bf16* a2_sm = a0_sm + (size_t)G * BJ * DPAD;                // [G][BJ][DPAD] dY_h o Va
    bf16* rawQ_sm = a2_sm + (size_t)G * BJ * DPAD;
    bf16* rawDY_sm = rawQ_sm + (RAW ? (size_t)G * BJ * DPAD : 0);
    bf16* xc_sm = rawDY_sm + (RAW ? (size_t)G * BJ * DPAD : 0);                // [2][BK][DPAD]
    bf16* vc_sm = xc_sm + 2 * BK * DPAD;                        // [2][BK][DPAD]
    float* anchX = reinterpret_cast<float*>(vc_sm + 2 * BK * DPAD);   // [D]
    float* anchV = anchX + D;                                   // [D]
    float* mr_sm = anchV + D;                                   // [G][BJ]
    float* ilr_sm = mr_sm + G * BJ;
    float* sr_sm = ilr_sm + G * BJ;
    float* wOut = sr_sm + G * BJ;                               // [G][WPH][NOUT*D]
    float* redOut = wOut + (size_t)G * WPH * NOUT * D;          // [G][NOUT*D]
    uint32_t* msk_sm = reinterpret_cast<uint32_t*>(redOut + (size_t)G * NOUT * D);   // [2][BJ]

    const int64_t kv_off = (int64_t)b * N * D;
    const int64_t q_off_h = ((int64_t)b * H + h0 + lh) * N * D;
    const int64_t st_off_h = ((int64_t)b * H + h0 + lh) * N;
    const bool* mb = mask + (int64_t)b * N * N;

    for (int d = tid; d < D; d += NTHR) {
        anchX[d] = scale * bf2f(Xa_bf[kv_off + (int64_t)a * D + d]);
        anchV[d] = bf2f(Va_bf[kv_off + (int64_t)a * D + d]);
    }
    for (int t = tid; t < G * NOUT * D; t += NTHR) redOut[t] = 0.0f;

    int j_lo, j_hi;
    sg_row_bounds(SG_QUERY_ROWS, a, N, win, BJ, j_lo, j_hi);
    for (int j0 = j_lo; j0 < j_hi; j0 += BJ) {
        __syncthreads();
        int k_lo, k_hi;
        sg_col_bounds(SG_QUERY_ROWS, a, j0, BJ, N, win, BK, k_lo, k_hi);

        auto stage_cols = [&](int k0, int buf) {
            bf16* xs = xc_sm + buf * BK * DPAD;
            bf16* vs = vc_sm + buf * BK * DPAD;
            for (int idx = tid; idx < BK * DV; idx += NTHR) {
                const int kl = idx / DV, dv = (idx % DV) * 8;
                const int k = k0 + kl;
                if (k < N) {
                    const int64_t off = kv_off + (int64_t)k * D + dv;
                    cp_async16(xs + kl * DPAD + dv, Xc_bf + off);
                    cp_async16(vs + kl * DPAD + dv, Vc_bf + off);
                } else {
                    const uint4 z = make_uint4(0, 0, 0, 0);
                    *reinterpret_cast<uint4*>(xs + kl * DPAD + dv) = z;
                    *reinterpret_cast<uint4*>(vs + kl * DPAD + dv) = z;
                }
            }
            for (int jl = tid; jl < BJ; jl += NTHR) {
                const int j = j0 + jl;
                msk_sm[buf * BJ + jl] = (j < N) ? sg_pack_mask32(mb + (int64_t)j * N + k0, min(BK, N - k0)) : 0u;
            }
        };

        // Per-head A operands from the head's own Q / dY rows and the shared anchor.
        {
            bf16* a0_h = a0_sm + (size_t)lh * BJ * DPAD;
            bf16* a2_h = a2_sm + (size_t)lh * BJ * DPAD;
            for (int idx = tid_h; idx < BJ * DV; idx += WPH * 32) {
                const int jl = idx / DV, dv = (idx % DV) * 8;
                const int j = j0 + jl;
                uint4 xq = make_uint4(0, 0, 0, 0), gq = xq;
                if (j < N) {
                    const int64_t off = q_off_h + (int64_t)j * D + dv;
                    xq = *reinterpret_cast<const uint4*>(Xr_bf + off);
                    gq = *reinterpret_cast<const uint4*>(gYr_bf + off);
                }
                if constexpr (RAW) {
                    *reinterpret_cast<uint4*>(rawQ_sm + ((size_t)lh * BJ + jl) * DPAD + dv) = xq;
                    *reinterpret_cast<uint4*>(rawDY_sm + ((size_t)lh * BJ + jl) * DPAD + dv) = gq;
                }
                const __nv_bfloat162* xp = reinterpret_cast<const __nv_bfloat162*>(&xq);
                const __nv_bfloat162* gp = reinterpret_cast<const __nv_bfloat162*>(&gq);
                uint4 o0, o2;
                __nv_bfloat162* p0 = reinterpret_cast<__nv_bfloat162*>(&o0);
                __nv_bfloat162* p2 = reinterpret_cast<__nv_bfloat162*>(&o2);
                #pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int d = dv + 2 * e;
                    const float2 x = __bfloat1622float2(xp[e]);
                    const float2 gy = __bfloat1622float2(gp[e]);
                    p0[e] = __floats2bfloat162_rn(x.x * anchX[d], x.y * anchX[d + 1]);
                    p2[e] = __floats2bfloat162_rn(gy.x * anchV[d], gy.y * anchV[d + 1]);
                }
                *reinterpret_cast<uint4*>(a0_h + jl * DPAD + dv) = o0;
                *reinterpret_cast<uint4*>(a2_h + jl * DPAD + dv) = o2;
            }
            for (int jl = tid_h; jl < BJ; jl += WPH * 32) {
                const int j = j0 + jl;
                if (j < N) {
                    mr_sm[lh * BJ + jl] = m_r[st_off_h + j];
                    float il = 1.0f / fmaxf(l_r[st_off_h + j], DENOM_EPS);
                    sr_sm[lh * BJ + jl] = sum_r[st_off_h + j];
                    if (!mb[(int64_t)j * N + a]) il = 0.0f;          // mask[query][anchor]
                    ilr_sm[lh * BJ + jl] = il;
                } else {
                    mr_sm[lh * BJ + jl] = 0.0f; ilr_sm[lh * BJ + jl] = 0.0f; sr_sm[lh * BJ + jl] = 0.0f;
                }
            }
        }
        stage_cols(k_lo, 0);
        asm volatile("cp.async.wait_all;\n" ::);
        __syncthreads();

        const bf16* a0_h = a0_sm + (size_t)lh * BJ * DPAD;
        const bf16* a2_h = a2_sm + (size_t)lh * BJ * DPAD;
        const float mr0 = mr_sm[lh * BJ + jw + g], sr0 = sr_sm[lh * BJ + jw + g];
        const float mr1 = mr_sm[lh * BJ + jw + g + 8], sr1 = sr_sm[lh * BJ + jw + g + 8];
        const float ilr0 = ilr_sm[lh * BJ + jw + g], ilr1 = ilr_sm[lh * BJ + jw + g + 8];
        const bool rpad0 = (j0 + jw + g) >= N, rpad1 = (j0 + jw + g + 8) >= N;

        float Ug[DH / 8][4], U1[DH / 8][4];
        #pragma unroll
        for (int nt = 0; nt < DH / 8; nt++) {
            #pragma unroll
            for (int e = 0; e < 4; e++) { Ug[nt][e] = 0.0f; U1[nt][e] = 0.0f; }
        }

        int cur = 0;
        for (int k0 = k_lo; k0 < k_hi; k0 += BK) {
            const int nxt = cur ^ 1;
            if (k0 + BK < k_hi) stage_cols(k0 + BK, nxt);
            const bf16* xc_cur = xc_sm + cur * BK * DPAD;
            const bf16* vc_cur = vc_sm + cur * BK * DPAD;

            #pragma unroll
            for (int s2 = 0; s2 < BK / 16; s2++) {
                float ax[2][4], adr[2][4];
                #pragma unroll
                for (int nt = 0; nt < 2; nt++) {
                    #pragma unroll
                    for (int e = 0; e < 4; e++) { ax[nt][e] = 0.0f; adr[nt][e] = 0.0f; }
                }
                const bf16* bx = xc_cur + (s2 * 16 + brow) * DPAD + bcol8;
                const bf16* bv = vc_cur + (s2 * 16 + brow) * DPAD + bcol8;
                #pragma unroll
                for (int ks = 0; ks < KS; ks++) {
                    uint32_t A0[4], A2[4], bfr[4];
                    const int roff = (jw + lrow) * DPAD + ks * 16 + lcol8;
                    ldmatrix_x4(A0, a0_h + roff);
                    ldmatrix_x4(A2, a2_h + roff);
                    ldmatrix_x4(bfr, bx + ks * 16);
                    mma_bf16_m16n8k16(ax[0], A0, bfr);
                    mma_bf16_m16n8k16(ax[1], A0, bfr + 2);
                    ldmatrix_x4(bfr, bv + ks * 16);
                    mma_bf16_m16n8k16(adr[0], A2, bfr);
                    mma_bf16_m16n8k16(adr[1], A2, bfr + 2);
                }

                uint32_t gAf[4], Prf[4];
                #pragma unroll
                for (int hf = 0; hf < 2; hf++) {
                    const int cl = s2 * 16 + hf * 8 + 2 * tig;
                    const uint32_t* mw = msk_sm + cur * BJ;
                    const uint32_t rc0 = mw[jw + g] >> cl, rc1 = mw[jw + g + 8] >> cl;
                    float gA[4], Pr[4];
                    #pragma unroll
                    for (int e = 0; e < 4; e++) {
                        const bool hi = (e >= 2), c1 = (e & 1);
                        const float mrr = hi ? mr1 : mr0, ilr = hi ? ilr1 : ilr0, srr = hi ? sr1 : sr0;
                        const float x = ax[hf][e];
                        const float ilrg = (((hi ? rc1 : rc0) >> c1) & 1u) ? ilr : 0.0f;
                        Pr[e] = __expf(fminf(x - mrr, 0.0f)) * ilrg;
                        gA[e] = (adr[hf][e] - srr) * Pr[e];
                    }
                    gAf[2 * hf + 0] = pack_bf162(gA[0], gA[1]);
                    gAf[2 * hf + 1] = pack_bf162(gA[2], gA[3]);
                    Prf[2 * hf + 0] = pack_bf162(Pr[0], Pr[1]);
                    Prf[2 * hf + 1] = pack_bf162(Pr[2], Pr[3]);
                }

                const bf16* px = xc_cur + (s2 * 16 + lrow) * DPAD + lcol8;
                const bf16* pv = vc_cur + (s2 * 16 + lrow) * DPAD + lcol8;
                #pragma unroll
                for (int np = 0; np < DH / 16; np++) {
                    uint32_t bfr[4];
                    ldmatrix_x4_trans(bfr, px + np * 16);
                    mma_bf16_m16n8k16(Ug[2 * np], gAf, bfr);
                    mma_bf16_m16n8k16(Ug[2 * np + 1], gAf, bfr + 2);
                    ldmatrix_x4_trans(bfr, pv + np * 16);
                    mma_bf16_m16n8k16(U1[2 * np], Prf, bfr);
                    mma_bf16_m16n8k16(U1[2 * np + 1], Prf, bfr + 2);
                }
            }
            asm volatile("cp.async.wait_all;\n" ::);
            __syncthreads();
            cur = nxt;
        }

        // ---- epilogue: Hadamard row-collapse of this warp's 16 rows (head's Q / dY rows) ----
        float ng[DH / 4], nv[DH / 4];
        const int64_t r0 = q_off_h + (int64_t)min(j0 + jw + g, N - 1) * D + 2 * tig;
        const int64_t r1 = q_off_h + (int64_t)min(j0 + jw + g + 8, N - 1) * D + 2 * tig;
        auto ld2 = [](const bf16* p, bool pad) {
            return pad ? make_float2(0.0f, 0.0f) : __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(p));
        };
        #pragma unroll
        for (int nt = 0; nt < DH / 8; nt++) {
            const float2 x0 = ld2(RAW ? rawQ_sm + ((size_t)lh * BJ + jw + g) * DPAD + 2 * tig + nt * 8 : Xr_bf + r0 + nt * 8, rpad0), x1 = ld2(RAW ? rawQ_sm + ((size_t)lh * BJ + jw + g + 8) * DPAD + 2 * tig + nt * 8 : Xr_bf + r1 + nt * 8, rpad1);
            ng[2 * nt + 0] = x0.x * Ug[nt][0] + x1.x * Ug[nt][2];
            ng[2 * nt + 1] = x0.y * Ug[nt][1] + x1.y * Ug[nt][3];
            const float2 g0 = ld2(RAW ? rawDY_sm + ((size_t)lh * BJ + jw + g) * DPAD + 2 * tig + nt * 8 : gYr_bf + r0 + nt * 8, rpad0), g1 = ld2(RAW ? rawDY_sm + ((size_t)lh * BJ + jw + g + 8) * DPAD + 2 * tig + nt * 8 : gYr_bf + r1 + nt * 8, rpad1);
            nv[2 * nt + 0] = g0.x * U1[nt][0] + g1.x * U1[nt][2];
            nv[2 * nt + 1] = g0.y * U1[nt][1] + g1.y * U1[nt][3];
        }
        #pragma unroll
        for (int off = 4; off <= 16; off <<= 1) {
            #pragma unroll
            for (int e = 0; e < DH / 4; e++) {
                ng[e] += __shfl_xor_sync(0xFFFFFFFF, ng[e], off);
                nv[e] += __shfl_xor_sync(0xFFFFFFFF, nv[e], off);
            }
        }
        if (lane < 4) {
            float* wo = wOut + ((size_t)lh * WPH + warp % WPH) * NOUT * D;
            #pragma unroll
            for (int nt = 0; nt < DH / 8; nt++) {
                wo[nt * 8 + 2 * lane] = ng[2 * nt + 0];
                wo[nt * 8 + 2 * lane + 1] = ng[2 * nt + 1];
                wo[D + nt * 8 + 2 * lane] = nv[2 * nt + 0];
                wo[D + nt * 8 + 2 * lane + 1] = nv[2 * nt + 1];
            }
        }
        __syncthreads();
        for (int t = tid_h; t < NOUT * D; t += WPH * 32) {
            float acc = 0.0f;
            #pragma unroll
            for (int w = 0; w < WPH; w++) acc += wOut[((size_t)lh * WPH + w) * NOUT * D + t];
            redOut[lh * NOUT * D + t] += acc;
        }
    }
    __syncthreads();

    const int64_t out_off = q_off_h + (int64_t)a * D;
    for (int t = tid_h; t < D; t += WPH * 32) {
        gradXa[out_off + t] = scale * redOut[lh * NOUT * D + t];
        gradVa[out_off + t] = redOut[lh * NOUT * D + D + t];
    }
#endif
}


// One CTA runs separate R and S warps for each of G/2 heads. Only the raw Q/dY
// staging is shared between the two directions; everything else is per warp.
template<int G, int WPH, int BK, bool RAW=false>
__global__ __launch_bounds__(G * WPH * 32, 1)
void Bwd_rows_tc_dual(
    const bf16* __restrict__ Xa_bf,   // anchor stream (R or S), [B,1,N,D]
    const bf16* __restrict__ Va_bf,   // its values (Vr or Vs)
    const bf16* __restrict__ Xr_bf,   // Q [B,H,N,D]
    const bf16* __restrict__ gYr_bf,  // dY [B,H,N,D]
    const bf16* __restrict__ Xc_bf,   // other key stream (S or R), [B,1,N,D]
    const bf16* __restrict__ Vc_bf,   // its values
    const float* __restrict__ m_r, const float* __restrict__ l_r, const float* __restrict__ sum_r,   // [B,H,N]
    float* __restrict__ gradXa, float* __restrict__ gradVa, float* __restrict__ gradXc, float* __restrict__ gradVc,   // per-head partials [B,H,N,D]
    const bool* __restrict__ mask, int H, int N, float scale, int win)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int D = 128, DPAD = 136, BJ = WPH * 16;
    constexpr int KS = D / 16, DH = D, DV = D / 8, NOUT = 2;
    constexpr int NTHR = G * WPH * 32;

    const int a = blockIdx.x;
    constexpr int HG = G / 2;
    const int h0 = blockIdx.y * HG;
    const int b = blockIdx.z;
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int lh = warp / WPH;
    const int role = lh / HG, head = lh % HG;
    const int jw = (warp % WPH) * 16;
    const int tid_h = tid % (WPH * 32);
    const int g = lane / 4, tig = lane % 4;
    const int lrow = (lane & 7) + ((lane >> 3) & 1) * 8, lcol8 = (lane >> 4) * 8;
    const int brow = (lane & 7) + ((lane >> 4) & 1) * 8, bcol8 = ((lane >> 3) & 1) * 8;

    extern __shared__ char smem_raw[];
    bf16* a0_sm = reinterpret_cast<bf16*>(smem_raw);            // [G][BJ][DPAD] scale*Q_h o Xa
    bf16* a2_sm = a0_sm + (size_t)G * BJ * DPAD;                // [G][BJ][DPAD] dY_h o Va
    bf16* rawQ_sm = a2_sm + (size_t)G * BJ * DPAD;
    bf16* rawDY_sm = rawQ_sm + (RAW ? (size_t)HG * BJ * DPAD : 0);
    bf16* xc_sm = rawDY_sm + (RAW ? (size_t)HG * BJ * DPAD : 0);                // [role][2][BK][DPAD]
    bf16* vc_sm = xc_sm + 4 * BK * DPAD;                        // [role][2][BK][DPAD]
    float* anchX = reinterpret_cast<float*>(vc_sm + 4 * BK * DPAD);   // [role][D]
    float* anchV = anchX + 2 * D;                                   // [role][D]
    float* mr_sm = anchV + 2 * D;                                   // [G][BJ]
    float* ilr_sm = mr_sm + G * BJ;
    float* sr_sm = ilr_sm + G * BJ;
    float* wOut = sr_sm + G * BJ;                               // [G][WPH][NOUT*D]
    float* redOut = wOut + (size_t)G * WPH * NOUT * D;          // [G][NOUT*D]
    uint32_t* msk_sm = reinterpret_cast<uint32_t*>(redOut + (size_t)G * NOUT * D);   // [2][BJ]

    const int64_t kv_off = (int64_t)b * N * D;
    const int64_t q_off_h = ((int64_t)b * H + h0 + head) * N * D;
    const int64_t st_off_h = ((int64_t)b * H + h0 + head) * N;
    const bool* mb = mask + (int64_t)b * N * N;

    for (int d = tid; d < 2 * D; d += NTHR) {
        const int channel = d % D;
        anchX[d] = scale * bf2f((d < D ? Xa_bf : Xc_bf)[kv_off + (int64_t)a * D + channel]);
        anchV[d] = bf2f((d < D ? Va_bf : Vc_bf)[kv_off + (int64_t)a * D + channel]);
    }
    for (int t = tid; t < G * NOUT * D; t += NTHR) redOut[t] = 0.0f;

    int j_lo, j_hi;
    sg_row_bounds(SG_QUERY_ROWS, a, N, win, BJ, j_lo, j_hi);
    for (int j0 = j_lo; j0 < j_hi; j0 += BJ) {
        __syncthreads();
        int k_lo, k_hi;
        sg_col_bounds(SG_QUERY_ROWS, a, j0, BJ, N, win, BK, k_lo, k_hi);

        auto stage_cols = [&](int k0, int buf) {
            bf16* xs = xc_sm + buf * BK * DPAD;
            bf16* vs = vc_sm + buf * BK * DPAD;
            bf16* xs1 = xc_sm + (2 + buf) * BK * DPAD;
            bf16* vs1 = vc_sm + (2 + buf) * BK * DPAD;
            for (int idx = tid; idx < BK * DV; idx += NTHR) {
                const int kl = idx / DV, dv = (idx % DV) * 8;
                const int k = k0 + kl;
                if (k < N) {
                    const int64_t off = kv_off + (int64_t)k * D + dv;
                    cp_async16(xs + kl * DPAD + dv, Xc_bf + off);
                    cp_async16(vs + kl * DPAD + dv, Vc_bf + off);
                    cp_async16(xs1 + kl * DPAD + dv, Xa_bf + off);
                    cp_async16(vs1 + kl * DPAD + dv, Va_bf + off);
                } else {
                    const uint4 z = make_uint4(0, 0, 0, 0);
                    *reinterpret_cast<uint4*>(xs + kl * DPAD + dv) = z;
                    *reinterpret_cast<uint4*>(vs + kl * DPAD + dv) = z;
                    *reinterpret_cast<uint4*>(xs1 + kl * DPAD + dv) = z;
                    *reinterpret_cast<uint4*>(vs1 + kl * DPAD + dv) = z;
                }
            }
            for (int jl = tid; jl < BJ; jl += NTHR) {
                const int j = j0 + jl;
                msk_sm[buf * BJ + jl] = (j < N) ? sg_pack_mask32(mb + (int64_t)j * N + k0, min(BK, N - k0)) : 0u;
            }
        };

        if (role == 0) {
            for (int idx = tid_h; idx < BJ * DV; idx += WPH * 32) {
                const int jl = idx / DV, dv = (idx % DV) * 8;
                const int j = j0 + jl;
                uint4 xq = make_uint4(0, 0, 0, 0), gq = xq;
                if (j < N) {
                    const int64_t off = q_off_h + (int64_t)j * D + dv;
                    xq = *reinterpret_cast<const uint4*>(Xr_bf + off);
                    gq = *reinterpret_cast<const uint4*>(gYr_bf + off);
                }
                *reinterpret_cast<uint4*>(rawQ_sm + ((size_t)head * BJ + jl) * DPAD + dv) = xq;
                *reinterpret_cast<uint4*>(rawDY_sm + ((size_t)head * BJ + jl) * DPAD + dv) = gq;
            }
        }
        __syncthreads();

        // Per-head A operands from the head's own Q / dY rows and the shared anchor.
        {
            bf16* a0_h = a0_sm + (size_t)lh * BJ * DPAD;
            bf16* a2_h = a2_sm + (size_t)lh * BJ * DPAD;
            for (int idx = tid_h; idx < BJ * DV; idx += WPH * 32) {
                const int jl = idx / DV, dv = (idx % DV) * 8;
                const int j = j0 + jl;
                uint4 xq = make_uint4(0, 0, 0, 0), gq = xq;
                if (j < N) {
                    const int64_t off = q_off_h + (int64_t)j * D + dv;
                    xq = *reinterpret_cast<const uint4*>(rawQ_sm + ((size_t)head * BJ + jl) * DPAD + dv);
                    gq = *reinterpret_cast<const uint4*>(rawDY_sm + ((size_t)head * BJ + jl) * DPAD + dv);
                }
                const __nv_bfloat162* xp = reinterpret_cast<const __nv_bfloat162*>(&xq);
                const __nv_bfloat162* gp = reinterpret_cast<const __nv_bfloat162*>(&gq);
                uint4 o0, o2;
                __nv_bfloat162* p0 = reinterpret_cast<__nv_bfloat162*>(&o0);
                __nv_bfloat162* p2 = reinterpret_cast<__nv_bfloat162*>(&o2);
                #pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int d = dv + 2 * e;
                    const float2 x = __bfloat1622float2(xp[e]);
                    const float2 gy = __bfloat1622float2(gp[e]);
                    p0[e] = __floats2bfloat162_rn(x.x * anchX[role * D + d], x.y * anchX[role * D + d + 1]);
                    p2[e] = __floats2bfloat162_rn(gy.x * anchV[role * D + d], gy.y * anchV[role * D + d + 1]);
                }
                *reinterpret_cast<uint4*>(a0_h + jl * DPAD + dv) = o0;
                *reinterpret_cast<uint4*>(a2_h + jl * DPAD + dv) = o2;
            }
            for (int jl = tid_h; jl < BJ; jl += WPH * 32) {
                const int j = j0 + jl;
                if (j < N) {
                    mr_sm[lh * BJ + jl] = m_r[st_off_h + j];
                    float il = 1.0f / fmaxf(l_r[st_off_h + j], DENOM_EPS);
                    sr_sm[lh * BJ + jl] = sum_r[st_off_h + j];
                    if (!mb[(int64_t)j * N + a]) il = 0.0f;          // mask[query][anchor]
                    ilr_sm[lh * BJ + jl] = il;
                } else {
                    mr_sm[lh * BJ + jl] = 0.0f; ilr_sm[lh * BJ + jl] = 0.0f; sr_sm[lh * BJ + jl] = 0.0f;
                }
            }
        }
        stage_cols(k_lo, 0);
        asm volatile("cp.async.wait_all;\n" ::);
        __syncthreads();

        const bf16* a0_h = a0_sm + (size_t)lh * BJ * DPAD;
        const bf16* a2_h = a2_sm + (size_t)lh * BJ * DPAD;
        const float mr0 = mr_sm[lh * BJ + jw + g], sr0 = sr_sm[lh * BJ + jw + g];
        const float mr1 = mr_sm[lh * BJ + jw + g + 8], sr1 = sr_sm[lh * BJ + jw + g + 8];
        const float ilr0 = ilr_sm[lh * BJ + jw + g], ilr1 = ilr_sm[lh * BJ + jw + g + 8];
        const bool rpad0 = (j0 + jw + g) >= N, rpad1 = (j0 + jw + g + 8) >= N;

        float Ug[DH / 8][4], U1[DH / 8][4];
        #pragma unroll
        for (int nt = 0; nt < DH / 8; nt++) {
            #pragma unroll
            for (int e = 0; e < 4; e++) { Ug[nt][e] = 0.0f; U1[nt][e] = 0.0f; }
        }

        int cur = 0;
        for (int k0 = k_lo; k0 < k_hi; k0 += BK) {
            const int nxt = cur ^ 1;
            if (k0 + BK < k_hi) stage_cols(k0 + BK, nxt);
            const bf16* xc_cur = xc_sm + (role * 2 + cur) * BK * DPAD;
            const bf16* vc_cur = vc_sm + (role * 2 + cur) * BK * DPAD;

            #pragma unroll
            for (int s2 = 0; s2 < BK / 16; s2++) {
                float ax[2][4], adr[2][4];
                #pragma unroll
                for (int nt = 0; nt < 2; nt++) {
                    #pragma unroll
                    for (int e = 0; e < 4; e++) { ax[nt][e] = 0.0f; adr[nt][e] = 0.0f; }
                }
                const bf16* bx = xc_cur + (s2 * 16 + brow) * DPAD + bcol8;
                const bf16* bv = vc_cur + (s2 * 16 + brow) * DPAD + bcol8;
                #pragma unroll
                for (int ks = 0; ks < KS; ks++) {
                    uint32_t A0[4], A2[4], bfr[4];
                    const int roff = (jw + lrow) * DPAD + ks * 16 + lcol8;
                    ldmatrix_x4(A0, a0_h + roff);
                    ldmatrix_x4(A2, a2_h + roff);
                    ldmatrix_x4(bfr, bx + ks * 16);
                    mma_bf16_m16n8k16(ax[0], A0, bfr);
                    mma_bf16_m16n8k16(ax[1], A0, bfr + 2);
                    ldmatrix_x4(bfr, bv + ks * 16);
                    mma_bf16_m16n8k16(adr[0], A2, bfr);
                    mma_bf16_m16n8k16(adr[1], A2, bfr + 2);
                }

                uint32_t gAf[4], Prf[4];
                #pragma unroll
                for (int hf = 0; hf < 2; hf++) {
                    const int cl = s2 * 16 + hf * 8 + 2 * tig;
                    const uint32_t* mw = msk_sm + cur * BJ;
                    const uint32_t rc0 = mw[jw + g] >> cl, rc1 = mw[jw + g + 8] >> cl;
                    float gA[4], Pr[4];
                    #pragma unroll
                    for (int e = 0; e < 4; e++) {
                        const bool hi = (e >= 2), c1 = (e & 1);
                        const float mrr = hi ? mr1 : mr0, ilr = hi ? ilr1 : ilr0, srr = hi ? sr1 : sr0;
                        const float x = ax[hf][e];
                        const float ilrg = (((hi ? rc1 : rc0) >> c1) & 1u) ? ilr : 0.0f;
                        Pr[e] = __expf(fminf(x - mrr, 0.0f)) * ilrg;
                        gA[e] = (adr[hf][e] - srr) * Pr[e];
                    }
                    gAf[2 * hf + 0] = pack_bf162(gA[0], gA[1]);
                    gAf[2 * hf + 1] = pack_bf162(gA[2], gA[3]);
                    Prf[2 * hf + 0] = pack_bf162(Pr[0], Pr[1]);
                    Prf[2 * hf + 1] = pack_bf162(Pr[2], Pr[3]);
                }

                const bf16* px = xc_cur + (s2 * 16 + lrow) * DPAD + lcol8;
                const bf16* pv = vc_cur + (s2 * 16 + lrow) * DPAD + lcol8;
                #pragma unroll
                for (int np = 0; np < DH / 16; np++) {
                    uint32_t bfr[4];
                    ldmatrix_x4_trans(bfr, px + np * 16);
                    mma_bf16_m16n8k16(Ug[2 * np], gAf, bfr);
                    mma_bf16_m16n8k16(Ug[2 * np + 1], gAf, bfr + 2);
                    ldmatrix_x4_trans(bfr, pv + np * 16);
                    mma_bf16_m16n8k16(U1[2 * np], Prf, bfr);
                    mma_bf16_m16n8k16(U1[2 * np + 1], Prf, bfr + 2);
                }
            }
            asm volatile("cp.async.wait_all;\n" ::);
            __syncthreads();
            cur = nxt;
        }

        // ---- epilogue: Hadamard row-collapse of this warp's 16 rows (head's Q / dY rows) ----
        float ng[DH / 4], nv[DH / 4];
        const int64_t r0 = q_off_h + (int64_t)min(j0 + jw + g, N - 1) * D + 2 * tig;
        const int64_t r1 = q_off_h + (int64_t)min(j0 + jw + g + 8, N - 1) * D + 2 * tig;
        auto ld2 = [](const bf16* p, bool pad) {
            return pad ? make_float2(0.0f, 0.0f) : __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(p));
        };
        #pragma unroll
        for (int nt = 0; nt < DH / 8; nt++) {
            const float2 x0 = ld2(RAW ? rawQ_sm + ((size_t)head * BJ + jw + g) * DPAD + 2 * tig + nt * 8 : Xr_bf + r0 + nt * 8, rpad0), x1 = ld2(RAW ? rawQ_sm + ((size_t)head * BJ + jw + g + 8) * DPAD + 2 * tig + nt * 8 : Xr_bf + r1 + nt * 8, rpad1);
            ng[2 * nt + 0] = x0.x * Ug[nt][0] + x1.x * Ug[nt][2];
            ng[2 * nt + 1] = x0.y * Ug[nt][1] + x1.y * Ug[nt][3];
            const float2 g0 = ld2(RAW ? rawDY_sm + ((size_t)head * BJ + jw + g) * DPAD + 2 * tig + nt * 8 : gYr_bf + r0 + nt * 8, rpad0), g1 = ld2(RAW ? rawDY_sm + ((size_t)head * BJ + jw + g + 8) * DPAD + 2 * tig + nt * 8 : gYr_bf + r1 + nt * 8, rpad1);
            nv[2 * nt + 0] = g0.x * U1[nt][0] + g1.x * U1[nt][2];
            nv[2 * nt + 1] = g0.y * U1[nt][1] + g1.y * U1[nt][3];
        }
        #pragma unroll
        for (int off = 4; off <= 16; off <<= 1) {
            #pragma unroll
            for (int e = 0; e < DH / 4; e++) {
                ng[e] += __shfl_xor_sync(0xFFFFFFFF, ng[e], off);
                nv[e] += __shfl_xor_sync(0xFFFFFFFF, nv[e], off);
            }
        }
        if (lane < 4) {
            float* wo = wOut + ((size_t)lh * WPH + warp % WPH) * NOUT * D;
            #pragma unroll
            for (int nt = 0; nt < DH / 8; nt++) {
                wo[nt * 8 + 2 * lane] = ng[2 * nt + 0];
                wo[nt * 8 + 2 * lane + 1] = ng[2 * nt + 1];
                wo[D + nt * 8 + 2 * lane] = nv[2 * nt + 0];
                wo[D + nt * 8 + 2 * lane + 1] = nv[2 * nt + 1];
            }
        }
        __syncthreads();
        for (int t = tid_h; t < NOUT * D; t += WPH * 32) {
            float acc = 0.0f;
            #pragma unroll
            for (int w = 0; w < WPH; w++) acc += wOut[((size_t)lh * WPH + w) * NOUT * D + t];
            redOut[lh * NOUT * D + t] += acc;
        }
    }
    __syncthreads();

    const int64_t out_off = q_off_h + (int64_t)a * D;
    for (int t = tid_h; t < D; t += WPH * 32) {
        (role == 0 ? gradXa : gradXc)[out_off + t] = scale * redOut[lh * NOUT * D + t];
        (role == 0 ? gradVa : gradVc)[out_off + t] = redOut[lh * NOUT * D + D + t];
    }
#endif
}



template<int G, int BK> constexpr size_t dual_smem() {
 constexpr int BJ=16, D=128, DPAD=136;
 return sizeof(bf16)*((size_t)3*G*BJ*DPAD + (size_t)8*BK*DPAD)
      + sizeof(float)*((size_t)4*D + (size_t)G*3*BJ + (size_t)G*2*D + (size_t)G*2*D)
      + sizeof(uint32_t)*2*BJ;
}
template<int G>
bool launch_fwd_grouped(const bf16* Q, const bf16* R, const bf16* S, const bf16* Vr, const bf16* Vs,
                        bf16* Y, float* m, float* l, const bool* mask, int B, int H, int N, float scale,
                        int max_smem_optin, cudaStream_t stream, int win)
{
    const size_t smem = fwd_grouped_smem(G);
    if (smem > (size_t)max_smem_optin) return false;
    int attribute_current_device=0; AT_CUDA_CHECK(cudaGetDevice(&attribute_current_device));
    static thread_local int attr_device=-1;
    if (attr_device != attribute_current_device) {
        AT_CUDA_CHECK(cudaFuncSetAttribute(Y_gather_tc_grouped<G>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        attr_device = attribute_current_device;
    }
    Y_gather_tc_grouped<G><<<dim3(N, H / G, B), G * SG_WPH * 32, smem, stream>>>(Q, R, S, Vr, Vs, Y, m, l, mask, H, N, scale, win);
    return true;
}

template<int G>
bool launch_bwd_grouped(const bf16* Xa, const bf16* Va, const bf16* Xr, const bf16* gYr, const bf16* Xc, const bf16* Vc,
                        const float* m, const float* l, const float* sum, float* gX, float* gV, const bool* mask,
                        int B, int H, int N, float scale, int max_smem_optin, cudaStream_t stream, int win)
{
    constexpr int WPH = G == 4 ? 1 : 2, BK = G == 4 ? 16 : 32;
    const size_t smem = bwd_grouped_smem(G);
    if (smem > (size_t)max_smem_optin) return false;
    int attribute_current_device=0; AT_CUDA_CHECK(cudaGetDevice(&attribute_current_device));
    static thread_local int attr_device=-1;
    if (attr_device != attribute_current_device) {
        AT_CUDA_CHECK(cudaFuncSetAttribute(Bwd_rows_tc_grouped<G, WPH, BK, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        attr_device = attribute_current_device;
    }
    Bwd_rows_tc_grouped<G, WPH, BK, true><<<dim3(N, H / G, B), G * WPH * 32, smem, stream>>>(Xa, Va, Xr, gYr, Xc, Vc, m, l, sum, gX, gV, mask, H, N, scale, win);
    return true;
}

const bf16* bp(const at::Tensor& t) { return reinterpret_cast<const bf16*>(t.data_ptr<at::BFloat16>()); }

}  // namespace

bool launch_Y_gather_shared_grouped(
    const at::Tensor& Q, const at::Tensor& R, const at::Tensor& S, const at::Tensor& Vr,
    const at::Tensor& Vs, at::Tensor& Y, at::Tensor& m, at::Tensor& l, const bool* mask,
    int B, int H, int N, float scale, int max_smem_optin, cudaStream_t stream, int win, int G)
{
    auto* y = reinterpret_cast<bf16*>(Y.data_ptr<at::BFloat16>());
    if (G == 2) return launch_fwd_grouped<2>(bp(Q), bp(R), bp(S), bp(Vr), bp(Vs), y, m.data_ptr<float>(), l.data_ptr<float>(),
                                             mask, B, H, N, scale, max_smem_optin, stream, win);
    if (G == 4) return launch_fwd_grouped<4>(bp(Q), bp(R), bp(S), bp(Vr), bp(Vs), y, m.data_ptr<float>(), l.data_ptr<float>(),
                                             mask, B, H, N, scale, max_smem_optin, stream, win);
    return false;
}

bool launch_bwd_rows_shared_grouped(
    const at::Tensor& Q, const at::Tensor& dY, const at::Tensor& R, const at::Tensor& Vr,
    const at::Tensor& S, const at::Tensor& Vs, const at::Tensor& m, const at::Tensor& l,
    const at::Tensor& delta, at::Tensor& pR, at::Tensor& pVr, at::Tensor& pS, at::Tensor& pVs,
    const bool* mask, int B, int H, int N, float scale, int max_smem_optin, cudaStream_t stream,
    int win, int G)
{
    auto fp = [](const at::Tensor& t) { return t.data_ptr<float>(); };
    if (G == 4) {
        // G=4 runs the dual kernel: 2 heads x 2 directions per CTA. The grid
        // only needs H % 2, but the shared check still requires H % 4.
        constexpr int HEADS = 2, WARPS = 4, BK = 16;
        constexpr size_t smem = dual_smem<WARPS, BK>();
        if (smem > (size_t)max_smem_optin) return false;
        int attribute_current_device=0; AT_CUDA_CHECK(cudaGetDevice(&attribute_current_device));
    static thread_local int attr_device=-1;
        if (attr_device != attribute_current_device) {
            AT_CUDA_CHECK(cudaFuncSetAttribute(Bwd_rows_tc_dual<WARPS, 1, BK, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
            attr_device = attribute_current_device;
        }
        Bwd_rows_tc_dual<WARPS, 1, BK, true><<<dim3(N, H / HEADS, B), WARPS * 32, smem, stream>>>(
            bp(R), bp(Vr), bp(Q), bp(dY), bp(S), bp(Vs), fp(m), fp(l), fp(delta),
            fp(pR), fp(pVr), fp(pS), fp(pVs), mask, H, N, scale, win);
        return true;
    }
    bool ok = false;
    if (G == 2) {
        ok = launch_bwd_grouped<2>(bp(R), bp(Vr), bp(Q), bp(dY), bp(S), bp(Vs), fp(m), fp(l), fp(delta), fp(pR), fp(pVr), mask, B, H, N, scale, max_smem_optin, stream, win)
          && launch_bwd_grouped<2>(bp(S), bp(Vs), bp(Q), bp(dY), bp(R), bp(Vr), fp(m), fp(l), fp(delta), fp(pS), fp(pVs), mask, B, H, N, scale, max_smem_optin, stream, win);
    } else if (G == 4) {
        ok = launch_bwd_grouped<4>(bp(R), bp(Vr), bp(Q), bp(dY), bp(S), bp(Vs), fp(m), fp(l), fp(delta), fp(pR), fp(pVr), mask, B, H, N, scale, max_smem_optin, stream, win)
          && launch_bwd_grouped<4>(bp(S), bp(Vs), bp(Q), bp(dY), bp(R), bp(Vr), fp(m), fp(l), fp(delta), fp(pS), fp(pVs), mask, B, H, N, scale, max_smem_optin, stream, win);
    }
    return ok;
}

bool launch_Y_gather_shared_auto(
    const at::Tensor& Q,const at::Tensor& R,const at::Tensor& S,const at::Tensor& Vr,
    const at::Tensor& Vs,at::Tensor& Y,at::Tensor& m,at::Tensor& l,const bool* mask,
    int B,int H,int N,float scale,int optin,cudaStream_t stream,int win,const char*& implementation) {
    auto bp=[](const at::Tensor& t){return reinterpret_cast<const bf16*>(t.data_ptr<at::BFloat16>());};
    auto y=reinterpret_cast<bf16*>(Y.data_ptr<at::BFloat16>());
    if (launch_att3_shared_hopper(bp(Q),bp(R),bp(S),bp(Vr),bp(Vs),y,
        m.data_ptr<float>(),l.data_ptr<float>(),mask,B,H,N,scale,optin,stream,win)) {
        implementation="hopper-retained"; return true;
    }
    if (att3_shared_fwd::launch(bp(Q),bp(R),bp(S),bp(Vr),bp(Vs),y,
        m.data_ptr<float>(),l.data_ptr<float>(),mask,B,H,N,scale,optin,stream,win)) {
        implementation="mma-retained"; return true;
    }
    return false;
}
