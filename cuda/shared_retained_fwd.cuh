#pragma once
// F04 one-warp geometry at w16; F10 private probability-fragment retention,
// short-lived value accumulators, raw windows retained across four queries,
// and dead transformed-row storage reused by the epilogue; F07 exact row-tile
// merging for grouped heads at w64/128. Independently derived from owned code.
// No query/head shares transformed features, probabilities or derivatives.
// Fixed-baseline timings and failed ablations are summarized in
// docs/KERNEL_HISTORY.md.
#include <ATen/cuda/CUDAContext.h>
#include "common.cuh"
namespace att3_shared_fwd {
namespace w16 {
constexpr int SG_D = 128;
constexpr int SG_DPAD = SG_D + 8;
constexpr int SG_WPH = 1;                 // warps per head
constexpr int SG_BJ = SG_WPH * 16;        // rows per head tile
constexpr int SG_BK = 16;                 // cols per tile
constexpr float SG_MASKED_THRESH = -5e29f;

// Measured on H100 (paired before/after, N128/256/512): a single BK=32 tile
// (matching w32 in one shot) is geometrically single-tile but *slower* than
// two double-buffered BK=16 tiles -- the cp.async prefetch of tile 2 overlaps
// tile 1's tensor-core compute, and that overlap outweighs visiting fewer,
// wider tiles. So BK stays 16 and double-buffered here; only the row-tile
// fold below (win == 32 == BJ makes it the only row tile) is simplified.
constexpr size_t fwd_grouped_smem(int G) {
    // rowp[G] + raw rows + V rows + 2 x (cols, V cols) in bf16; col_mul, row_mul,
    // anchor[G], wN[G][WPH], wML[G][WPH], redN[G], redML[G] in fp32.
    return sizeof(bf16) * ((size_t)(G + 2) * SG_BJ * SG_DPAD + (size_t)4 * SG_BK * SG_DPAD)
         + sizeof(float) * ((size_t)2 * SG_BK + SG_BJ + (size_t)G * (SG_D + SG_WPH * SG_D + SG_WPH * 2 + SG_D + 2));
}

// =============================================================================
// Grouped forward: block = (query i, G heads, batch b)
// =============================================================================
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
    // redN/redML need no prior-state init: the fold below always runs as the
    // first (and only) row tile, so it writes them outright rather than
    // merging into an old value.

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

        // Head-scaled row tile, private per head: the same fp32 product and single
        // bf16 rounding as the per-head kernel's staging.
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

        // ---- combine the head's warps into (M, L, N): win == 32 == BJ makes
        // this the only row tile (single_gather_shared_check enforces window
        // == 32), so there is no prior running state to fold in -- Mold is
        // always -inf and Lold always 0, making their rescale dead weight.
        {
            const float* wML_h = wML + lh * WPH * 2;
            const float* wN_hh = wN + lh * WPH * D;
            float* redN_h = redN + lh * D;
            float* redML_h = redML + lh * 2;
            float Mnew = NEG_INF;
            #pragma unroll
            for (int wd = 0; wd < WPH; wd++) Mnew = fmaxf(Mnew, wML_h[wd * 2]);
            for (int c=tid_h; c<D; c+=WPH*32) {
                float v=0.0f;
                #pragma unroll
                for(int wd=0;wd<WPH;++wd)v+=__expf(wML_h[wd*2]-Mnew)*wN_hh[wd*D+c];
                redN_h[c]=v;
            }
            float lNew = 0.0f;
            #pragma unroll
            for (int wd = 0; wd < WPH; wd++) lNew += __expf(wML_h[wd * 2] - Mnew) * wML_h[wd * 2 + 1];
            __syncthreads();
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

}
namespace w32 {
constexpr int SG_D = 128;
constexpr int SG_DPAD = SG_D + 8;
constexpr int SG_WPH = 2;                 // warps per head
constexpr int SG_BJ = SG_WPH * 16;        // rows per head tile
constexpr int SG_BK = 16;                 // cols per tile
constexpr float SG_MASKED_THRESH = -5e29f;

// Measured on H100 (paired before/after, N128/256/512): a single BK=32 tile
// (matching w32 in one shot) is geometrically single-tile but *slower* than
// two double-buffered BK=16 tiles -- the cp.async prefetch of tile 2 overlaps
// tile 1's tensor-core compute, and that overlap outweighs visiting fewer,
// wider tiles. So BK stays 16 and double-buffered here; only the row-tile
// fold below (win == 32 == BJ makes it the only row tile) is simplified.
constexpr size_t fwd_grouped_smem(int G) {
    // rowp[G] + raw rows + V rows + 2 x (cols, V cols) in bf16; col_mul, row_mul,
    // anchor[G], wN[G][WPH], wML[G][WPH], redN[G], redML[G] in fp32.
    return sizeof(bf16) * ((size_t)(G + 2) * SG_BJ * SG_DPAD + (size_t)4 * SG_BK * SG_DPAD)
         + sizeof(float) * ((size_t)2 * SG_BK + SG_BJ + (size_t)G * (SG_D + SG_WPH * SG_D + SG_WPH * 2 + SG_D + 2));
}

// =============================================================================
// Grouped forward: block = (query i, G heads, batch b)
// =============================================================================
template<int G, int QCOUNT, bool REUSE>
__global__ __launch_bounds__(G * SG_WPH * 32)
void Y_gather_tc_lifetime(
    const bf16* __restrict__ Q, const bf16* __restrict__ R, const bf16* __restrict__ S,
    const bf16* __restrict__ Vr, const bf16* __restrict__ Vs,
    bf16* __restrict__ Y, float* __restrict__ m_out, float* __restrict__ l_out,
    const bool* __restrict__ mask, int H, int N, float scale, int win)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int D = SG_D, DPAD = SG_DPAD, BJ = SG_BJ, BK = SG_BK, WPH = SG_WPH;
    constexpr int KS = D / 16, NT = D / 8, CT = BK / 8, DV = D / 8;
    constexpr int NTHR = G * WPH * 32;

    const int i0 = blockIdx.x * QCOUNT;
    constexpr int CACHE = SG_BJ + QCOUNT - 1;
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
    bf16* rawR = rowp_sm + (size_t)G * BJ * DPAD;
    bf16* rawVr = rawR + CACHE * DPAD;
    bf16* rawS = rawVr + CACHE * DPAD;
    bf16* rawVs = rawS + CACHE * DPAD;
    float* col_mul = reinterpret_cast<float*>(rawVs + CACHE * DPAD);
    float* row_mul = col_mul + 2 * BK;                                  // [BJ]
    float* anchor_sm = row_mul + BJ;                                    // [G][D]
    float* wN  = reinterpret_cast<float*>(rowp_sm);                                     // [G][WPH][D]
    float* wML = wN + G * WPH * D;                                      // [G][WPH][2]
    float* redN = wN;                                    // [G][D]
    float* redML = wML;                                        // [G][2]

    const int64_t kv_off = (int64_t)b * N * D;                          // Hkv = 1
    const int union_lo = max(0, i0 - win + 1);
    auto stage_raw = [&](int cache_lo) {
        for (int idx = tid; idx < CACHE * DV; idx += NTHR) {
            const int p = idx / DV, dv = (idx % DV) * 8, token = cache_lo + p;
            if (token < N) {
                const int64_t off = kv_off + (int64_t)token * D + dv;
                cp_async16(rawR + p * DPAD + dv, R + off);
                cp_async16(rawVr + p * DPAD + dv, Vr + off);
                cp_async16(rawS + p * DPAD + dv, S + off);
                cp_async16(rawVs + p * DPAD + dv, Vs + off);
            } else {
                const uint4 z = make_uint4(0,0,0,0);
                *reinterpret_cast<uint4*>(rawR + p * DPAD + dv) = z;
                *reinterpret_cast<uint4*>(rawVr + p * DPAD + dv) = z;
                *reinterpret_cast<uint4*>(rawS + p * DPAD + dv) = z;
                *reinterpret_cast<uint4*>(rawVs + p * DPAD + dv) = z;
            }
        }
        asm volatile("cp.async.wait_all;\n" ::);
        __syncthreads();
    };
    if constexpr (REUSE) stage_raw(union_lo);
    for (int qi = 0; qi < QCOUNT && i0 + qi < N; ++qi) {
    const int i = i0 + qi;
    const int cache_lo = REUSE ? union_lo : max(0, i - win + 1);
    if constexpr (!REUSE) stage_raw(cache_lo);
    const bool* mrow = mask + ((int64_t)b * N + i) * N;

    for (int t = tid; t < G * D; t += NTHR) {
        const int hh = t / D, d = t % D;
        anchor_sm[t] = scale * bf2f(Q[(((int64_t)b * H + h0 + hh) * N + i) * D + d]);
    }
    // redN/redML need no prior-state init: the fold below always runs as the
    // first (and only) row tile, so it writes them outright rather than
    // merging into an old value.

    auto stage_cols = [&](int k0, int buf) {
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

        const bf16* rows_sm = rawR + (j0 - cache_lo) * DPAD;
        const bf16* v_rows_sm = rawVr + (j0 - cache_lo) * DPAD;
        for (int jl = tid; jl < BJ; jl += NTHR) {
            const int j = j0 + jl;
            row_mul[jl] = (j < N && mrow[j]) ? 1.0f : 0.0f;
        }
        stage_cols(k_lo, 0);
        asm volatile("cp.async.wait_all;\n" ::);
        __syncthreads();

        // Head-scaled row tile, private per head: the same fp32 product and single
        // bf16 rounding as the per-head kernel's staging.
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
        // Raw R has no further readers. Reuse its storage for per-query
        // probability fragments and the original online rescaling factors.
        uint32_t saved_p[2][4];
        float saved_alpha[2][2];
        int cur = 0;
        #pragma unroll
        for (int ti=0; ti<2; ++ti) {
            const int k0=k_lo+ti*BK;
            if(k0>=k_hi)break;
            const int nxt = cur ^ 1;
            if (k0 + BK < k_hi) stage_cols(k0 + BK, nxt);
            const bf16* cols_cur = rawS + (k0 - cache_lo) * DPAD;
            const bf16* v_cols_cur = rawVs + (k0 - cache_lo) * DPAD;

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
            const int tile = (k0 - k_lo) / BK;
            saved_alpha[tile][0] = a0;
            saved_alpha[tile][1] = a1;
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
            for (int z=0; z<4; ++z)
                saved_p[tile][z] = pfr[0][z];
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

        float* wN_h = wN + (lh * WPH + warp % WPH) * D;
        // One output microtile remains live at a time. The exact sequence
        // first-PV -> original alpha -> second-PV is unchanged for each value.
        #pragma unroll
        for (int np=0; np<NT/2; ++np) {
            float u[2][4] = {};
            #pragma unroll
            for (int tile=0; tile<2; ++tile) {
                if(tile>=(k_hi-k_lo)/BK)break;
                const float a0 = saved_alpha[tile][0];
                const float a1 = saved_alpha[tile][1];
                #pragma unroll
                for(int z=0;z<2;++z){u[z][0]*=a0;u[z][1]*=a0;u[z][2]*=a1;u[z][3]*=a1;}
                uint32_t pfr[4];
                #pragma unroll
                for(int z=0;z<4;++z)pfr[z]=saved_p[tile][z];
                const bf16* bp = rawVs + (k_lo - cache_lo + tile * BK) * DPAD + lrow * DPAD + lcol8 + np*16;
                uint32_t bfr[4];ldmatrix_x4_trans(bfr,bp);
                mma_bf16_m16n8k16(u[0],pfr,bfr);
                mma_bf16_m16n8k16(u[1],pfr,bfr+2);
            }
            float nacc[4];
            #pragma unroll
            for(int z=0;z<2;++z){
                const int nt=np*2+z;
                const float2 v0=__bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(v_rows_sm+(rw+g)*DPAD+2*tig+nt*8));
                const float2 v1=__bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(v_rows_sm+(rw+g+8)*DPAD+2*tig+nt*8));
                nacc[z*2]=w0*v0.x*u[z][0]+w1*v1.x*u[z][2];
                nacc[z*2+1]=w0*v0.y*u[z][1]+w1*v1.y*u[z][3];
            }
            #pragma unroll
            for(int off=4;off<=16;off<<=1){
                #pragma unroll
                for(int z=0;z<4;++z)nacc[z]+=__shfl_xor_sync(0xffffffff,nacc[z],off);
            }
            if(lane<4){
                #pragma unroll
                for(int z=0;z<2;++z){wN_h[np*16+z*8+2*lane]=nacc[2*z];wN_h[np*16+z*8+2*lane+1]=nacc[2*z+1];}
            }
        }
        if (lane == 0) { wML[(lh * WPH + warp % WPH) * 2] = Mw; wML[(lh * WPH + warp % WPH) * 2 + 1] = Lw; }
        __syncthreads();

        // ---- combine the head's warps into (M, L, N): win == 32 == BJ makes
        // this the only row tile (single_gather_shared_check enforces window
        // == 32), so there is no prior running state to fold in -- Mold is
        // always -inf and Lold always 0, making their rescale dead weight.
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
    __syncthreads();
    } // sequential queries; the accumulator set is reused
#endif
}


template<int G, int QCOUNT> constexpr size_t retained_bytes() {
 return sizeof(bf16) * ((size_t)G*SG_BJ*SG_DPAD + (size_t)4*(SG_BJ+QCOUNT-1)*SG_DPAD)
      + sizeof(float) * ((size_t)2*SG_BK+SG_BJ + (size_t)G*(SG_D));
}

}
namespace wider {
constexpr int SG_D = 128;
constexpr int SG_DPAD = SG_D + 8;
constexpr int SG_WPH = 2;                 // warps per head
constexpr int SG_BJ = SG_WPH * 16;        // rows per head tile
constexpr int SG_BK = 16;                 // cols per tile
constexpr float SG_MASKED_THRESH = -5e29f;

// Measured on H100 (paired before/after, N128/256/512): a single BK=32 tile
// (matching w32 in one shot) is geometrically single-tile but *slower* than
// two double-buffered BK=16 tiles -- the cp.async prefetch of tile 2 overlaps
// tile 1's tensor-core compute, and that overlap outweighs visiting fewer,
// wider tiles. So BK stays 16 and double-buffered here; only the row-tile
// fold below (win == 32 == BJ makes it the only row tile) is simplified.
constexpr size_t fwd_grouped_smem(int G) {
    // rowp[G] + raw rows + V rows + 2 x (cols, V cols) in bf16; col_mul, row_mul,
    // anchor[G], wN[G][WPH], wML[G][WPH], redN[G], redML[G] in fp32.
    return sizeof(bf16) * ((size_t)(G + 2) * SG_BJ * SG_DPAD + (size_t)4 * SG_BK * SG_DPAD)
         + sizeof(float) * ((size_t)2 * SG_BK + SG_BJ + (size_t)G * (SG_D + SG_WPH * SG_D + SG_WPH * 2 + SG_D + 2));
}

// =============================================================================
// Grouped forward: block = (query i, G heads, batch b)
// =============================================================================
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
    // redN/redML need no prior-state init: the fold below always runs as the
    // first (and only) row tile, so it writes them outright rather than
    // merging into an old value.

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

    for(int d=tid_h;d<D;d+=WPH*32)redN[lh*D+d]=0.0f;
    if(tid_h==0){redML[lh*2]=NEG_INF;redML[lh*2+1]=0.0f;}
    __syncthreads();
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

        // Head-scaled row tile, private per head: the same fp32 product and single
        // bf16 rounding as the per-head kernel's staging.
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

        // ---- combine the head's warps into (M, L, N): win == 32 == BJ makes
        // this the only row tile (single_gather_shared_check enforces window
        // == 32), so there is no prior running state to fold in -- Mold is
        // always -inf and Lold always 0, making their rescale dead weight.
        {
            const float* wML_h = wML + lh * WPH * 2;
            const float* wN_hh = wN + lh * WPH * D;
            float* redN_h = redN + lh * D;
            float* redML_h = redML + lh * 2;
            const float Mold = redML_h[0], Lold = redML_h[1];
            float Mnew = Mold;
            #pragma unroll
            for (int wd = 0; wd < WPH; wd++) Mnew = fmaxf(Mnew, wML_h[wd * 2]);
            for (int c=tid_h; c<D; c+=WPH*32) {
                float v=__expf(Mold-Mnew)*redN_h[c];
                #pragma unroll
                for(int wd=0;wd<WPH;++wd)v+=__expf(wML_h[wd*2]-Mnew)*wN_hh[wd*D+c];
                redN_h[c]=v;
            }
            float lNew = __expf(Mold-Mnew)*Lold;
            #pragma unroll
            for (int wd = 0; wd < WPH; wd++) lNew += __expf(wML_h[wd * 2] - Mnew) * wML_h[wd * 2 + 1];
            __syncthreads();
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

}
inline bool launch(const bf16* Q,const bf16* R,const bf16* S,const bf16* Vr,const bf16* Vs,
 bf16* Y,float* m,float* l,const bool* mask,int B,int H,int N,float scale,
 int max_smem_optin,cudaStream_t stream,int win) {
 if(H%2 || N%16)return false;
 if(win==16){
  constexpr size_t smem=w16::fwd_grouped_smem(2);
  if(smem>(size_t)max_smem_optin)return false;
  int device=0; AT_CUDA_CHECK(cudaGetDevice(&device));
  static thread_local int attribute_device=-1;
  if(attribute_device!=device){AT_CUDA_CHECK(cudaFuncSetAttribute(w16::Y_gather_tc_grouped<2>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));attribute_device=device;}
  w16::Y_gather_tc_grouped<2><<<dim3(N,H/2,B),64,smem,stream>>>(Q,R,S,Vr,Vs,Y,m,l,mask,H,N,scale,win);
  return true;
 }
 if(win==32 && ((H>=64 && N>=128)||(H>=16 && N>=256))){
  constexpr size_t smem=w32::retained_bytes<2,4>();
  if(smem>(size_t)max_smem_optin)return false;
  int device=0; AT_CUDA_CHECK(cudaGetDevice(&device));
  static thread_local int attribute_device=-1;
  if(attribute_device!=device){AT_CUDA_CHECK(cudaFuncSetAttribute(w32::Y_gather_tc_lifetime<2,4,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));attribute_device=device;}
  w32::Y_gather_tc_lifetime<2,4,true><<<dim3(ceil_div(N,4),H/2,B),128,smem,stream>>>(Q,R,S,Vr,Vs,Y,m,l,mask,H,N,scale,win);
  return true;
 }
 if(win==64 || win==128){
  constexpr size_t smem=wider::fwd_grouped_smem(2);
  if(smem>(size_t)max_smem_optin)return false;
  int device=0; AT_CUDA_CHECK(cudaGetDevice(&device));
  static thread_local int attribute_device=-1;
  if(attribute_device!=device){AT_CUDA_CHECK(cudaFuncSetAttribute(wider::Y_gather_tc_grouped<2>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));attribute_device=device;}
  wider::Y_gather_tc_grouped<2><<<dim3(N,H/2,B),128,smem,stream>>>(Q,R,S,Vr,Vs,Y,m,l,mask,H,N,scale,win);
  return true;
 }
 return false;
}
} // namespace att3_shared_fwd
