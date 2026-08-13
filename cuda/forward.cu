/**
 * @file forward.cu
 * @brief Forward pass CUDA kernels for hypergraph attention.
 *
 * Implements gather and scatter attention patterns using online softmax
 * (FlashAttention-style) for memory efficiency. Supports FP32 and BF16.
 *
 * Copyright (c) 2026 Springtail AI. MIT License.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "common.cuh"
#include "../cpp/cuda_bindings.h"

// =============================================================================
// Shared constants and helpers
// =============================================================================

constexpr int N_I_GATHER = 4;       // anchor vectors per block in the split gather
constexpr int GATHER_SMEM_PAD = 4;  // bf16 tile-row padding against bank conflicts
constexpr int SCAT_SMEM_PAD = 4;
constexpr int SCAT_ATTN_PAD = 2;
constexpr int MIN_SPLIT_CHUNKS = 4;  // Only split loops across blocks when >= this many chunks
constexpr int MAX_SPLIT_CHUNKS = 16; // Cap split workspace growth and reducer scratch.

__device__ __forceinline__ bool mask_pair_allowed(
    const bool* mask,
    int N,
    int b,
    int anchor,
    int a,
    int c
) {
    if (mask == nullptr) {
        return true;
    }
    const int64_t base = ((int64_t)b * N + anchor) * N;
    return mask[base + a] && mask[base + c];
}

// =============================================================================
// Tensor-core gather (Yq/Yr/Ys, D=64, resident dim <= TC_MAX_K)
// =============================================================================
// FlashAttention-style reformulation (cuda_docs/gather_readme.md), written in Yq terms
// below; the trilinear score is symmetric in Q/R/S, so Yr and Ys launch the
// same kernel with permuted roles (anchor=R rows=Q resident=S, and
// anchor=S rows=Q resident=R). One CTA owns one (b, h, i). Absorbing
// scale*Q[i,:] as a diagonal rescale of R turns both heavy stages into
// tensor-core GEMMs:
//
//   Qp[j,d] = scale * Q[i,d] * R[j,d]
//   x[j,k]  = (Qp @ S^T)[j,k]                        GEMM 1 (mma.m16n8k16)
//   U[j,d]  = sum_k exp(x[j,k] - m_j) * V2[k,d]      GEMM 2, per-row online softmax
//   Yq[i,d] = sum_j exp(m_j - M) * V1[j,d] * U[j,d] / L,   L = sum_j exp(m_j-M) l_j
//
// The joint (j,k) softmax is recovered exactly by the log-sum-exp reweighting of
// rows in the epilogue, so split-J partials and the reducer pass disappear: the
// kernel writes Y_q (bf16) and the m_i / l_i stats directly, with semantics
// identical to reduce_gather_partials (backward consumes them unchanged).
//
// S and V2 stay resident in shared memory for the whole CTA (K <= TC_MAX_K
// specialization; V2 stays row-major, GEMM-2 B fragments use ldmatrix.trans).
// j is processed in blocks of TC_BJ rows, one m16 row-tile per warp. GEMM 1
// accumulator fragments feed GEMM 2 A fragments directly in registers (the
// C-frag/A-frag layouts coincide), so P never round-trips through shared
// memory.

constexpr int TC_BJ = 128;        // j rows per block iteration (8 warps x 16)
constexpr int TC_BK = 64;         // k columns per inner iteration
constexpr int TC_WARPS = 8;
constexpr int TC_MAX_K = 256;     // S/V2 must fit in shared memory
// A GEMM-1 accumulator this negative marks a masked cell (valid |scores| are
// bounded far below this; NEG_INF itself is -1e30).
constexpr float TC_MASKED_THRESH = -5e29f;

// mma_bf16_m16n8k16 / pack_bf162 / ldmatrix_x4 / ldmatrix_x4_trans /
// cp_async16 live in common.cuh (shared with the tensor-core backward).

// MASKED=false is the fast path for the common case (no attention mask, no
// J/K padding): score masking, kmul/jmul reads, and the masked-exp selects
// drop out of the hot loop entirely.
template<int D_CONST, bool MASKED>
__global__ __launch_bounds__(TC_WARPS * 32)
void Y_gather_tc(
    const bf16* __restrict__ Q_bf,
    const bf16* __restrict__ R_bf,
    const bf16* __restrict__ S_bf,
    const bf16* __restrict__ V1_bf,
    const bf16* __restrict__ V2_bf,
    bf16*  __restrict__ Y,          // [B,H,I,D] final output (no reducer pass)
    float* __restrict__ m_i_out,    // [B,H,I]
    float* __restrict__ l_i_out,    // [B,H,I]
    const bool* __restrict__ mask,  // [B,N,N] or null
    int H, int I, int J, int K, int K_pad, float scale,
    int J_valid, int K_valid)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    static_assert(D_CONST == 64, "Y_gather_tc supports D=64 only");
    constexpr int D = D_CONST;
    constexpr int DPAD = D + 8;     // bf16 row stride: 144 B, conflict-free for frags

    const int i = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;

    const int tid  = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int g    = lane / 4;      // mma group id (row within a fragment)
    const int tig  = lane % 4;      // thread-in-group (column pairs)
    // ldmatrix.x4 source rows: lane bit 3 selects the row-half of the 16-wide
    // tile, bit 4 the column-half (A and trans-B); GEMM-1 B swaps the two.
    const int lrow  = (lane & 7) + ((lane >> 3) & 1) * 8;
    const int lcol8 = (lane >> 4) * 8;
    const int brow  = (lane & 7) + ((lane >> 4) & 1) * 8;
    const int bcol8 = ((lane >> 3) & 1) * 8;

    extern __shared__ char smem_raw[];
    bf16* qp_sm  = reinterpret_cast<bf16*>(smem_raw);           // [TC_BJ][DPAD]
    bf16* v1_sm  = qp_sm + TC_BJ * DPAD;                        // [TC_BJ][DPAD]
    bf16* s_sm   = v1_sm + TC_BJ * DPAD;                        // [K_pad][DPAD]
    bf16* v2_sm  = s_sm + K_pad * DPAD;                         // [K_pad][DPAD]
    float* kmul  = reinterpret_cast<float*>(v2_sm + K_pad * DPAD);  // [K_pad]
    float* jmul  = kmul + K_pad;                                // [TC_BJ]
    float* wN    = jmul + TC_BJ;                                // [TC_WARPS][D]
    float* wML   = wN + TC_WARPS * D;                           // [TC_WARPS][2]
    float* redN  = wML + TC_WARPS * 2;                          // [D]
    float* redML = redN + D;                                    // {M_run, L_run}
    float* q_sm  = redML + 2;                                   // [D] fp32 scale*q

    const int64_t bh = (int64_t)b * H + h;
    const bool* mrow = (mask != nullptr) ? (mask + ((int64_t)b * I + i) * I) : nullptr;

    // ---- one-time loads: resident S and V2 (async), scale*q, k-validity ----
    constexpr int DV = D / 8;       // 16-byte vectors per row
    const int64_t sv_off = bh * K * D;
    for (int idx = tid; idx < K_pad * DV; idx += blockDim.x) {
        const int k = idx / DV, dv = (idx % DV) * 8;
        if (k < K) {
            cp_async16(s_sm + k * DPAD + dv, S_bf + sv_off + (int64_t)k * D + dv);
            cp_async16(v2_sm + k * DPAD + dv, V2_bf + sv_off + (int64_t)k * D + dv);
        } else {
            const uint4 z = make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(s_sm + k * DPAD + dv) = z;
            *reinterpret_cast<uint4*>(v2_sm + k * DPAD + dv) = z;
        }
    }
    const int64_t q_off = (bh * I + i) * D;
    for (int d = tid; d < D; d += blockDim.x) {
        q_sm[d] = scale * bf2f(Q_bf[q_off + d]);
    }
    if constexpr (MASKED) {
        for (int k = tid; k < K_pad; k += blockDim.x) {
            kmul[k] = (k < K_valid && (mrow == nullptr || mrow[k])) ? 1.0f : 0.0f;
        }
    }
    if (tid == 0) { redML[0] = NEG_INF; redML[1] = 0.0f; }
    for (int d = tid; d < D; d += blockDim.x) redN[d] = 0.0f;

    // ---- j blocks of TC_BJ rows, one 16-row tile per warp ----
    for (int j0 = 0; j0 < J; j0 += TC_BJ) {
        __syncthreads();  // previous iteration's smem reads (and initial loads) done

        for (int idx = tid; idx < TC_BJ * DV; idx += blockDim.x) {
            const int jl = idx / DV, dv = (idx % DV) * 8;
            const int j = j0 + jl;
            uint4 rq = make_uint4(0, 0, 0, 0), vq = rq;
            if (j < J) {
                const int64_t off = (bh * J + j) * D + dv;
                rq = *reinterpret_cast<const uint4*>(R_bf + off);
                vq = *reinterpret_cast<const uint4*>(V1_bf + off);
            }
            __nv_bfloat162* rp = reinterpret_cast<__nv_bfloat162*>(&rq);
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                const float2 rf = __bfloat1622float2(rp[e]);
                rp[e] = __floats2bfloat162_rn(q_sm[dv + 2 * e] * rf.x,
                                              q_sm[dv + 2 * e + 1] * rf.y);
            }
            *reinterpret_cast<uint4*>(qp_sm + jl * DPAD + dv) = rq;
            *reinterpret_cast<uint4*>(v1_sm + jl * DPAD + dv) = vq;
        }
        if constexpr (MASKED) {
            for (int jl = tid; jl < TC_BJ; jl += blockDim.x) {
                int j = j0 + jl;
                jmul[jl] = (j < J_valid && (mrow == nullptr || mrow[j])) ? 1.0f : 0.0f;
            }
        }
        asm volatile("cp.async.wait_all;\n" ::);
        __syncthreads();

        const int jw = warp * 16;   // warp's row-tile base within the block
        const float jm0 = MASKED ? jmul[jw + g] : 1.0f;
        const float jm1 = MASKED ? jmul[jw + g + 8] : 1.0f;

        // GEMM-1 A fragments (Qp rows) are k-invariant: hoist all 4 k-steps.
        uint32_t aQ[4][4];
        #pragma unroll
        for (int ks = 0; ks < 4; ks++) {
            ldmatrix_x4(aQ[ks], qp_sm + (jw + lrow) * DPAD + ks * 16 + lcol8);
        }

        float m0 = NEG_INF, m1 = NEG_INF;   // rows g / g+8 (replicated in quad)
        float l0 = 0.0f, l1 = 0.0f;         // per-lane partials, quad-reduced later
        float U[8][4];                      // GEMM-2 accumulators over D
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            U[nt][0] = U[nt][1] = U[nt][2] = U[nt][3] = 0.0f;
        }

        for (int k0 = 0; k0 < K_pad; k0 += TC_BK) {
            // GEMM 1: x[16 j][TC_BK k] as 8 n-tiles of accumulator fragments
            float acc[8][4];
            #pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.0f;
            }
            #pragma unroll
            for (int p = 0; p < 4; p++) {
                const bf16* bp = s_sm + (k0 + p * 16 + brow) * DPAD + bcol8;
                #pragma unroll
                for (int ks = 0; ks < 4; ks++) {
                    uint32_t bfr[4];
                    ldmatrix_x4(bfr, bp + ks * 16);
                    mma_bf16_m16n8k16(acc[2 * p],     aQ[ks], bfr);
                    mma_bf16_m16n8k16(acc[2 * p + 1], aQ[ks], bfr + 2);
                }
            }

            // Mask invalid cells to NEG_INF, take the running row max.
            float mt0 = NEG_INF, mt1 = NEG_INF;
            #pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                if constexpr (MASKED) {
                    const int kc = k0 + nt * 8 + 2 * tig;
                    const float k0f = kmul[kc], k1f = kmul[kc + 1];
                    acc[nt][0] = (jm0 * k0f > 0.5f) ? acc[nt][0] : NEG_INF;
                    acc[nt][1] = (jm0 * k1f > 0.5f) ? acc[nt][1] : NEG_INF;
                    acc[nt][2] = (jm1 * k0f > 0.5f) ? acc[nt][2] : NEG_INF;
                    acc[nt][3] = (jm1 * k1f > 0.5f) ? acc[nt][3] : NEG_INF;
                }
                mt0 = fmaxf(mt0, fmaxf(acc[nt][0], acc[nt][1]));
                mt1 = fmaxf(mt1, fmaxf(acc[nt][2], acc[nt][3]));
            }
            #pragma unroll
            for (int off = 1; off <= 2; off <<= 1) {
                mt0 = fmaxf(mt0, __shfl_xor_sync(0xFFFFFFFF, mt0, off));
                mt1 = fmaxf(mt1, __shfl_xor_sync(0xFFFFFFFF, mt1, off));
            }

            // Online-softmax rescale of the running state.
            const float mn0 = fmaxf(m0, mt0), mn1 = fmaxf(m1, mt1);
            const float a0 = __expf(m0 - mn0), a1 = __expf(m1 - mn1);
            l0 *= a0; l1 *= a1;
            #pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                U[nt][0] *= a0; U[nt][1] *= a0;
                U[nt][2] *= a1; U[nt][3] *= a1;
            }
            m0 = mn0; m1 = mn1;

            // exp + repack: two adjacent GEMM-1 C tiles form one GEMM-2 A
            // fragment (identical thread layouts) — no shuffles, no smem.
            uint32_t pfr[4][4];
            #pragma unroll
            for (int s2 = 0; s2 < 4; s2++) {
                #pragma unroll
                for (int half = 0; half < 2; half++) {
                    const int nt = 2 * s2 + half;
                    float p0, p1, p2, p3;
                    if constexpr (MASKED) {
                        p0 = (acc[nt][0] < TC_MASKED_THRESH) ? 0.0f : __expf(acc[nt][0] - mn0);
                        p1 = (acc[nt][1] < TC_MASKED_THRESH) ? 0.0f : __expf(acc[nt][1] - mn0);
                        p2 = (acc[nt][2] < TC_MASKED_THRESH) ? 0.0f : __expf(acc[nt][2] - mn1);
                        p3 = (acc[nt][3] < TC_MASKED_THRESH) ? 0.0f : __expf(acc[nt][3] - mn1);
                    } else {
                        p0 = __expf(acc[nt][0] - mn0);
                        p1 = __expf(acc[nt][1] - mn0);
                        p2 = __expf(acc[nt][2] - mn1);
                        p3 = __expf(acc[nt][3] - mn1);
                    }
                    l0 += p0 + p1;
                    l1 += p2 + p3;
                    pfr[s2][2 * half + 0] = pack_bf162(p0, p1);
                    pfr[s2][2 * half + 1] = pack_bf162(p2, p3);
                }
            }

            // GEMM 2: U += P @ V2 (V2 row-major, B fragments via ldmatrix.trans).
            #pragma unroll
            for (int s2 = 0; s2 < 4; s2++) {
                const bf16* bp = v2_sm + (k0 + s2 * 16 + lrow) * DPAD + lcol8;
                #pragma unroll
                for (int np = 0; np < 4; np++) {
                    uint32_t bfr[4];
                    ldmatrix_x4_trans(bfr, bp + np * 16);
                    mma_bf16_m16n8k16(U[2 * np],     pfr[s2], bfr);
                    mma_bf16_m16n8k16(U[2 * np + 1], pfr[s2], bfr + 2);
                }
            }
        }

        // ---- epilogue: V1-weighted row collapse of this warp's 16 rows ----
        #pragma unroll
        for (int off = 1; off <= 2; off <<= 1) {
            l0 += __shfl_xor_sync(0xFFFFFFFF, l0, off);
            l1 += __shfl_xor_sync(0xFFFFFFFF, l1, off);
        }
        float Mw = fmaxf(m0, m1);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            Mw = fmaxf(Mw, __shfl_xor_sync(0xFFFFFFFF, Mw, off));
        }
        const float w0 = __expf(m0 - Mw), w1 = __expf(m1 - Mw);

        float Lw = w0 * l0 + w1 * l1;
        #pragma unroll
        for (int off = 4; off <= 16; off <<= 1) {
            Lw += __shfl_xor_sync(0xFFFFFFFF, Lw, off);
        }

        float nacc[16];
        const bf16* v1r0 = v1_sm + (jw + g) * DPAD + 2 * tig;
        const bf16* v1r1 = v1_sm + (jw + g + 8) * DPAD + 2 * tig;
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            const float2 v10 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(v1r0 + nt * 8));
            const float2 v11 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(v1r1 + nt * 8));
            nacc[2 * nt + 0] = w0 * v10.x * U[nt][0] + w1 * v11.x * U[nt][2];
            nacc[2 * nt + 1] = w0 * v10.y * U[nt][1] + w1 * v11.y * U[nt][3];
        }
        #pragma unroll
        for (int off = 4; off <= 16; off <<= 1) {
            #pragma unroll
            for (int e = 0; e < 16; e++) {
                nacc[e] += __shfl_xor_sync(0xFFFFFFFF, nacc[e], off);
            }
        }

        if (lane < 4) {
            #pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                wN[warp * D + nt * 8 + 2 * lane]     = nacc[2 * nt + 0];
                wN[warp * D + nt * 8 + 2 * lane + 1] = nacc[2 * nt + 1];
            }
        }
        if (lane == 0) {
            wML[warp * 2]     = Mw;
            wML[warp * 2 + 1] = Lw;
        }
        __syncthreads();

        // ---- fold the 8 warp results into the CTA running (M, L, N) ----
        const float Mold = redML[0], Lold = redML[1];
        float Mnew = Mold;
        #pragma unroll
        for (int wd = 0; wd < TC_WARPS; wd++) Mnew = fmaxf(Mnew, wML[wd * 2]);
        const float aR = __expf(Mold - Mnew);
        float nNew = 0.0f;
        if (tid < D) {
            nNew = redN[tid] * aR;
            #pragma unroll
            for (int wd = 0; wd < TC_WARPS; wd++) {
                nNew += __expf(wML[wd * 2] - Mnew) * wN[wd * D + tid];
            }
        }
        float lNew = Lold * aR;
        #pragma unroll
        for (int wd = 0; wd < TC_WARPS; wd++) {
            lNew += __expf(wML[wd * 2] - Mnew) * wML[wd * 2 + 1];
        }
        __syncthreads();
        if (tid < D) redN[tid] = nNew;
        if (tid == 0) { redML[0] = Mnew; redML[1] = lNew; }
    }
    __syncthreads();

    // ---- final normalize + direct output (reducer semantics) ----
    const float Lfin = redML[1];
    const float inv = (Lfin > 1e-20f) ? (1.0f / Lfin) : 0.0f;
    const int64_t ybase = bh * I + i;
    for (int d = tid; d < D; d += blockDim.x) {
        Y[ybase * D + d] = f2bf(redN[d] * inv);
    }
    if (tid == 0) {
        if (m_i_out) m_i_out[ybase] = redML[0];
        if (l_i_out) l_i_out[ybase] = Lfin;
    }
#endif  // __CUDA_ARCH__ >= 800
}


// =============================================================================
// Split gather (Yq/Yr/Ys fallback path, any D)
// =============================================================================
// Warp-parallel split gather, written in Yq terms; as with Y_gather_tc, the
// score symmetry lets Yr and Ys launch the same kernel with roles permuted
// (see the role table at the tensor-core kernel above). Four anchor vectors
// per block, one per warp; the J range is split across num_j_chunks blocks
// and K is walked in 16x16 score tiles with warp-shuffle online softmax.
// Each block emits per-chunk (O, m, l) partials that reduce_gather_partials
// combines exactly.
//
// I/J/K are the *padded* sizes (multiple of TILE_J/K) used for tile loops and
// shared-memory loads. J_valid/K_valid are the *original* (pre-pad) sizes used
// to mask the dot-product softmax: cells with j_global >= J_valid or
// k_global >= K_valid get NEG_INF so the padded slots drop out of both the
// running max and the denominator, matching the unpadded reference.
template<int D_CONST>
__global__
void Y_gather(
    const bf16* __restrict__ Q_bf,
    const bf16* __restrict__ R_bf,
    const bf16* __restrict__ S_bf,
    const bf16* __restrict__ V1_bf,
    const bf16* __restrict__ V2_bf,
    float*       __restrict__ Y,
    float*       __restrict__ m_i_out,
    float*       __restrict__ l_i_out,
    const bool*  __restrict__ mask,
    int B, int H, int I, int J, int K, float scale,
    int num_j_chunks,
    int J_valid, int K_valid)
{
    const int j_chunk = blockIdx.x % num_j_chunks;
    const int i_base = (blockIdx.x / num_j_chunks) * N_I_GATHER;
    const int h = blockIdx.y;
    const int b = blockIdx.z;

    // Compute j-range for this chunk
    const int total_j_tiles = (J + TILE_J - 1) / TILE_J;
    const int j_tpc = (total_j_tiles + num_j_chunks - 1) / num_j_chunks;
    const int j_start = j_chunk * j_tpc * TILE_J;
    const int j_end = min(j_start + j_tpc * TILE_J, J);

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int block_size = blockDim.x;
    constexpr int DP = D_CONST + GATHER_SMEM_PAD;

    extern __shared__ char smem_raw[];
    bf16* q_vecs  = reinterpret_cast<bf16*>(smem_raw);
    bf16* r_tile  = q_vecs + N_I_GATHER * D_CONST;
    bf16* s_tile  = r_tile + TILE_J * DP;
    bf16* v1_tile = s_tile + TILE_K * DP;
    bf16* v2_tile = v1_tile + TILE_J * DP;
    float* p_tiles = reinterpret_cast<float*>(v2_tile + TILE_K * DP);
    float* m_l_sh  = p_tiles + N_I_GATHER * TILE_J * TILE_K;
    float* o_sh    = m_l_sh + N_I_GATHER * 2;
    
    const int my_i = i_base + warp_id;
    const bool my_i_valid = (my_i < I);
    bf16* my_q = q_vecs + warp_id * D_CONST;
    float* my_p = p_tiles + warp_id * TILE_J * TILE_K;
    float* my_o = o_sh + warp_id * D_CONST;
    float* my_ml = m_l_sh + warp_id * 2;

    for (int n = 0; n < N_I_GATHER; n++) {
        int i_global = i_base + n;
        if (i_global < I) {
            const int64_t q_off = (((int64_t)b * H + h) * I + i_global) * D_CONST;
            for (int d = tid; d < D_CONST; d += block_size) {
                q_vecs[n * D_CONST + d] = Q_bf[q_off + d];
            }
        }
    }

    if (tid < N_I_GATHER * 2) {
        m_l_sh[tid] = (tid % 2 == 0) ? NEG_INF : 0.0f;
    }
    for (int idx = tid; idx < N_I_GATHER * D_CONST; idx += block_size) {
        o_sh[idx] = 0.0f;
    }
    __syncthreads();

    for (int j0 = j_start; j0 < j_end; j0 += TILE_J) {
        // R / V1 depend only on j0, so load them once per j-tile rather than
        // re-fetching on every k0 iteration (the inner loop only varies k).
        for (int idx = tid; idx < TILE_J * D_CONST; idx += block_size) {
            int jt = idx / D_CONST;
            int d = idx % D_CONST;
            int j_global = j0 + jt;
            if (j_global < J) {
                const int64_t r_off = (((int64_t)b * H + h) * J + j_global) * D_CONST + d;
                r_tile[jt * DP + d] = R_bf[r_off];
                v1_tile[jt * DP + d] = V1_bf[r_off];
            }
        }
        __syncthreads();

        for (int k0 = 0; k0 < K; k0 += TILE_K) {
            
            for (int idx = tid; idx < TILE_K * D_CONST; idx += block_size) {
                int kt = idx / D_CONST;
                int d = idx % D_CONST;
                int k_global = k0 + kt;
                if (k_global < K) {
                    const int64_t s_off = (((int64_t)b * H + h) * K + k_global) * D_CONST + d;
                    s_tile[kt * DP + d] = S_bf[s_off];
                    v2_tile[kt * DP + d] = V2_bf[s_off];
                }
            }
            __syncthreads();

            #pragma unroll 1
            for (int cell = 0; cell < 8; cell++) {
                int cell_idx = lane_id + cell * 32;
                if (cell_idx < TILE_J * TILE_K) {
                    int jt = cell_idx / TILE_K;
                    int kt = cell_idx % TILE_K;
                    const int j_global = j0 + jt;
                    const int k_global = k0 + kt;
                    
                    float dot = 0.0f;
                    const bool cell_valid = my_i_valid
                        && (j_global < J_valid)
                        && (k_global < K_valid)
                        && mask_pair_allowed(mask, I, b, my_i, j_global, k_global);
                    if (cell_valid) {
                        float d_accum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                        #pragma unroll 4
                        for (int d = 0; d < D_CONST; d += 4) {
                            const float2 q01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(my_q + d));
                            const float2 q23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(my_q + d + 2));
                            const float2 r01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(r_tile + jt * DP + d));
                            const float2 r23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(r_tile + jt * DP + d + 2));
                            const float2 s01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(s_tile + kt * DP + d));
                            const float2 s23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(s_tile + kt * DP + d + 2));
                            d_accum[0] += q01.x * r01.x * s01.x;
                            d_accum[1] += q01.y * r01.y * s01.y;
                            d_accum[2] += q23.x * r23.x * s23.x;
                            d_accum[3] += q23.y * r23.y * s23.y;
                        }
                        dot = d_accum[0] + d_accum[1] + d_accum[2] + d_accum[3];
                        my_p[cell_idx] = dot;
                    } else {
                        my_p[cell_idx] = NEG_INF;
                    }
                }
            }
            __syncthreads();

            float m_ij = NEG_INF;
            #pragma unroll
            for (int cell = 0; cell < 8; cell++) {
                int cell_idx = lane_id + cell * 32;
                if (cell_idx < TILE_J * TILE_K) {
                    m_ij = fmaxf(m_ij, my_p[cell_idx] * scale);
                }
            }
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                m_ij = fmaxf(m_ij, __shfl_down_sync(0xFFFFFFFF, m_ij, offset));
            }
            m_ij = __shfl_sync(0xFFFFFFFF, m_ij, 0);
            
            float l_ij = 0.0f;
            #pragma unroll
            for (int cell = 0; cell < 8; cell++) {
                int cell_idx = lane_id + cell * 32;
                if (cell_idx < TILE_J * TILE_K) {
                    int jt = cell_idx / TILE_K;
                    int kt = cell_idx % TILE_K;
                    const int j_global = j0 + jt;
                    const int k_global = k0 + kt;
                    const bool cell_valid = my_i_valid
                        && (j_global < J_valid)
                        && (k_global < K_valid)
                        && mask_pair_allowed(mask, I, b, my_i, j_global, k_global);
                    if (cell_valid) {
                        float p_val = expf(my_p[cell_idx] * scale - m_ij);
                        my_p[cell_idx] = p_val;
                        l_ij += p_val;
                    } else {
                        my_p[cell_idx] = 0.0f;
                    }
                }
            }
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                l_ij += __shfl_down_sync(0xFFFFFFFF, l_ij, offset);
            }
            l_ij = __shfl_sync(0xFFFFFFFF, l_ij, 0);

            float m_old = my_ml[0];
            float l_old = my_ml[1];
            float m_new = fmaxf(m_old, m_ij);
            float alpha = expf(m_old - m_new);
            float beta = expf(m_ij - m_new);
            float l_new = alpha * l_old + beta * l_ij;

            // Every lane read my_ml above; lane 0 is about to overwrite it.
            // Independent thread scheduling lets lane 0 run ahead, so the
            // reads need a fence of their own (racecheck flags this).
            __syncwarp();
            if (lane_id == 0) {
                my_ml[0] = m_new;
                my_ml[1] = l_new;
            }
            alpha = __shfl_sync(0xFFFFFFFF, alpha, 0);
            beta = __shfl_sync(0xFFFFFFFF, beta, 0);
            l_old = __shfl_sync(0xFFFFFFFF, l_old, 0);
            l_new = __shfl_sync(0xFFFFFFFF, l_new, 0);

            for (int d = lane_id * 2; d < D_CONST; d += 64) {
                float2 new_o = make_float2(0.0f, 0.0f);
                if (my_i_valid) {
                    float2 T_k[TILE_K];
                    #pragma unroll
                    for (int kt = 0; kt < TILE_K; kt++) T_k[kt] = make_float2(0.0f, 0.0f);

                    for (int jt = 0; jt < TILE_J; jt++) {
                        if (j0 + jt >= J) continue;
                        const float2 v1_val =
                            __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(v1_tile + jt * DP + d));
                        #pragma unroll
                        for (int kt = 0; kt < TILE_K; kt++) {
                            const float p = my_p[jt * TILE_K + kt];
                            T_k[kt].x += p * v1_val.x;
                            T_k[kt].y += p * v1_val.y;
                        }
                    }

                    #pragma unroll
                    for (int kt = 0; kt < TILE_K; kt++) {
                        if (k0 + kt < K) {
                            const float2 v2_val =
                                __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(v2_tile + kt * DP + d));
                            new_o.x += T_k[kt].x * v2_val.x;
                            new_o.y += T_k[kt].y * v2_val.y;
                        }
                    }
                }
                
                if (l_new > 1e-20f) {
                    my_o[d] = (alpha * l_old * my_o[d] + beta * new_o.x) / l_new;
                    my_o[d + 1] = (alpha * l_old * my_o[d + 1] + beta * new_o.y) / l_new;
                }
            }
            
            __syncthreads();
        }
    }

    for (int n = 0; n < N_I_GATHER; n++) {
        int i_global = i_base + n;
        if (i_global < I) {
            const int64_t out_off = ((((int64_t)b * H + h) * I + i_global) * num_j_chunks + j_chunk) * D_CONST;
            for (int d = tid; d < D_CONST; d += block_size) {
                Y[out_off + d] = o_sh[n * D_CONST + d];
            }
        }
    }

    if (lane_id == 0 && m_i_out != nullptr && l_i_out != nullptr && my_i_valid) {
        int64_t stats_idx = (((int64_t)b * H + h) * I + my_i) * num_j_chunks + j_chunk;
        m_i_out[stats_idx] = my_ml[0];
        l_i_out[stats_idx] = my_ml[1];
    }
}


// =============================================================================
// Reduce kernel for split-gather partial results
// =============================================================================
// Combines partial (O, m, l) from multiple chunks using log-sum-exp correction.
// Each block handles one output vector: Y[b,h,n,:] from num_chunks partials.
// Grid: (N, H, B), Block: (D_CONST) threads

template<int D_CONST>
__global__
void reduce_gather_partials(
    const float* __restrict__ O_partial,  // [B, H, N, num_chunks, D]
    const float* __restrict__ m_partial,  // [B, H, N, num_chunks]
    const float* __restrict__ l_partial,  // [B, H, N, num_chunks]
    bf16* __restrict__ Y,                 // [B, H, N, D]
    float* __restrict__ m_out,            // [B, H, N]
    float* __restrict__ l_out,            // [B, H, N]
    int N, int num_chunks)
{
    const int n = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    const int tid = threadIdx.x;
    const int H = gridDim.y;

    if (n >= N) return;

    const int64_t base = ((int64_t)b * H + h) * N + n;
    const int64_t ml_off = base * num_chunks;
    const int64_t o_off = ml_off * D_CONST;
    const int64_t y_off = base * D_CONST;

    // Pass 1: find global max and per-chunk correction factors.
    float alpha_l[MAX_SPLIT_CHUNKS];
    float m_global = NEG_INF;
    for (int c = 0; c < num_chunks; c++) {
        float mc = m_partial[ml_off + c];
        m_global = fmaxf(m_global, mc);
        alpha_l[c] = mc;  // temporarily store m_c
    }

    float l_global = 0.0f;
    if (m_global > NEG_INF * 0.5f) {
        for (int c = 0; c < num_chunks; c++) {
            alpha_l[c] = expf(alpha_l[c] - m_global) * l_partial[ml_off + c];
            l_global += alpha_l[c];
        }
    } else {
        for (int c = 0; c < num_chunks; c++) {
            alpha_l[c] = 0.0f;
        }
    }

    // Pass 2: combine partial outputs
    float inv_l = (l_global > 1e-20f) ? (1.0f / l_global) : 0.0f;
    for (int d = tid; d < D_CONST; d += blockDim.x) {
        float o = 0.0f;
        for (int c = 0; c < num_chunks; c++) {
            o += O_partial[o_off + c * D_CONST + d] * alpha_l[c];
        }
        Y[y_off + d] = f2bf(o * inv_l);
    }

    // Write final softmax stats (needed by scatter kernels and backward pass)
    if (tid == 0) {
        if (m_out) m_out[base] = m_global;
        if (l_out) l_out[base] = l_global;
    }
}


// =============================================================================
// Reduce kernel for split-scatter partial results
// =============================================================================
// Combines partial outputs from multiple chunks by simple addition.
// Unlike gather reduce (which needs log-sum-exp), scatter is just a sum.
// Each block handles one output vector: Y[b,h,n,:] = sum_c Y_partial[b,h,n,c,:]
// Grid: (N_out, H, B), Block: (D_CONST) threads

template<int D_CONST>
__global__
void reduce_scatter_partials(
    const float* __restrict__ Y_partial,  // [B*H*N_out*num_chunks*D]
    bf16* __restrict__ Y,                 // [B*H*N_out*D]
    int N_out, int num_chunks)
{
    const int n = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    const int tid = threadIdx.x;
    const int H = gridDim.y;

    if (n >= N_out) return;

    const int64_t base = ((int64_t)b * H + h) * N_out + n;
    const int64_t partial_base = base * num_chunks * D_CONST;
    const int64_t y_off = base * D_CONST;

    for (int d = tid; d < D_CONST; d += blockDim.x) {
        float sum = 0.0f;
        for (int c = 0; c < num_chunks; c++) {
            sum += Y_partial[partial_base + c * D_CONST + d];
        }
        Y[y_off + d] = f2bf(sum);
    }
}


// =============================================================================
// Scatter Kernels
// =============================================================================

/**
 * @brief Y[anchor,:] = sum_{a,c} A[..] * V_loop1[a,:] * V_loop2[c,:]
 *
 * Uses parallel outer-product approach for computing 3-way attention scores.
 * Each block processes a tile of anchor outputs, iterating over all loop1 and
 * loop2 tiles. This eliminates atomic operations by ensuring each output is
 * owned by exactly one block.
 *
 * The trilinear score is symmetric in Q/R/S, so Y_q_/Y_r_/Y_s_ are the same
 * kernel with permuted roles: anchor is the output mode, loop1 is the split
 * (outer) mode, loop2 the inner one; the two loop modes supply the softmax
 * stats and the V2 operands, the anchor's own m/l are never consumed.
 *
 * Grid:  (ceil(n_anchor/TILE_I) * num_chunks, H, B)
 * Block: 256 threads (1D)
 *
 * Requirements:
 * - D must be a multiple of 4
 * - I, J, K must be equal and multiples of TILE_I (16)
 * - I, J, K must be >= TILE_I
 */
template<int D_CONST>
__global__
__launch_bounds__(256, 2)
void Y_scatter(
    const bf16* __restrict__ anchor,
    const bf16* __restrict__ loop1,
    const bf16* __restrict__ loop2,
    const bf16* __restrict__ V_loop1,
    const bf16* __restrict__ V_loop2,
    const float* __restrict__ m1_in,
    const float* __restrict__ l1_in,
    const float* __restrict__ m2_in,
    const float* __restrict__ l2_in,
    float* __restrict__ Y_part,
    const bool* __restrict__ mask,
    int B, int H, int n_anchor, int n_loop1, int n_loop2, float scale,
    int num_chunks
) {
    // --- Grid/Block Mapping (split over loop1) ---
    const int b = blockIdx.z;
    const int h = blockIdx.y;
    const int chunk = blockIdx.x % num_chunks;
    const int a_start = (blockIdx.x / num_chunks) * TILE_I;

    // Compute loop1-range for this chunk
    const int total_1_tiles = (n_loop1 + TILE_J - 1) / TILE_J;
    const int tpc_1 = (total_1_tiles + num_chunks - 1) / num_chunks;
    const int loop1_start = chunk * tpc_1 * TILE_J;
    const int loop1_end = min(loop1_start + tpc_1 * TILE_J, n_loop1);

    const int tid = threadIdx.x;
    const int tpb = blockDim.x;
    constexpr int DP = D_CONST + SCAT_SMEM_PAD;
    constexpr int ATTN_K_STRIDE = TILE_K + SCAT_ATTN_PAD;
    constexpr int ATTN_I_STRIDE = TILE_J * ATTN_K_STRIDE;

    // --- Memory Offsets ---
    const int64_t a_bh_offset = (int64_t)(b * H + h) * n_anchor * D_CONST;
    const int64_t l1_bh_offset = (int64_t)(b * H + h) * n_loop1 * D_CONST;
    const int64_t l2_bh_offset = (int64_t)(b * H + h) * n_loop2 * D_CONST;
    const int64_t m1_bh_offset = (int64_t)(b * H + h) * n_loop1;
    const int64_t m2_bh_offset = (int64_t)(b * H + h) * n_loop2;

    // --- Shared Memory Layout ---
    extern __shared__ char smem_raw[];
    bf16* a_tile = reinterpret_cast<bf16*>(smem_raw);
    bf16* l1_tile = a_tile + TILE_I * DP;
    bf16* l2_tile = l1_tile + TILE_J * DP;
    bf16* v1_tile = l2_tile + TILE_K * DP;
    bf16* v2_tile = v1_tile + TILE_J * DP;
    float* attn_tile = reinterpret_cast<float*>(v2_tile + TILE_K * DP);
    float* m1_tile = attn_tile + TILE_I * ATTN_I_STRIDE;
    float* ln1_tile = m1_tile + TILE_J;
    float* m2_tile = ln1_tile + TILE_J;
    float* ln2_tile = m2_tile + TILE_K;

    // --- Thread Indexing for Cooperative Loads ---
    const int i_load = tid / D_CONST;
    const int d_load = tid % D_CONST;
    constexpr int load_pairs = D_CONST / 2;
    constexpr int load_iters = (TILE_I * load_pairs + 256 - 1) / 256;
    constexpr int load_step = 256 / D_CONST;

    // --- Per-Thread Output Accumulators (registers) ---
    float2 y_acc[8];
    #pragma unroll
    for (int n = 0; n < 8; n++) {
        y_acc[n] = make_float2(0.0f, 0.0f);
    }

    // --- Load Anchor Tile (fixed for entire block) ---
    for (int n = 0; n < TILE_I; n += load_step) {
        int a_global = a_start + n + i_load;
        if (n + i_load < TILE_I && a_global < n_anchor) {
            a_tile[(n + i_load) * DP + d_load] = anchor[a_bh_offset + a_global * D_CONST + d_load];
        }
    }
    __syncthreads();

    // --- Main Loop: Iterate Over loop1 Tiles (split range) ---
    for (int jt = loop1_start; jt < loop1_end; jt += TILE_J) {
        for (int n = 0; n < TILE_J; n += load_step) {
            int j_global = jt + n + i_load;
            if (n + i_load < TILE_J && j_global < n_loop1) {
                l1_tile[(n + i_load) * DP + d_load] = loop1[l1_bh_offset + j_global * D_CONST + d_load];
            }
        }
        for (int n = 0; n < TILE_J; n += load_step) {
            int j_global = jt + n + i_load;
            if (n + i_load < TILE_J && j_global < n_loop1) {
                v1_tile[(n + i_load) * DP + d_load] = V_loop1[l1_bh_offset + j_global * D_CONST + d_load];
            }
        }
        if (tid < TILE_J && jt + tid < n_loop1) {
            const float l = l1_in[m1_bh_offset + jt + tid];
            if (l <= DENOM_EPS) {
                m1_tile[tid] = 0.0f;
                ln1_tile[tid] = 0.0f;
            } else {
                m1_tile[tid] = m1_in[m1_bh_offset + jt + tid];
                ln1_tile[tid] = 1.0f / l;
            }
        }

        // --- Inner Loop: Iterate Over loop2 Tiles ---
        for (int kt = 0; kt < n_loop2; kt += TILE_K) {
            for (int n = 0; n < TILE_K; n += load_step) {
                int k_global = kt + n + i_load;
                if (n + i_load < TILE_K && k_global < n_loop2) {
                    l2_tile[(n + i_load) * DP + d_load] = loop2[l2_bh_offset + k_global * D_CONST + d_load];
                }
            }
            if (tid < TILE_K && kt + tid < n_loop2) {
                const float l = l2_in[m2_bh_offset + kt + tid];
                if (l <= DENOM_EPS) {
                    m2_tile[tid] = 0.0f;
                    ln2_tile[tid] = 0.0f;
                } else {
                    m2_tile[tid] = m2_in[m2_bh_offset + kt + tid];
                    ln2_tile[tid] = 1.0f / l;
                }
            }
            __syncthreads();

            // --- Parallel Outer-Product Attention Computation ---
            float acc[4][4][4];
            #pragma unroll
            for (int i0 = 0; i0 < 4; i0++) {
                #pragma unroll
                for (int i1 = 0; i1 < 4; i1++) {
                    #pragma unroll
                    for (int i2 = 0; i2 < 4; i2++) {
                        acc[i0][i1][i2] = 0.0f;
                    }
                }
            }

            const int da = tid / (TILE_I * TILE_J * TILE_K / 64);
            const int ia = (tid / (TILE_J * TILE_K / 16)) % (TILE_I / 4);
            const int ja = (tid / (TILE_K / 4)) % (TILE_J / 4);
            const int ka = tid % (TILE_K / 4);

            float2 qa[4], ra[4], sa[4];
            constexpr int d_per_slice = D_CONST / 4;

            for (int db = 0; db < d_per_slice; db += 2) {
                const int d_idx = da * d_per_slice + db;
                #pragma unroll
                for (int u = 0; u < 4; u++) {
                    qa[u] = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(a_tile + (ia * 4 + u) * DP + d_idx));
                    ra[u] = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(l1_tile + (ja * 4 + u) * DP + d_idx));
                    sa[u] = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(l2_tile + (ka * 4 + u) * DP + d_idx));
                }
                #pragma unroll
                for (int i0 = 0; i0 < 4; i0++) {
                    #pragma unroll
                    for (int i1 = 0; i1 < 4; i1++) {
                        // Hoist q*r outside the i2 loop: each (i0,i1) is reused
                        // across all 4 values of i2, so the two q*r FMULs need
                        // happen only once per (i0,i1,db) instead of per i2.
                        const float qrx = qa[i0].x * ra[i1].x;
                        const float qry = qa[i0].y * ra[i1].y;
                        #pragma unroll
                        for (int i2 = 0; i2 < 4; i2++) {
                            acc[i0][i1][i2] += qrx * sa[i2].x + qry * sa[i2].y;
                        }
                    }
                }
            }

            // --- Reduction Across Dimension Slices ---
            for (int u = 1; u < 4; u++) {
                if (da == u) {
                    #pragma unroll
                    for (int i0 = 0; i0 < 4; i0++) {
                        #pragma unroll
                        for (int i1 = 0; i1 < 4; i1++) {
                            #pragma unroll
                            for (int i2 = 0; i2 < 4; i2++) {
                                attn_tile[(ia * 4 + i0) * ATTN_I_STRIDE +
                                          (ja * 4 + i1) * ATTN_K_STRIDE +
                                          (ka * 4 + i2)] = acc[i0][i1][i2];
                            }
                        }
                    }
                }
                __syncthreads();

                if (da == 0) {
                    #pragma unroll
                    for (int i0 = 0; i0 < 4; i0++) {
                        #pragma unroll
                        for (int i1 = 0; i1 < 4; i1++) {
                            #pragma unroll
                            for (int i2 = 0; i2 < 4; i2++) {
                                acc[i0][i1][i2] += attn_tile[(ia * 4 + i0) * ATTN_I_STRIDE +
                                                             (ja * 4 + i1) * ATTN_K_STRIDE +
                                                             (ka * 4 + i2)];
                            }
                        }
                    }
                }
                __syncthreads();
            }

            // --- Apply Softmax Scaling ---
            if (da == 0) {
                // Fuse exp(logit-m1t)*l1t * exp(logit-m2t)*l2t into a single
                // exp(2*logit - m1t - m2t) * l1t * l2t: halves MUFU expf traffic.
                const float two_scale = 2.0f * scale;
                #pragma unroll
                for (int i0 = 0; i0 < 4; i0++) {
                    #pragma unroll
                    for (int i1 = 0; i1 < 4; i1++) {
                        const float m1t = m1_tile[ja * 4 + i1];
                        const float l1t = ln1_tile[ja * 4 + i1];
                        #pragma unroll
                        for (int i2 = 0; i2 < 4; i2++) {
                            const int a_global = a_start + ia * 4 + i0;
                            const int j_global = jt + ja * 4 + i1;
                            const int k_global = kt + ka * 4 + i2;
                            // Mask stride is I == n_anchor (scatter requires I == J == K).
                            const bool cell_valid = (a_global < n_anchor)
                                && (j_global < n_loop1)
                                && (k_global < n_loop2)
                                && mask_pair_allowed(mask, n_anchor, b, a_global, j_global, k_global);
                            const float m2t = m2_tile[ka * 4 + i2];
                            const float l2t = ln2_tile[ka * 4 + i2];
                            attn_tile[(ia * 4 + i0) * ATTN_I_STRIDE +
                                      (ja * 4 + i1) * ATTN_K_STRIDE +
                                      (ka * 4 + i2)] = cell_valid
                                        ? __expf(acc[i0][i1][i2] * two_scale - m1t - m2t) * l1t * l2t
                                        : 0.0f;
                        }
                    }
                }
            }
            __syncthreads();

            for (int n = 0; n < TILE_K; n += load_step) {
                int k_global = kt + n + i_load;
                if (n + i_load < TILE_K && k_global < n_loop2) {
                    v2_tile[(n + i_load) * DP + d_load] = V_loop2[l2_bh_offset + k_global * D_CONST + d_load];
                }
            }
            __syncthreads();

            // --- Accumulate Output (factored outer product: f = u^T (A v)) ---
            // dy depends only on (tid % load_pairs); since tpb=256 is a multiple
            // of load_pairs=32, dy is identical across both n=0 and n=1 iters —
            // so v1ts/v2ts can be loaded once and reused.
            const int dy_h = (tid % load_pairs) * 2;
            float2 v1ts[TILE_J];
            float2 v2ts[TILE_K];
            #pragma unroll
            for (int jy = 0; jy < TILE_J; jy++) {
                v1ts[jy] = __bfloat1622float2(
                    *reinterpret_cast<const __nv_bfloat162*>(v1_tile + jy * DP + dy_h));
            }
            #pragma unroll
            for (int ky = 0; ky < TILE_K; ky++) {
                v2ts[ky] = __bfloat1622float2(
                    *reinterpret_cast<const __nv_bfloat162*>(v2_tile + ky * DP + dy_h));
            }
            for (int n = 0; n < load_iters; n++) {
                int tid_n = tid + n * tpb;
                if (tid_n < TILE_I * load_pairs) {
                    int iy = tid_n / load_pairs;

                    // Pair-process 2 jy values per outer iter: 4 parallel inner
                    // FMA chains (vs 2 with single jy), keeping the FFMA pipe
                    // saturated through dependency latency. No extra reduction
                    // adds — each `fx +=` is one FMA, two of them per outer iter.
                    float fx = 0.0f, fy = 0.0f;
                    #pragma unroll
                    for (int jy = 0; jy < TILE_J; jy += 2) {
                        float tx_a = 0.0f, ty_a = 0.0f;
                        float tx_b = 0.0f, ty_b = 0.0f;
                        #pragma unroll
                        for (int ky = 0; ky < TILE_K; ky++) {
                            const float p_a = attn_tile[iy * ATTN_I_STRIDE + (jy + 0) * ATTN_K_STRIDE + ky];
                            const float p_b = attn_tile[iy * ATTN_I_STRIDE + (jy + 1) * ATTN_K_STRIDE + ky];
                            tx_a += p_a * v2ts[ky].x;  tx_b += p_b * v2ts[ky].x;
                            ty_a += p_a * v2ts[ky].y;  ty_b += p_b * v2ts[ky].y;
                        }
                        fx += v1ts[jy + 0].x * tx_a;
                        fx += v1ts[jy + 1].x * tx_b;
                        fy += v1ts[jy + 0].y * ty_a;
                        fy += v1ts[jy + 1].y * ty_b;
                    }
                    y_acc[n].x += fx;
                    y_acc[n].y += fy;
                }
            }
            __syncthreads();
        }
    }

    // --- Write Output to Global Memory (chunked layout) ---
    const int64_t bh_A = a_bh_offset / D_CONST;  // = (b*H + h) * n_anchor
    for (int n = 0; n < load_iters; n++) {
        int tid_n = tid + n * tpb;
        if (tid_n < TILE_I * load_pairs) {
            int iy = tid_n / load_pairs;
            int dy = (tid_n % load_pairs) * 2;
            int a_global = a_start + iy;
            if (a_global < n_anchor) {
                int64_t out_off = (bh_A + a_global) * (int64_t)num_chunks * D_CONST
                                + (int64_t)chunk * D_CONST + dy;
                Y_part[out_off] = y_acc[n].x;
                Y_part[out_off + 1] = y_acc[n].y;
            }
        }
    }
}


std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
           at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> forward_cuda(
    at::Tensor Q, at::Tensor R, at::Tensor S,
    at::Tensor Vq_1, at::Tensor Vq_2,
    at::Tensor Vr_1, at::Tensor Vr_2,
    at::Tensor Vs_1, at::Tensor Vs_2,
    at::Tensor mask,
    double dropout_rate,
    int64_t I_valid, int64_t J_valid, int64_t K_valid) {
    Q = Q.contiguous();  
    R = R.contiguous();  
    S = S.contiguous();
    Vq_1 = Vq_1.contiguous();
    Vq_2 = Vq_2.contiguous();
    Vr_1 = Vr_1.contiguous();
    Vr_2 = Vr_2.contiguous();
    Vs_1 = Vs_1.contiguous();
    Vs_2 = Vs_2.contiguous();
    if (mask.defined()) {
        mask = mask.contiguous();
    }

    const auto B = Q.size(0);
    const auto H = Q.size(1);
    const auto I = Q.size(2);
    const auto J = R.size(2);
    const auto K = S.size(2);
    const auto D = Q.size(3);
    const bool use_mask = mask.defined() && mask.numel() > 0;
    if (use_mask) {
        TORCH_CHECK(mask.scalar_type() == at::kBool, "forward mask must be bool");
        TORCH_CHECK(mask.is_cuda(), "forward mask must be on CUDA device");
        TORCH_CHECK(mask.dim() == 3, "forward mask must have shape [B, N, N]");
        TORCH_CHECK(I == J && J == K, "masked forward requires I == J == K");
        TORCH_CHECK(mask.size(0) == B, "forward mask batch dim mismatch");
        TORCH_CHECK(mask.size(1) == I && mask.size(2) == I,
            "forward mask shape must be [B, N, N] with N matching padded sequence length");
    }
    const bool* mask_ptr = use_mask ? mask.data_ptr<bool>() : nullptr;

    // Default I_valid/J_valid/K_valid to the padded sizes (no masking) when
    // callers don't supply them. This preserves the legacy behavior where the
    // softmax denominator includes padded slots.
    if (I_valid <= 0 || I_valid > I) I_valid = I;
    if (J_valid <= 0 || J_valid > J) J_valid = J;
    if (K_valid <= 0 || K_valid > K) K_valid = K;

    const float scale = 1.0f / sqrtf((float)D); 

    // allocate outputs on GPU
    auto opts = Q.options();
    auto opts_fp32 = Q.options().dtype(at::kFloat);
    auto Y_q  = torch::zeros({B,H,I,D}, opts);
    auto Y_r  = torch::zeros({B,H,J,D}, opts);
    auto Y_s  = torch::zeros({B,H,K,D}, opts);
    auto Y_q_ = torch::zeros({B,H,I,D}, opts);
    auto Y_r_ = torch::zeros({B,H,J,D}, opts);
    auto Y_s_ = torch::zeros({B,H,K,D}, opts);
    
    // Allocate softmax stats tensors - gather kernels will populate these
    // Stats are computed during gather and reused by scatter kernels + backward pass
    auto m_i = torch::zeros({B,H,I}, opts_fp32);
    auto l_i = torch::zeros({B,H,I}, opts_fp32);
    auto m_j = torch::zeros({B,H,J}, opts_fp32);
    auto l_j = torch::zeros({B,H,J}, opts_fp32);
    auto m_k = torch::zeros({B,H,K}, opts_fp32);
    auto l_k = torch::zeros({B,H,K}, opts_fp32);
    
    const int TpB = 128;
    dim3 block(TpB);

    // Create 3 streams for parallel kernel execution
    cudaStream_t streams[3];
    for (int i = 0; i < 3; i++) {
        AT_CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }
    
    // Create events for synchronization barrier between gather and scatter
    cudaEvent_t gather_done[3];
    for (int i = 0; i < 3; i++) {
        AT_CUDA_CHECK(cudaEventCreate(&gather_done[i]));
    }

    TORCH_CHECK(Q.scalar_type() == at::kBFloat16, "forward expects bfloat16 inputs.");
    TORCH_CHECK(D % 4 == 0, "D must be multiple of 4.");
    TORCH_CHECK(D == 16 || D == 32 || D == 64, "forward: unsupported D=", D, ". Supported: 16, 32, 64");

    // D-dispatch to D_TMPL in {16, 32, 64}.
    #define FWD_DISPATCH_D(D_VAL, ...) \
      [&] { \
        if ((D_VAL) == 16)      { constexpr int D_TMPL = 16; __VA_ARGS__; } \
        else if ((D_VAL) == 32) { constexpr int D_TMPL = 32; __VA_ARGS__; } \
        else if ((D_VAL) == 64) { constexpr int D_TMPL = 64; __VA_ARGS__; } \
      }()

    FWD_DISPATCH_D(D, {

    // Split-gather chunk counts (thresholded and capped).
    const int raw_j_chunks_q = (J + TILE_J - 1) / TILE_J;
    const int raw_i_chunks_r = (I + TILE_I - 1) / TILE_I;
    const int raw_i_chunks_s = (I + TILE_I - 1) / TILE_I;
    const auto pick_split_chunks = [](int raw_chunks) {
        const int capped = (raw_chunks > MAX_SPLIT_CHUNKS) ? MAX_SPLIT_CHUNKS : raw_chunks;
        return (capped >= MIN_SPLIT_CHUNKS) ? capped : 1;
    };
    const int num_j_chunks_q = pick_split_chunks(raw_j_chunks_q);
    const int num_i_chunks_r = pick_split_chunks(raw_i_chunks_r);
    const int num_i_chunks_s = pick_split_chunks(raw_i_chunks_s);

    // Allocate partial buffers for split-gather
    // Layout: [B, H, N, num_chunks, D] for O, [B, H, N, num_chunks] for m/l
    at::Tensor O_part_q, m_part_q, l_part_q;
    at::Tensor O_part_r, m_part_r, l_part_r;
    at::Tensor O_part_s, m_part_s, l_part_s;

    O_part_q = torch::empty({B * H * I * num_j_chunks_q * D_TMPL}, opts_fp32);
    m_part_q = torch::empty({B * H * I * num_j_chunks_q}, opts_fp32);
    l_part_q = torch::empty({B * H * I * num_j_chunks_q}, opts_fp32);
    O_part_r = torch::empty({B * H * J * num_i_chunks_r * D_TMPL}, opts_fp32);
    m_part_r = torch::empty({B * H * J * num_i_chunks_r}, opts_fp32);
    l_part_r = torch::empty({B * H * J * num_i_chunks_r}, opts_fp32);
    O_part_s = torch::empty({B * H * K * num_i_chunks_s * D_TMPL}, opts_fp32);
    m_part_s = torch::empty({B * H * K * num_i_chunks_s}, opts_fp32);
    l_part_s = torch::empty({B * H * K * num_i_chunks_s}, opts_fp32);

    // Split-scatter chunk counts (same policy as gather).
    const int raw_scat_j_chunks = (J + TILE_J - 1) / TILE_J;
    const int raw_scat_i_chunks = (I + TILE_I - 1) / TILE_I;
    const int scat_j_chunks_q = pick_split_chunks(raw_scat_j_chunks);
    const int scat_i_chunks_r = pick_split_chunks(raw_scat_i_chunks);
    const int scat_i_chunks_s = pick_split_chunks(raw_scat_i_chunks);

    TORCH_CHECK(num_j_chunks_q <= MAX_SPLIT_CHUNKS, "num_j_chunks_q exceeds MAX_SPLIT_CHUNKS");
    TORCH_CHECK(num_i_chunks_r <= MAX_SPLIT_CHUNKS, "num_i_chunks_r exceeds MAX_SPLIT_CHUNKS");
    TORCH_CHECK(num_i_chunks_s <= MAX_SPLIT_CHUNKS, "num_i_chunks_s exceeds MAX_SPLIT_CHUNKS");

    // Allocate scatter partial buffers
    // Layout: [B*H*N_out*num_chunks*D] — simple additive reduce, no softmax stats
    at::Tensor Yq_scat_part, Yr_scat_part, Ys_scat_part;

    Yq_scat_part = torch::empty({B * H * I * scat_j_chunks_q * D_TMPL}, opts_fp32);
    Yr_scat_part = torch::empty({B * H * J * scat_i_chunks_r * D_TMPL}, opts_fp32);
    Ys_scat_part = torch::empty({B * H * K * scat_i_chunks_s * D_TMPL}, opts_fp32);

    // Scatter shared memory size (same for all 3 role permutations)
    constexpr int SCAT_DP = D_TMPL + SCAT_SMEM_PAD;
    constexpr int SCAT_ATTN_K_STRIDE = TILE_K + SCAT_ATTN_PAD;
    const size_t scatter_smem_size =
        sizeof(bf16) * (
            TILE_I * SCAT_DP +             // a_tile (anchor / output dim)
            TILE_J * SCAT_DP +             // l1_tile
            TILE_K * SCAT_DP +             // l2_tile
            TILE_J * SCAT_DP +             // v1_tile
            TILE_K * SCAT_DP               // v2_tile
        ) +
        sizeof(float) * (
            TILE_I * TILE_J * SCAT_ATTN_K_STRIDE + // attn_tile
            TILE_J + TILE_J +              // m1_tile, ln1_tile (loop1 stats)
            TILE_K + TILE_K                // m2_tile, ln2_tile (loop2 stats)
        );

    // Tensor-core fast path for D=64 and resident dim <= 256 (cuda_docs/gather_readme.md):
    // the fused FlashAttention-style kernel writes Y/m/l directly, no reducer.
    // All three gathers use it with permuted roles. Disable with ATT3_YQ_TC=0.
    bool yq_tc_done = false, yr_tc_done = false, ys_tc_done = false;
    if constexpr (D_TMPL == 64) {
        // sm_80+ only: below that the kernel body compiles to a no-op (bf16
        // mma/ldmatrix/cp.async), so launching it would silently return zeros.
        static const int max_smem_optin = []() {
            int dev = 0, major = 0, v = 0;
            cudaGetDevice(&dev);
            cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
            if (major < 8) return 0;
            cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
            return v;
        }();
        // anchor: one output vector per CTA; rows: GEMM-1 M / V1 side;
        // res: shared-memory-resident side (GEMM-1 N / V2).
        auto launch_tc = [&](const at::Tensor& anchor, const at::Tensor& rows,
                             const at::Tensor& res, const at::Tensor& V1,
                             const at::Tensor& V2, at::Tensor& Yout,
                             at::Tensor& m_out, at::Tensor& l_out,
                             int n_anchor, int n_rows, int rows_valid,
                             int n_res, int res_valid, cudaStream_t stream) -> bool {
            if (!att3_tc::state().fwd_enabled || n_res > TC_MAX_K) return false;
            const int res_pad = ceil_div(n_res, TC_BK) * TC_BK;
            constexpr int TC_DPAD = D_TMPL + 8;
            const size_t smem_tc =
                sizeof(bf16) * ((size_t)2 * TC_BJ * TC_DPAD + (size_t)2 * res_pad * TC_DPAD) +
                sizeof(float) * ((size_t)res_pad + TC_BJ + TC_WARPS * D_TMPL
                                 + TC_WARPS * 2 + D_TMPL + 2 + D_TMPL);
            if (smem_tc > (size_t)max_smem_optin) return false;
            static size_t smem_attr_set = 0;
            if (smem_tc > smem_attr_set) {
                AT_CUDA_CHECK(cudaFuncSetAttribute(
                    Y_gather_tc<64, false>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_tc));
                AT_CUDA_CHECK(cudaFuncSetAttribute(
                    Y_gather_tc<64, true>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_tc));
                smem_attr_set = smem_tc;
            }
            // Fast path also requires whole row-blocks: a partial block's
            // zero-filled Qp rows must be masked out of the softmax.
            const bool tc_masked = (mask_ptr != nullptr)
                || (rows_valid < n_rows) || (res_valid < n_res)
                || (res_pad != n_res) || (n_rows % TC_BJ != 0);
            auto* tc_kernel = tc_masked ? Y_gather_tc<64, true>
                                        : Y_gather_tc<64, false>;
            dim3 grid_tc(n_anchor, H, B);
            tc_kernel<<<grid_tc, dim3(TC_WARPS * 32), smem_tc, stream>>>(
                reinterpret_cast<const bf16*>(anchor.data_ptr<at::BFloat16>()),
                reinterpret_cast<const bf16*>(rows.data_ptr<at::BFloat16>()),
                reinterpret_cast<const bf16*>(res.data_ptr<at::BFloat16>()),
                reinterpret_cast<const bf16*>(V1.data_ptr<at::BFloat16>()),
                reinterpret_cast<const bf16*>(V2.data_ptr<at::BFloat16>()),
                reinterpret_cast<bf16*>(Yout.data_ptr<at::BFloat16>()),
                m_out.data_ptr<float>(),
                l_out.data_ptr<float>(),
                mask_ptr,
                H, n_anchor, n_rows, n_res, res_pad, scale,
                rows_valid, res_valid
            );
            ++att3_tc::state().fwd_launches;
            return true;
        };
        yq_tc_done = launch_tc(Q, R, S, Vr_1, Vs_1, Y_q, m_i, l_i,
                               I, J, (int)J_valid, K, (int)K_valid, streams[0]);
        yr_tc_done = launch_tc(R, Q, S, Vq_1, Vs_1, Y_r, m_j, l_j,
                               J, I, (int)I_valid, K, (int)K_valid, streams[1]);
        ys_tc_done = launch_tc(S, Q, R, Vq_1, Vr_1, Y_s, m_k, l_k,
                               K, I, (int)I_valid, J, (int)J_valid, streams[2]);
    }

    // Legacy split gather + exact reducer, for streams the TC path didn't take.
    // Role permutation matches launch_tc: anchor / rows+V1 (split dim) /
    // cols+V2 (looped dim).
    auto launch_gather = [&](const at::Tensor& anchor, const at::Tensor& rows,
                             const at::Tensor& cols, const at::Tensor& V1,
                             const at::Tensor& V2, at::Tensor& O_part,
                             at::Tensor& m_part, at::Tensor& l_part,
                             at::Tensor& Yout, at::Tensor& m_out, at::Tensor& l_out,
                             int n_anchor, int n_rows, int rows_valid,
                             int n_cols, int cols_valid, int num_chunks,
                             cudaStream_t stream) {
        const int num_anchor_tiles = ceil_div(n_anchor, N_I_GATHER);
        dim3 grid(num_anchor_tiles * num_chunks, H, B);
        constexpr int DP = D_TMPL + GATHER_SMEM_PAD;
        const size_t smem_size =
            sizeof(bf16) * (
                N_I_GATHER * D_TMPL +               // q_vecs
                2 * TILE_J * DP + 2 * TILE_K * DP   // r/s/v1/v2 tiles (padded)
            ) +
            sizeof(float) * (
                N_I_GATHER * TILE_J * TILE_K +      // p_tiles
                N_I_GATHER * 2 +                    // m_l_sh
                N_I_GATHER * D_TMPL                 // o_sh
            );
        Y_gather<D_TMPL><<<grid, block, smem_size, stream>>>(
            reinterpret_cast<const bf16*>(anchor.data_ptr<at::BFloat16>()),
            reinterpret_cast<const bf16*>(rows.data_ptr<at::BFloat16>()),
            reinterpret_cast<const bf16*>(cols.data_ptr<at::BFloat16>()),
            reinterpret_cast<const bf16*>(V1.data_ptr<at::BFloat16>()),
            reinterpret_cast<const bf16*>(V2.data_ptr<at::BFloat16>()),
            O_part.data_ptr<float>(),
            m_part.data_ptr<float>(),
            l_part.data_ptr<float>(),
            mask_ptr,
            B, H, n_anchor, n_rows, n_cols, scale,
            num_chunks,
            rows_valid, cols_valid
        );
        reduce_gather_partials<D_TMPL><<<dim3(n_anchor, H, B), D_TMPL, 0, stream>>>(
            O_part.data_ptr<float>(),
            m_part.data_ptr<float>(),
            l_part.data_ptr<float>(),
            reinterpret_cast<bf16*>(Yout.data_ptr<at::BFloat16>()),
            m_out.data_ptr<float>(),
            l_out.data_ptr<float>(),
            n_anchor, num_chunks
        );
    };

    if (!yq_tc_done) {
        launch_gather(Q, R, S, Vr_1, Vs_1, O_part_q, m_part_q, l_part_q,
                      Y_q, m_i, l_i, I, J, (int)J_valid, K, (int)K_valid,
                      num_j_chunks_q, streams[0]);
    }
    AT_CUDA_CHECK(cudaEventRecord(gather_done[0], streams[0]));

    if (!yr_tc_done) {
        launch_gather(R, Q, S, Vq_1, Vs_1, O_part_r, m_part_r, l_part_r,
                      Y_r, m_j, l_j, J, I, (int)I_valid, K, (int)K_valid,
                      num_i_chunks_r, streams[1]);
    }
    AT_CUDA_CHECK(cudaEventRecord(gather_done[1], streams[1]));

    if (!ys_tc_done) {
        launch_gather(S, Q, R, Vq_1, Vr_1, O_part_s, m_part_s, l_part_s,
                      Y_s, m_k, l_k, K, I, (int)I_valid, J, (int)J_valid,
                      num_i_chunks_s, streams[2]);
    }
    AT_CUDA_CHECK(cudaEventRecord(gather_done[2], streams[2]));

    // Barrier: all scatter kernels wait for all gather kernels to complete
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            AT_CUDA_CHECK(cudaStreamWaitEvent(streams[i], gather_done[j], 0));
        }
    }

    // SCATTER kernels (split outer loops for occupancy, additive reduce).
    // When all V2 operands are zero (scatter unused) the scatter outputs are
    // identically zero; Y_q_/Y_r_/Y_s_ are zero-initialized, so the kernels
    // can be skipped outright. Single host round-trip, mirrors the backward's
    // scatter gate.
    const bool scatter_used =
        (Vq_2.ne(0).any() | Vr_2.ne(0).any() | Vs_2.ne(0).any()).item<bool>();
    if (scatter_used) {

    TORCH_CHECK(I == J && J == K, "Scatter requires I == J == K");
    TORCH_CHECK(I % TILE_I == 0, "Scatter requires I to be a multiple of TILE_I (16)");

    // Role permutation: anchor (output) / loop1 (split dim, V2+stats) /
    // loop2 (inner dim, V2+stats). The anchor's own m/l are never consumed.
    auto launch_scatter = [&](const at::Tensor& anchor, const at::Tensor& loop1,
                              const at::Tensor& loop2, const at::Tensor& V1,
                              const at::Tensor& V2, at::Tensor& m1, at::Tensor& l1,
                              at::Tensor& m2, at::Tensor& l2,
                              at::Tensor& Y_part, at::Tensor& Yout,
                              int n_anchor, int n_loop1, int n_loop2,
                              int num_chunks, cudaStream_t stream) {
        dim3 grid(ceil_div(n_anchor, TILE_I) * num_chunks, H, B);
        Y_scatter<D_TMPL><<<grid, dim3(256), scatter_smem_size, stream>>>(
            reinterpret_cast<const bf16*>(anchor.data_ptr<at::BFloat16>()),
            reinterpret_cast<const bf16*>(loop1.data_ptr<at::BFloat16>()),
            reinterpret_cast<const bf16*>(loop2.data_ptr<at::BFloat16>()),
            reinterpret_cast<const bf16*>(V1.data_ptr<at::BFloat16>()),
            reinterpret_cast<const bf16*>(V2.data_ptr<at::BFloat16>()),
            m1.data_ptr<float>(), l1.data_ptr<float>(),
            m2.data_ptr<float>(), l2.data_ptr<float>(),
            Y_part.data_ptr<float>(),
            mask_ptr,
            B, H, n_anchor, n_loop1, n_loop2, scale,
            num_chunks
        );
        reduce_scatter_partials<D_TMPL><<<dim3(n_anchor, H, B), D_TMPL, 0, stream>>>(
            Y_part.data_ptr<float>(),
            reinterpret_cast<bf16*>(Yout.data_ptr<at::BFloat16>()),
            n_anchor, num_chunks
        );
    };

    launch_scatter(Q, R, S, Vr_2, Vs_2, m_j, l_j, m_k, l_k,
                   Yq_scat_part, Y_q_, I, J, K, scat_j_chunks_q, streams[0]);
    launch_scatter(R, Q, S, Vq_2, Vs_2, m_i, l_i, m_k, l_k,
                   Yr_scat_part, Y_r_, J, I, K, scat_i_chunks_r, streams[1]);
    launch_scatter(S, Q, R, Vq_2, Vr_2, m_i, l_i, m_j, l_j,
                   Ys_scat_part, Y_s_, K, I, J, scat_i_chunks_s, streams[2]);

    } // end if (scatter_used)

    }); // end FWD_DISPATCH_D

    // Synchronize all streams before returning to Python
    for (int i = 0; i < 3; i++) {
        AT_CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }
    
    // Cleanup
    for (int i = 0; i < 3; i++) {
        AT_CUDA_CHECK(cudaEventDestroy(gather_done[i]));
        AT_CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }

    // Return outputs + softmax stats (for reuse in backward pass)
    return std::make_tuple(Y_q, Y_r, Y_s, Y_q_, Y_r_, Y_s_,
                           m_i, l_i, m_j, l_j, m_k, l_k);
}




