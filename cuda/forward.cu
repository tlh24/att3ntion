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
#include <mma.h>

#include "common.cuh"
#include "../cpp/cuda_bindings.h"

// =============================================================================
// Tensor Core Configuration (WMMA)
// =============================================================================
// WMMA tile dimensions for FP16 tensor cores on Ada Lovelace (RTX 4080)
// Using 16x16x16 tiles which map perfectly to TILE_J=16, TILE_K=16

constexpr int WMMA_M = 16;  // Rows of output tile
constexpr int WMMA_N = 16;  // Cols of output tile  
constexpr int WMMA_K = 16;  // Inner dimension for accumulation

// DISABLED: WMMA tensor core path has issues on sm_120+ (Blackwell).
// The WMMA API's internal memory access patterns on these architectures
// may access beyond the declared matrix bounds in shared memory.
// TODO: Re-enable after porting to PTX-level mma or CUTLASS for sm_120+.
#define USE_TENSOR_CORES 0

// ===================== gather kernels ======================

constexpr int N_I_GATHER = 4;  // i-values per block in multi-i gather
constexpr int SMEM_PAD = 1;    // Padding to eliminate bank conflicts

// Multi-i warp-parallel gather: 4 output vectors per block, warp-shuffle softmax
template<int D_CONST>
__global__
void Yq_gather(
    const float4* __restrict__ Q_f4,
    const float4* __restrict__ R_f4,
    const float4* __restrict__ S_f4,
    const float4* __restrict__ V1_f4,
    const float4* __restrict__ V2_f4,
    float*       __restrict__ Y,
    float*       __restrict__ m_i_out,
    float*       __restrict__ l_i_out,
    int B, int H, int I, int J, int K, float scale)
{
    const int i_base = blockIdx.x * N_I_GATHER;
    const int h = blockIdx.y;
    const int b = blockIdx.z;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int block_size = blockDim.x;
    constexpr int DP = D_CONST + SMEM_PAD;

    extern __shared__ float smem[];
    float* q_vecs  = smem;
    float* r_tile  = q_vecs + N_I_GATHER * D_CONST;
    float* s_tile  = r_tile + TILE_J * DP;
    float* v1_tile = s_tile + TILE_K * DP;
    float* v2_tile = v1_tile + TILE_J * DP;
    float* p_tiles = v2_tile + TILE_K * DP;
    float* m_l_sh  = p_tiles + N_I_GATHER * TILE_J * TILE_K;
    float* o_sh    = m_l_sh + N_I_GATHER * 2;
    
    const int my_i = i_base + warp_id;
    const bool my_i_valid = (my_i < I);
    float* my_q = q_vecs + warp_id * D_CONST;
    float* my_p = p_tiles + warp_id * TILE_J * TILE_K;
    float* my_o = o_sh + warp_id * D_CONST;
    float* my_ml = m_l_sh + warp_id * 2;

    for (int n = 0; n < N_I_GATHER; n++) {
        int i_global = i_base + n;
        if (i_global < I) {
            const int64_t q_off = (((int64_t)b * H + h) * I + i_global) * D_CONST;
            for (int d = tid; d < D_CONST; d += block_size) {
                q_vecs[n * D_CONST + d] = ((const float*)Q_f4)[q_off + d];
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

    for (int j0 = 0; j0 < J; j0 += TILE_J) {
        for (int k0 = 0; k0 < K; k0 += TILE_K) {
            
            for (int idx = tid; idx < TILE_J * D_CONST; idx += block_size) {
                int jt = idx / D_CONST;
                int d = idx % D_CONST;
                int j_global = j0 + jt;
                if (j_global < J) {
                    const int64_t r_off = (((int64_t)b * H + h) * J + j_global) * D_CONST + d;
                    r_tile[jt * DP + d] = ((const float*)R_f4)[r_off];
                    v1_tile[jt * DP + d] = ((const float*)V1_f4)[r_off];
                }
            }
            
            for (int idx = tid; idx < TILE_K * D_CONST; idx += block_size) {
                int kt = idx / D_CONST;
                int d = idx % D_CONST;
                int k_global = k0 + kt;
                if (k_global < K) {
                    const int64_t s_off = (((int64_t)b * H + h) * K + k_global) * D_CONST + d;
                    s_tile[kt * DP + d] = ((const float*)S_f4)[s_off];
                    v2_tile[kt * DP + d] = ((const float*)V2_f4)[s_off];
                }
            }
            __syncthreads();

            #pragma unroll 1
            for (int cell = 0; cell < 8; cell++) {
                int cell_idx = lane_id + cell * 32;
                if (cell_idx < TILE_J * TILE_K) {
                    int jt = cell_idx / TILE_K;
                    int kt = cell_idx % TILE_K;
                    
                    float dot = 0.0f;
                    if (my_i_valid && j0 + jt < J && k0 + kt < K) {
                        float d_accum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                        #pragma unroll 4
                        for (int d = 0; d < D_CONST; d += 4) {
                            d_accum[0] += my_q[d+0] * r_tile[jt*DP + d+0] * s_tile[kt*DP + d+0];
                            d_accum[1] += my_q[d+1] * r_tile[jt*DP + d+1] * s_tile[kt*DP + d+1];
                            d_accum[2] += my_q[d+2] * r_tile[jt*DP + d+2] * s_tile[kt*DP + d+2];
                            d_accum[3] += my_q[d+3] * r_tile[jt*DP + d+3] * s_tile[kt*DP + d+3];
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
                    if (my_i_valid && j0 + jt < J && k0 + kt < K) {
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
            
            if (lane_id == 0) {
                my_ml[0] = m_new;
                my_ml[1] = l_new;
            }
            alpha = __shfl_sync(0xFFFFFFFF, alpha, 0);
            beta = __shfl_sync(0xFFFFFFFF, beta, 0);
            l_old = __shfl_sync(0xFFFFFFFF, l_old, 0);
            l_new = __shfl_sync(0xFFFFFFFF, l_new, 0);

            for (int d = lane_id; d < D_CONST; d += 32) {
                float new_o_d = 0.0f;
                if (my_i_valid) {
                    for (int jt = 0; jt < TILE_J; jt++) {
                        if (j0 + jt >= J) continue;
                        float v1_val = v1_tile[jt * DP + d];
                        for (int kt = 0; kt < TILE_K; kt++) {
                            if (k0 + kt >= K) continue;
                            new_o_d += my_p[jt * TILE_K + kt] * v1_val * v2_tile[kt * DP + d];
                        }
                    }
                }
                
                if (l_new > 1e-20f) {
                    my_o[d] = (alpha * l_old * my_o[d] + beta * new_o_d) / l_new;
                }
            }
            
            __syncthreads();
        }
    }

    for (int n = 0; n < N_I_GATHER; n++) {
        int i_global = i_base + n;
        if (i_global < I) {
            const int64_t out_off = (((int64_t)b * H + h) * I + i_global) * D_CONST;
            for (int d = tid; d < D_CONST; d += block_size) {
                Y[out_off + d] = o_sh[n * D_CONST + d];
            }
        }
    }

    if (lane_id == 0 && m_i_out != nullptr && l_i_out != nullptr && my_i_valid) {
        int64_t stats_idx = ((int64_t)b * H + h) * I + my_i;
        m_i_out[stats_idx] = my_ml[0];
        l_i_out[stats_idx] = my_ml[1];
    }
}

// =============================================================================
// TENSOR CORE VERSION: Yq_gather using WMMA (FP16 compute, FP32 accumulate)
// =============================================================================
// This kernel reformulates the triple product as GEMMs to utilize tensor cores:
//   1. Attention: A[j,k] = (Q⊙R)[j,:] @ S[k,:]^T  → WMMA GEMM
//   2. Output:    M[j,d] = P[j,k] @ V2[k,d]       → WMMA GEMM
//   3. Final:     O[d] = Σ_j V1[j,d] * M[j,d]     → Element-wise + reduce
//
// Requirements:
//   - D must be multiple of 16 (WMMA_K)
//   - TILE_J = TILE_K = 16 (matches WMMA_M, WMMA_N)
//   - Block size should be 128 (4 warps)

__global__
void Yq_gather_tensor_core(
    const half* __restrict__ Q_h,      // [B,H,I,D] - FP16 input
    const half* __restrict__ R_h,      // [B,H,J,D]
    const half* __restrict__ S_h,      // [B,H,K,D]
    const half* __restrict__ V1_h,     // [B,H,J,D] - Vr_1
    const half* __restrict__ V2_h,     // [B,H,K,D] - Vs_1
    float*      __restrict__ Y,        // [B,H,I,D] - FP32 output
    float*      __restrict__ m_i_out,  // softmax stats output
    float*      __restrict__ l_i_out,  // softmax stats output
    int B, int H, int I, int J, int K, int D, float scale)
{
    using namespace nvcuda::wmma;
    
    // --- Grid Mapping ---
    // Each block computes one output vector Y[b,h,i,:]
    const int i = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;  // Expected: 128
    const int warp_id = tid / 32;
    const int num_warps = block_size / 32;  // Expected: 4
    (void)num_warps;  // May be unused in simple single-warp WMMA path
    
    // --- Shared Memory Layout ---
    // We need FP16 tiles for WMMA and FP32 buffers for softmax/output
    extern __shared__ char smem_raw[];
    
    // FP16 tiles for tensor core operations
    half* q_vec_h    = (half*)smem_raw;                              // D halfs
    half* qr_tile_h  = q_vec_h + D;                                  // TILE_J * D halfs
    half* s_tile_h   = qr_tile_h + TILE_J * D;                       // TILE_K * D halfs
    half* v1_tile_h  = s_tile_h + TILE_K * D;                        // TILE_J * D halfs
    half* v2_tile_h  = v1_tile_h + TILE_J * D;                       // TILE_K * D halfs
    half* p_tile_h   = v2_tile_h + TILE_K * D;                       // TILE_J * TILE_K halfs
    
    // FP32 buffers for softmax and output accumulation
    float* p_tile_f  = (float*)(p_tile_h + TILE_J * TILE_K);         // TILE_J * TILE_K floats
    float* m_tile_f  = p_tile_f + TILE_J * TILE_K;                   // TILE_J * D floats (intermediate)
    float* red_buf   = m_tile_f + TILE_J * D;                        // block_size floats
    float* m_l_sh    = red_buf + block_size;                         // 2 floats
    float* o_sh      = m_l_sh + 2;                                   // D floats
    
    // --- Load Q[b,h,i,:] to Shared Memory (FP16) ---
    const int64_t q_base_off = (((int64_t)b * H + h) * I + i) * D;
    for (int d = tid; d < D; d += block_size) {
        q_vec_h[d] = Q_h[q_base_off + d];
    }
    
    // --- Initialize Online Softmax State ---
    if (tid == 0) {
        m_l_sh[0] = NEG_INF;  // m_i
        m_l_sh[1] = 0.0f;     // l_i
    }
    for (int d = tid; d < D; d += block_size) {
        o_sh[d] = 0.0f;
    }
    __syncthreads();
    
    // --- Main Loop over R and S tiles ---
    for (int j0 = 0; j0 < J; j0 += TILE_J) {
        for (int k0 = 0; k0 < K; k0 += TILE_K) {
            const int tile_j_size = min(TILE_J, J - j0);
            const int tile_k_size = min(TILE_K, K - k0);
            
            // =========================================================
            // PHASE 1: Load R, S, V1, V2 tiles (FP16) and compute QR
            // =========================================================
            
            // Load R tile and compute QR = Q ⊙ R (element-wise broadcast)
            for (int idx = tid; idx < TILE_J * D; idx += block_size) {
                int jt = idx / D;
                int d = idx % D;
                int j = j0 + jt;
                if (j < J) {
                    int64_t r_off = (((int64_t)b * H + h) * J + j) * D + d;
                    half r_val = R_h[r_off];
                    half q_val = q_vec_h[d];
                    // QR[jt,d] = Q[d] * R[jt,d]
                    qr_tile_h[jt * D + d] = __hmul(q_val, r_val);
                    // Also load V1
                    v1_tile_h[jt * D + d] = V1_h[r_off];
                } else {
                    qr_tile_h[jt * D + d] = __float2half(0.0f);
                    v1_tile_h[jt * D + d] = __float2half(0.0f);
                }
            }
            
            // Load S and V2 tiles
            for (int idx = tid; idx < TILE_K * D; idx += block_size) {
                int kt = idx / D;
                int d = idx % D;
                int k = k0 + kt;
                if (k < K) {
                    int64_t s_off = (((int64_t)b * H + h) * K + k) * D + d;
                    s_tile_h[kt * D + d] = S_h[s_off];
                    v2_tile_h[kt * D + d] = V2_h[s_off];
                } else {
                    s_tile_h[kt * D + d] = __float2half(0.0f);
                    v2_tile_h[kt * D + d] = __float2half(0.0f);
                }
            }
            __syncthreads();
            
            // =========================================================
            // PHASE 2: TENSOR CORE GEMM - Attention Scores
            // A[j,k] = QR[j,:] @ S[k,:]^T = (TILE_J x D) @ (D x TILE_K)
            // =========================================================
            
            // Only warp 0 performs the WMMA for attention scores
            // (Other warps could handle different tiles in a more advanced impl)
            if (warp_id == 0) {
                // Declare WMMA fragments
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> qr_frag;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> s_frag;
                fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
                
                // Initialize accumulator
                fill_fragment(acc_frag, 0.0f);
                
                // Accumulate over D dimension in chunks of WMMA_K=16
                for (int d0 = 0; d0 < D; d0 += WMMA_K) {
                    // Load QR[0:16, d0:d0+16] - row major, stride = D
                    load_matrix_sync(qr_frag, qr_tile_h + d0, D);
                    
                    // Load S[0:16, d0:d0+16]^T - col major means transposed
                    load_matrix_sync(s_frag, s_tile_h + d0, D);
                    
                    // C += A @ B^T
                    mma_sync(acc_frag, qr_frag, s_frag, acc_frag);
                }
                
                // Store result to FP32 shared memory for softmax
                store_matrix_sync(p_tile_f, acc_frag, TILE_K, mem_row_major);
            }
            __syncthreads();
            
            // =========================================================
            // PHASE 3: Softmax (FP32 for numerical stability)
            // =========================================================
            
            // Find tile max (m_ij)
            float m_ij = NEG_INF;
            for (int flat_idx = tid; flat_idx < TILE_J * TILE_K; flat_idx += block_size) {
                int jt = flat_idx / TILE_K;
                int kt = flat_idx % TILE_K;
                if (j0 + jt < J && k0 + kt < K) {
                    m_ij = fmaxf(m_ij, p_tile_f[flat_idx] * scale);
                }
            }
            red_buf[tid] = m_ij;
            __syncthreads();
            for (int s = block_size / 2; s > 0; s >>= 1) {
                if (tid < s) red_buf[tid] = fmaxf(red_buf[tid], red_buf[tid + s]);
                __syncthreads();
            }
            m_ij = red_buf[0];
            __syncthreads();
            
            // Compute softmax numerator and denominator
            float l_ij = 0.0f;
            for (int flat_idx = tid; flat_idx < TILE_J * TILE_K; flat_idx += block_size) {
                int jt = flat_idx / TILE_K;
                int kt = flat_idx % TILE_K;
                if (j0 + jt < J && k0 + kt < K) {
                    float p_val = expf(p_tile_f[flat_idx] * scale - m_ij);
                    p_tile_f[flat_idx] = p_val;
                    l_ij += p_val;
                } else {
                    p_tile_f[flat_idx] = 0.0f;
                }
            }
            red_buf[tid] = l_ij;
            __syncthreads();
            for (int s = block_size / 2; s > 0; s >>= 1) {
                if (tid < s) red_buf[tid] += red_buf[tid + s];
                __syncthreads();
            }
            l_ij = red_buf[0];
            __syncthreads();
            
            // Update online softmax state
            float m_i_old, l_i_old, m_new, alpha, beta, l_new;
            if (tid == 0) {
                m_i_old = m_l_sh[0];
                l_i_old = m_l_sh[1];
                m_new = fmaxf(m_i_old, m_ij);
                alpha = expf(m_i_old - m_new);
                beta = expf(m_ij - m_new);
                l_new = alpha * l_i_old + beta * l_ij;
                m_l_sh[0] = m_new;
                m_l_sh[1] = l_new;
                red_buf[1] = alpha;
                red_buf[2] = beta;
                red_buf[3] = l_i_old;
            }
            __syncthreads();
            
            alpha = red_buf[1];
            beta = red_buf[2];
            l_i_old = red_buf[3];
            l_new = m_l_sh[1];
            
            // Convert softmax result to FP16 for next WMMA
            for (int idx = tid; idx < TILE_J * TILE_K; idx += block_size) {
                p_tile_h[idx] = __float2half(p_tile_f[idx]);
            }
            __syncthreads();
            
            // =========================================================
            // PHASE 4: TENSOR CORE GEMM - Output Intermediate
            // M[j,d] = P[j,k] @ V2[k,d] = (TILE_J x TILE_K) @ (TILE_K x D)
            // =========================================================
            
            // Process D in chunks of WMMA_N=16, distribute across warps
            // Each warp handles one or more D-chunks
            const int d_chunks = D / WMMA_N;
            
            for (int d_chunk = warp_id; d_chunk < d_chunks; d_chunk += num_warps) {
                int d0 = d_chunk * WMMA_N;
                
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> p_frag;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> v2_frag;
                fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> m_frag;
                
                fill_fragment(m_frag, 0.0f);
                
                // P is (TILE_J x TILE_K) = (16 x 16), V2 chunk is (TILE_K x 16)
                // This is a single WMMA operation since TILE_K = WMMA_K = 16
                load_matrix_sync(p_frag, p_tile_h, TILE_K);
                load_matrix_sync(v2_frag, v2_tile_h + d0, D);
                mma_sync(m_frag, p_frag, v2_frag, m_frag);
                
                // Store M[0:16, d0:d0+16] to shared memory
                store_matrix_sync(m_tile_f + d0, m_frag, D, mem_row_major);
            }
            __syncthreads();
            
            // =========================================================
            // PHASE 5: Final Output - Element-wise V1 * M + Reduction
            // O[d] = Σ_j V1[j,d] * M[j,d]
            // =========================================================
            
            for (int d = tid; d < D; d += block_size) {
                float new_o_d = 0.0f;
                for (int jt = 0; jt < TILE_J; ++jt) {
                    if (j0 + jt < J) {
                        float v1_val = __half2float(v1_tile_h[jt * D + d]);
                        float m_val = m_tile_f[jt * D + d];
                        new_o_d += v1_val * m_val;
                    }
                }
                
                // Online softmax rescaling
                if (l_new > 1e-20f) {
                    o_sh[d] = (alpha * l_i_old * o_sh[d] + beta * new_o_d) / l_new;
                }
            }
            __syncthreads();
        }
    }
    
    // --- Write Final Output ---
    for (int d = tid; d < D; d += block_size) {
        Y[q_base_off + d] = o_sh[d];
    }
    
    // --- Write Softmax Stats ---
    if (tid == 0 && m_i_out != nullptr && l_i_out != nullptr) {
        int64_t stats_idx = ((int64_t)b * H + h) * I + i;
        m_i_out[stats_idx] = m_l_sh[0];
        l_i_out[stats_idx] = m_l_sh[1];
    }
}


// Multi-j warp-parallel gather: 4 output vectors per block, warp-shuffle softmax
template<int D_CONST>
__global__
void Yr_gather(
    const float4* __restrict__ R_query_f4,
    const float4* __restrict__ Q_f4,
    const float4* __restrict__ S_f4,
    const float4* __restrict__ V1_f4,
    const float4* __restrict__ V2_f4,
    float*       __restrict__ Y,
    float*       __restrict__ m_j_out,
    float*       __restrict__ l_j_out,
    int B, int H, int I, int J, int K, float scale)
{
    const int j_base = blockIdx.x * N_I_GATHER;
    const int h = blockIdx.y;
    const int b = blockIdx.z;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int block_size = blockDim.x;
    constexpr int DP = D_CONST + SMEM_PAD;

    extern __shared__ float smem[];
    float* q_vecs  = smem;
    float* i_tile  = q_vecs + N_I_GATHER * D_CONST;
    float* k_tile  = i_tile + TILE_I * DP;
    float* v1_tile = k_tile + TILE_K * DP;
    float* v2_tile = v1_tile + TILE_I * DP;
    float* p_tiles = v2_tile + TILE_K * DP;
    float* m_l_sh  = p_tiles + N_I_GATHER * TILE_I * TILE_K;
    float* o_sh    = m_l_sh + N_I_GATHER * 2;
    
    const int my_j = j_base + warp_id;
    const bool my_j_valid = (my_j < J);
    float* my_q = q_vecs + warp_id * D_CONST;
    float* my_p = p_tiles + warp_id * TILE_I * TILE_K;
    float* my_o = o_sh + warp_id * D_CONST;
    float* my_ml = m_l_sh + warp_id * 2;

    for (int n = 0; n < N_I_GATHER; n++) {
        int j_global = j_base + n;
        if (j_global < J) {
            const int64_t q_off = (((int64_t)b * H + h) * J + j_global) * D_CONST;
            for (int d = tid; d < D_CONST; d += block_size) {
                q_vecs[n * D_CONST + d] = ((const float*)R_query_f4)[q_off + d];
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

    for (int i0 = 0; i0 < I; i0 += TILE_I) {
        for (int k0 = 0; k0 < K; k0 += TILE_K) {
            
            for (int idx = tid; idx < TILE_I * D_CONST; idx += block_size) {
                int it = idx / D_CONST;
                int d = idx % D_CONST;
                int i_global = i0 + it;
                if (i_global < I) {
                    const int64_t i_off = (((int64_t)b * H + h) * I + i_global) * D_CONST + d;
                    i_tile[it * DP + d] = ((const float*)Q_f4)[i_off];
                    v1_tile[it * DP + d] = ((const float*)V1_f4)[i_off];
                }
            }
            
            for (int idx = tid; idx < TILE_K * D_CONST; idx += block_size) {
                int kt = idx / D_CONST;
                int d = idx % D_CONST;
                int k_global = k0 + kt;
                if (k_global < K) {
                    const int64_t k_off = (((int64_t)b * H + h) * K + k_global) * D_CONST + d;
                    k_tile[kt * DP + d] = ((const float*)S_f4)[k_off];
                    v2_tile[kt * DP + d] = ((const float*)V2_f4)[k_off];
                }
            }
            __syncthreads();

            #pragma unroll 1
            for (int cell = 0; cell < 8; cell++) {
                int cell_idx = lane_id + cell * 32;
                if (cell_idx < TILE_I * TILE_K) {
                    int it = cell_idx / TILE_K;
                    int kt = cell_idx % TILE_K;
                    
                    float dot = 0.0f;
                    if (my_j_valid && i0 + it < I && k0 + kt < K) {
                        float d_accum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                        #pragma unroll 4
                        for (int d = 0; d < D_CONST; d += 4) {
                            d_accum[0] += my_q[d+0] * i_tile[it*DP + d+0] * k_tile[kt*DP + d+0];
                            d_accum[1] += my_q[d+1] * i_tile[it*DP + d+1] * k_tile[kt*DP + d+1];
                            d_accum[2] += my_q[d+2] * i_tile[it*DP + d+2] * k_tile[kt*DP + d+2];
                            d_accum[3] += my_q[d+3] * i_tile[it*DP + d+3] * k_tile[kt*DP + d+3];
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
                if (cell_idx < TILE_I * TILE_K) {
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
                if (cell_idx < TILE_I * TILE_K) {
                    int it = cell_idx / TILE_K;
                    int kt = cell_idx % TILE_K;
                    if (my_j_valid && i0 + it < I && k0 + kt < K) {
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
            
            if (lane_id == 0) {
                my_ml[0] = m_new;
                my_ml[1] = l_new;
            }
            alpha = __shfl_sync(0xFFFFFFFF, alpha, 0);
            beta = __shfl_sync(0xFFFFFFFF, beta, 0);
            l_old = __shfl_sync(0xFFFFFFFF, l_old, 0);
            l_new = __shfl_sync(0xFFFFFFFF, l_new, 0);

            for (int d = lane_id; d < D_CONST; d += 32) {
                float new_o_d = 0.0f;
                if (my_j_valid) {
                    for (int it = 0; it < TILE_I; it++) {
                        if (i0 + it >= I) continue;
                        float v1_val = v1_tile[it * DP + d];
                        for (int kt = 0; kt < TILE_K; kt++) {
                            if (k0 + kt >= K) continue;
                            new_o_d += my_p[it * TILE_K + kt] * v1_val * v2_tile[kt * DP + d];
                        }
                    }
                }
                
                if (l_new > 1e-20f) {
                    my_o[d] = (alpha * l_old * my_o[d] + beta * new_o_d) / l_new;
                }
            }
            
            __syncthreads();
        }
    }

    for (int n = 0; n < N_I_GATHER; n++) {
        int j_global = j_base + n;
        if (j_global < J) {
            const int64_t out_off = (((int64_t)b * H + h) * J + j_global) * D_CONST;
            for (int d = tid; d < D_CONST; d += block_size) {
                Y[out_off + d] = o_sh[n * D_CONST + d];
            }
        }
    }

    if (lane_id == 0 && m_j_out != nullptr && l_j_out != nullptr && my_j_valid) {
        int64_t stats_idx = ((int64_t)b * H + h) * J + my_j;
        m_j_out[stats_idx] = my_ml[0];
        l_j_out[stats_idx] = my_ml[1];
    }
}


// Multi-k warp-parallel gather: 4 output vectors per block, warp-shuffle softmax
template<int D_CONST>
__global__
void Ys_gather(
    const float4* __restrict__ S_query_f4,
    const float4* __restrict__ Q_f4,
    const float4* __restrict__ R_f4,
    const float4* __restrict__ V1_f4,
    const float4* __restrict__ V2_f4,
    float*       __restrict__ Y,
    float*       __restrict__ m_k_out,
    float*       __restrict__ l_k_out,
    int B, int H, int I, int J, int K, float scale)
{
    const int k_base = blockIdx.x * N_I_GATHER;
    const int h = blockIdx.y;
    const int b = blockIdx.z;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int block_size = blockDim.x;
    constexpr int DP = D_CONST + SMEM_PAD;

    extern __shared__ float smem[];
    float* q_vecs  = smem;
    float* i_tile  = q_vecs + N_I_GATHER * D_CONST;
    float* j_tile  = i_tile + TILE_I * DP;
    float* v1_tile = j_tile + TILE_J * DP;
    float* v2_tile = v1_tile + TILE_I * DP;
    float* p_tiles = v2_tile + TILE_J * DP;
    float* m_l_sh  = p_tiles + N_I_GATHER * TILE_I * TILE_J;
    float* o_sh    = m_l_sh + N_I_GATHER * 2;
    
    const int my_k = k_base + warp_id;
    const bool my_k_valid = (my_k < K);
    float* my_q = q_vecs + warp_id * D_CONST;
    float* my_p = p_tiles + warp_id * TILE_I * TILE_J;
    float* my_o = o_sh + warp_id * D_CONST;
    float* my_ml = m_l_sh + warp_id * 2;

    for (int n = 0; n < N_I_GATHER; n++) {
        int k_global = k_base + n;
        if (k_global < K) {
            const int64_t q_off = (((int64_t)b * H + h) * K + k_global) * D_CONST;
            for (int d = tid; d < D_CONST; d += block_size) {
                q_vecs[n * D_CONST + d] = ((const float*)S_query_f4)[q_off + d];
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

    for (int i0 = 0; i0 < I; i0 += TILE_I) {
        for (int j0 = 0; j0 < J; j0 += TILE_J) {
            
            for (int idx = tid; idx < TILE_I * D_CONST; idx += block_size) {
                int it = idx / D_CONST;
                int d = idx % D_CONST;
                int i_global = i0 + it;
                if (i_global < I) {
                    const int64_t i_off = (((int64_t)b * H + h) * I + i_global) * D_CONST + d;
                    i_tile[it * DP + d] = ((const float*)Q_f4)[i_off];
                    v1_tile[it * DP + d] = ((const float*)V1_f4)[i_off];
                }
            }
            
            for (int idx = tid; idx < TILE_J * D_CONST; idx += block_size) {
                int jt = idx / D_CONST;
                int d = idx % D_CONST;
                int j_global = j0 + jt;
                if (j_global < J) {
                    const int64_t j_off = (((int64_t)b * H + h) * J + j_global) * D_CONST + d;
                    j_tile[jt * DP + d] = ((const float*)R_f4)[j_off];
                    v2_tile[jt * DP + d] = ((const float*)V2_f4)[j_off];
                }
            }
            __syncthreads();

            #pragma unroll 1
            for (int cell = 0; cell < 8; cell++) {
                int cell_idx = lane_id + cell * 32;
                if (cell_idx < TILE_I * TILE_J) {
                    int it = cell_idx / TILE_J;
                    int jt = cell_idx % TILE_J;
                    
                    float dot = 0.0f;
                    if (my_k_valid && i0 + it < I && j0 + jt < J) {
                        float d_accum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                        #pragma unroll 4
                        for (int d = 0; d < D_CONST; d += 4) {
                            d_accum[0] += my_q[d+0] * i_tile[it*DP + d+0] * j_tile[jt*DP + d+0];
                            d_accum[1] += my_q[d+1] * i_tile[it*DP + d+1] * j_tile[jt*DP + d+1];
                            d_accum[2] += my_q[d+2] * i_tile[it*DP + d+2] * j_tile[jt*DP + d+2];
                            d_accum[3] += my_q[d+3] * i_tile[it*DP + d+3] * j_tile[jt*DP + d+3];
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
                if (cell_idx < TILE_I * TILE_J) {
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
                if (cell_idx < TILE_I * TILE_J) {
                    int it = cell_idx / TILE_J;
                    int jt = cell_idx % TILE_J;
                    if (my_k_valid && i0 + it < I && j0 + jt < J) {
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
            
            if (lane_id == 0) {
                my_ml[0] = m_new;
                my_ml[1] = l_new;
            }
            alpha = __shfl_sync(0xFFFFFFFF, alpha, 0);
            beta = __shfl_sync(0xFFFFFFFF, beta, 0);
            l_old = __shfl_sync(0xFFFFFFFF, l_old, 0);
            l_new = __shfl_sync(0xFFFFFFFF, l_new, 0);

            for (int d = lane_id; d < D_CONST; d += 32) {
                float new_o_d = 0.0f;
                if (my_k_valid) {
                    for (int it = 0; it < TILE_I; it++) {
                        if (i0 + it >= I) continue;
                        float v1_val = v1_tile[it * DP + d];
                        for (int jt = 0; jt < TILE_J; jt++) {
                            if (j0 + jt >= J) continue;
                            new_o_d += my_p[it * TILE_J + jt] * v1_val * v2_tile[jt * DP + d];
                        }
                    }
                }
                
                if (l_new > 1e-20f) {
                    my_o[d] = (alpha * l_old * my_o[d] + beta * new_o_d) / l_new;
                }
            }
            
            __syncthreads();
        }
    }

    for (int n = 0; n < N_I_GATHER; n++) {
        int k_global = k_base + n;
        if (k_global < K) {
            const int64_t out_off = (((int64_t)b * H + h) * K + k_global) * D_CONST;
            for (int d = tid; d < D_CONST; d += block_size) {
                Y[out_off + d] = o_sh[n * D_CONST + d];
            }
        }
    }

    if (lane_id == 0 && m_k_out != nullptr && l_k_out != nullptr && my_k_valid) {
        int64_t stats_idx = ((int64_t)b * H + h) * K + my_k;
        m_k_out[stats_idx] = my_ml[0];
        l_k_out[stats_idx] = my_ml[1];
    }
}




// =============================================================================
// Scatter Kernels
// =============================================================================

/**
 * @brief Computes Y_q_ = sum_{j,k} Ar[i,j,k] * As[i,j,k] * Vr_2[j,:] * Vs_2[k,:]
 *
 * Uses parallel outer-product approach for computing 3-way attention scores.
 * Each block processes a tile of I outputs, iterating over all J and K tiles.
 * This eliminates atomic operations by ensuring each output is owned by exactly
 * one block.
 *
 * Grid:  (ceil(I/TILE_I), H, B)
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
void Yq_scatter(
    const float* __restrict__ Q,
    const float* __restrict__ R,
    const float* __restrict__ S,
    const float* __restrict__ Vr_2,
    const float* __restrict__ Vs_2,
    const float* __restrict__ m_j_in,
    const float* __restrict__ l_j_in,
    const float* __restrict__ m_k_in,
    const float* __restrict__ l_k_in,
    float* __restrict__ Y_q_,
    int B, int H, int I, int J, int K, float scale
) {
    // --- Grid/Block Mapping ---
    const int b = blockIdx.z;
    const int h = blockIdx.y;
    const int i_start = blockIdx.x * TILE_I;

    const int tid = threadIdx.x;
    const int tpb = blockDim.x;

    // --- Memory Offsets ---
    const int64_t q_bh_offset = (int64_t)(b * H + h) * I * D_CONST;
    const int64_t r_bh_offset = (int64_t)(b * H + h) * J * D_CONST;
    const int64_t s_bh_offset = (int64_t)(b * H + h) * K * D_CONST;
    const int64_t mj_bh_offset = (int64_t)(b * H + h) * J;
    const int64_t mk_bh_offset = (int64_t)(b * H + h) * K;

    // --- Shared Memory Layout ---
    extern __shared__ float smem[];
    float* q_tile = smem;
    float* r_tile = q_tile + TILE_I * D_CONST;
    float* s_tile = r_tile + TILE_J * D_CONST;
    float* vr_tile = s_tile + TILE_K * D_CONST;
    float* vs_tile = vr_tile + TILE_J * D_CONST;
    float* attn_tile = vs_tile + TILE_K * D_CONST;
    float* mj_tile = attn_tile + TILE_I * TILE_J * TILE_K;
    float* lj_tile = mj_tile + TILE_J;
    float* mk_tile = lj_tile + TILE_J;
    float* lk_tile = mk_tile + TILE_K;

    // --- Thread Indexing for Cooperative Loads ---
    constexpr int i_load_D = D_CONST;  // compile-time for efficient div/mod
    const int i_load = tid / i_load_D;
    const int d_load = tid % i_load_D;
    constexpr int load_iters = (TILE_I * D_CONST + 256 - 1) / 256;
    constexpr int load_step = 256 / D_CONST;

    // --- Per-Thread Output Accumulators (registers) ---
    float yq_acc[8];
    #pragma unroll
    for (int n = 0; n < 8; n++) {
        yq_acc[n] = 0.0f;
    }

    // --- Load Q Tile (fixed for entire block) ---
    for (int n = 0; n < TILE_I; n += load_step) {
        int i_global = i_start + n + i_load;
        if (n + i_load < TILE_I && i_global < I) {
            q_tile[(n + i_load) * D_CONST + d_load] = Q[q_bh_offset + i_global * D_CONST + d_load];
        }
    }
    __syncthreads();

    // --- Main Loop: Iterate Over J Tiles ---
    for (int jt = 0; jt < J; jt += TILE_J) {
        for (int n = 0; n < TILE_J; n += load_step) {
            int j_global = jt + n + i_load;
            if (n + i_load < TILE_J && j_global < J) {
                r_tile[(n + i_load) * D_CONST + d_load] = R[r_bh_offset + j_global * D_CONST + d_load];
            }
        }
        for (int n = 0; n < TILE_J; n += load_step) {
            int j_global = jt + n + i_load;
            if (n + i_load < TILE_J && j_global < J) {
                vr_tile[(n + i_load) * D_CONST + d_load] = Vr_2[r_bh_offset + j_global * D_CONST + d_load];
            }
        }
        if (tid < TILE_J && jt + tid < J) {
            mj_tile[tid] = m_j_in[mj_bh_offset + jt + tid];
            lj_tile[tid] = 1.0f / l_j_in[mj_bh_offset + jt + tid];
        }

        // --- Inner Loop: Iterate Over K Tiles ---
        for (int kt = 0; kt < K; kt += TILE_K) {
            for (int n = 0; n < TILE_K; n += load_step) {
                int k_global = kt + n + i_load;
                if (n + i_load < TILE_K && k_global < K) {
                    s_tile[(n + i_load) * D_CONST + d_load] = S[s_bh_offset + k_global * D_CONST + d_load];
                }
            }
            if (tid < TILE_K && kt + tid < K) {
                mk_tile[tid] = m_k_in[mk_bh_offset + kt + tid];
                lk_tile[tid] = 1.0f / l_k_in[mk_bh_offset + kt + tid];
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

            float qa[4], ra[4], sa[4];
            constexpr int d_per_slice = D_CONST / 4;

            for (int db = 0; db < d_per_slice; db++) {
                const int d_idx = da * d_per_slice + db;
                #pragma unroll
                for (int u = 0; u < 4; u++) {
                    qa[u] = q_tile[(ia * 4 + u) * D_CONST + d_idx];
                    ra[u] = r_tile[(ja * 4 + u) * D_CONST + d_idx];
                    sa[u] = s_tile[(ka * 4 + u) * D_CONST + d_idx];
                }
                #pragma unroll
                for (int i0 = 0; i0 < 4; i0++) {
                    #pragma unroll
                    for (int i1 = 0; i1 < 4; i1++) {
                        #pragma unroll
                        for (int i2 = 0; i2 < 4; i2++) {
                            acc[i0][i1][i2] += qa[i0] * ra[i1] * sa[i2];
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
                                attn_tile[(ia * 4 + i0) * TILE_J * TILE_K +
                                          (ja * 4 + i1) * TILE_K +
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
                                acc[i0][i1][i2] += attn_tile[(ia * 4 + i0) * TILE_J * TILE_K +
                                                             (ja * 4 + i1) * TILE_K +
                                                             (ka * 4 + i2)];
                            }
                        }
                    }
                }
                __syncthreads();
            }

            // --- Apply Softmax Scaling ---
            if (da == 0) {
                #pragma unroll
                for (int i0 = 0; i0 < 4; i0++) {
                    #pragma unroll
                    for (int i1 = 0; i1 < 4; i1++) {
                        float mjt = mj_tile[ja * 4 + i1];
                        float ljt = lj_tile[ja * 4 + i1];
                        #pragma unroll
                        for (int i2 = 0; i2 < 4; i2++) {
                            float logit = acc[i0][i1][i2] * scale;
                            float ar = expf(logit - mjt) * ljt;
                            float as = expf(logit - mk_tile[ka * 4 + i2]) * lk_tile[ka * 4 + i2];
                            attn_tile[(ia * 4 + i0) * TILE_J * TILE_K +
                                      (ja * 4 + i1) * TILE_K +
                                      (ka * 4 + i2)] = ar * as;
                        }
                    }
                }
            }
            __syncthreads();

            for (int n = 0; n < TILE_K; n += load_step) {
                int k_global = kt + n + i_load;
                if (n + i_load < TILE_K && k_global < K) {
                    vs_tile[(n + i_load) * D_CONST + d_load] = Vs_2[s_bh_offset + k_global * D_CONST + d_load];
                }
            }
            __syncthreads();

            // --- Accumulate Output ---
            for (int n = 0; n < load_iters; n++) {
                int tid_n = tid + n * tpb;
                if (tid_n < TILE_I * D_CONST) {
                    int iy = tid_n / D_CONST;
                    int dy = tid_n % D_CONST;

                    float f = 0.0f;
                    for (int jy = 0; jy < TILE_J; jy++) {
                        float vrt = vr_tile[jy * D_CONST + dy];
                        for (int ky = 0; ky < TILE_K; ky++) {
                            f += attn_tile[iy * TILE_J * TILE_K + jy * TILE_K + ky] * vrt * vs_tile[ky * D_CONST + dy];
                        }
                    }
                    yq_acc[n] += f;
                }
            }
            __syncthreads();
        }
    }

    // --- Write Output to Global Memory ---
    for (int n = 0; n < load_iters; n++) {
        int tid_n = tid + n * tpb;
        if (tid_n < TILE_I * D_CONST) {
            int iy = tid_n / D_CONST;
            int dy = tid_n % D_CONST;
            int i_global = i_start + iy;
            if (i_global < I) {
                Y_q_[q_bh_offset + i_global * D_CONST + dy] = yq_acc[n];
            }
        }
    }
}

template<int D_CONST>
void Yq_scatter_launcher_with_stats(
    const at::Tensor& Q, const at::Tensor& R, const at::Tensor& S,
    const at::Tensor& Vr_2, const at::Tensor& Vs_2,
    const at::Tensor& m_j, const at::Tensor& l_j,
    const at::Tensor& m_k, const at::Tensor& l_k,
    at::Tensor& Y_q_, float scale,
    cudaStream_t stream = 0
) {
    const auto B = Q.size(0);
    const auto H = Q.size(1);
    const auto I = Q.size(2);
    const auto J = R.size(2);
    const auto K = S.size(2);

    TORCH_CHECK(I == J && J == K, "Yq_scatter requires I == J == K");
    TORCH_CHECK(I % TILE_I == 0, "Yq_scatter requires I to be a multiple of TILE_I (16)");
    TORCH_CHECK(I >= TILE_I, "Yq_scatter requires I >= TILE_I (16)");

    const int TPB = 256;
    dim3 grid((I + TILE_I - 1) / TILE_I, H, B);
    dim3 block(TPB);

    size_t smem_size = sizeof(float) * (
        TILE_I * D_CONST +              // q_tile
        TILE_J * D_CONST +              // r_tile
        TILE_K * D_CONST +              // s_tile
        TILE_J * D_CONST +              // vr_tile
        TILE_K * D_CONST +              // vs_tile
        TILE_I * TILE_J * TILE_K +      // attn_tile
        TILE_J + TILE_J +               // mj_tile, lj_tile
        TILE_K + TILE_K                 // mk_tile, lk_tile
    );

    Yq_scatter<D_CONST><<<grid, block, smem_size, stream>>>(
        Q.data_ptr<float>(), R.data_ptr<float>(), S.data_ptr<float>(),
        Vr_2.data_ptr<float>(), Vs_2.data_ptr<float>(),
        m_j.data_ptr<float>(), l_j.data_ptr<float>(),
        m_k.data_ptr<float>(), l_k.data_ptr<float>(),
        Y_q_.data_ptr<float>(),
        B, H, I, J, K, scale
    );
}


/**
 * @brief Computes Y_r_ = sum_{i,k} Ar[i,j,k] * As[i,j,k] * Vq_2[i,:] * Vs_2[k,:]
 *
 * Uses parallel outer-product approach for computing 3-way attention scores.
 * Each block processes a tile of J outputs, iterating over all I and K tiles.
 * This eliminates atomic operations by ensuring each output is owned by exactly
 * one block.
 *
 * Grid:  (ceil(J/TILE_J), H, B)
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
void Yr_scatter(
    const float* __restrict__ Q,
    const float* __restrict__ R,
    const float* __restrict__ S,
    const float* __restrict__ Vq_2,
    const float* __restrict__ Vs_2,
    const float* __restrict__ m_i_in,
    const float* __restrict__ l_i_in,
    const float* __restrict__ m_k_in,
    const float* __restrict__ l_k_in,
    float* __restrict__ Y_r_,
    int B, int H, int I, int J, int K, float scale
) {
    // --- Grid/Block Mapping ---
    const int b = blockIdx.z;
    const int h = blockIdx.y;
    const int j_start = blockIdx.x * TILE_J;

    const int tid = threadIdx.x;
    const int tpb = blockDim.x;

    // --- Memory Offsets ---
    const int64_t r_bh_offset = (int64_t)(b * H + h) * J * D_CONST;
    const int64_t q_bh_offset = (int64_t)(b * H + h) * I * D_CONST;
    const int64_t s_bh_offset = (int64_t)(b * H + h) * K * D_CONST;
    const int64_t mi_bh_offset = (int64_t)(b * H + h) * I;
    const int64_t mk_bh_offset = (int64_t)(b * H + h) * K;

    // --- Shared Memory Layout ---
    extern __shared__ float smem[];
    float* r_tile = smem;
    float* q_tile = r_tile + TILE_J * D_CONST;
    float* s_tile = q_tile + TILE_I * D_CONST;
    float* vq_tile = s_tile + TILE_K * D_CONST;
    float* vs_tile = vq_tile + TILE_I * D_CONST;
    float* attn_tile = vs_tile + TILE_K * D_CONST;
    float* mi_tile = attn_tile + TILE_I * TILE_J * TILE_K;
    float* li_tile = mi_tile + TILE_I;
    float* mk_tile = li_tile + TILE_I;
    float* lk_tile = mk_tile + TILE_K;

    // --- Thread Indexing for Cooperative Loads ---
    const int j_load = tid / D_CONST;
    const int d_load = tid % D_CONST;
    constexpr int load_iters = (TILE_J * D_CONST + 256 - 1) / 256;
    constexpr int load_step = 256 / D_CONST;

    // --- Per-Thread Output Accumulators (registers) ---
    float yr_acc[8];
    #pragma unroll
    for (int n = 0; n < 8; n++) {
        yr_acc[n] = 0.0f;
    }

    // --- Load R Tile (fixed for entire block) ---
    for (int n = 0; n < TILE_J; n += load_step) {
        int j_global = j_start + n + j_load;
        if (n + j_load < TILE_J && j_global < J) {
            r_tile[(n + j_load) * D_CONST + d_load] = R[r_bh_offset + j_global * D_CONST + d_load];
        }
    }
    __syncthreads();

    // --- Main Loop: Iterate Over I Tiles ---
    for (int it = 0; it < I; it += TILE_I) {
        for (int n = 0; n < TILE_I; n += load_step) {
            int i_global = it + n + j_load;
            if (n + j_load < TILE_I && i_global < I) {
                q_tile[(n + j_load) * D_CONST + d_load] = Q[q_bh_offset + i_global * D_CONST + d_load];
            }
        }
        for (int n = 0; n < TILE_I; n += load_step) {
            int i_global = it + n + j_load;
            if (n + j_load < TILE_I && i_global < I) {
                vq_tile[(n + j_load) * D_CONST + d_load] = Vq_2[q_bh_offset + i_global * D_CONST + d_load];
            }
        }
        if (tid < TILE_I && it + tid < I) {
            mi_tile[tid] = m_i_in[mi_bh_offset + it + tid];
            li_tile[tid] = 1.0f / l_i_in[mi_bh_offset + it + tid];
        }

        // --- Inner Loop: Iterate Over K Tiles ---
        for (int kt = 0; kt < K; kt += TILE_K) {
            for (int n = 0; n < TILE_K; n += load_step) {
                int k_global = kt + n + j_load;
                if (n + j_load < TILE_K && k_global < K) {
                    s_tile[(n + j_load) * D_CONST + d_load] = S[s_bh_offset + k_global * D_CONST + d_load];
                }
            }
            if (tid < TILE_K && kt + tid < K) {
                mk_tile[tid] = m_k_in[mk_bh_offset + kt + tid];
                lk_tile[tid] = 1.0f / l_k_in[mk_bh_offset + kt + tid];
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

            float qa[4], ra[4], sa[4];
            constexpr int d_per_slice = D_CONST / 4;

            for (int db = 0; db < d_per_slice; db++) {
                const int d_idx = da * d_per_slice + db;
                #pragma unroll
                for (int u = 0; u < 4; u++) {
                    qa[u] = q_tile[(ia * 4 + u) * D_CONST + d_idx];
                    ra[u] = r_tile[(ja * 4 + u) * D_CONST + d_idx];
                    sa[u] = s_tile[(ka * 4 + u) * D_CONST + d_idx];
                }
                #pragma unroll
                for (int i0 = 0; i0 < 4; i0++) {
                    #pragma unroll
                    for (int i1 = 0; i1 < 4; i1++) {
                        #pragma unroll
                        for (int i2 = 0; i2 < 4; i2++) {
                            acc[i0][i1][i2] += qa[i0] * ra[i1] * sa[i2];
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
                                attn_tile[(ia * 4 + i0) * TILE_J * TILE_K +
                                          (ja * 4 + i1) * TILE_K +
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
                                acc[i0][i1][i2] += attn_tile[(ia * 4 + i0) * TILE_J * TILE_K +
                                                             (ja * 4 + i1) * TILE_K +
                                                             (ka * 4 + i2)];
                            }
                        }
                    }
                }
                __syncthreads();
            }

            // --- Apply Softmax Scaling ---
            if (da == 0) {
                #pragma unroll
                for (int i0 = 0; i0 < 4; i0++) {
                    float mit = mi_tile[ia * 4 + i0];
                    float lit = li_tile[ia * 4 + i0];
                    #pragma unroll
                    for (int i1 = 0; i1 < 4; i1++) {
                        #pragma unroll
                        for (int i2 = 0; i2 < 4; i2++) {
                            float logit = acc[i0][i1][i2] * scale;
                            float ai = expf(logit - mit) * lit;
                            float as = expf(logit - mk_tile[ka * 4 + i2]) * lk_tile[ka * 4 + i2];
                            attn_tile[(ia * 4 + i0) * TILE_J * TILE_K +
                                      (ja * 4 + i1) * TILE_K +
                                      (ka * 4 + i2)] = ai * as;
                        }
                    }
                }
            }
            __syncthreads();

            for (int n = 0; n < TILE_K; n += load_step) {
                int k_global = kt + n + j_load;
                if (n + j_load < TILE_K && k_global < K) {
                    vs_tile[(n + j_load) * D_CONST + d_load] = Vs_2[s_bh_offset + k_global * D_CONST + d_load];
                }
            }
            __syncthreads();

            // --- Accumulate Output ---
            for (int n = 0; n < load_iters; n++) {
                int tid_n = tid + n * tpb;
                if (tid_n < TILE_J * D_CONST) {
                    int jy = tid_n / D_CONST;
                    int dy = tid_n % D_CONST;

                    float f = 0.0f;
                    for (int iy = 0; iy < TILE_I; iy++) {
                        float vqt = vq_tile[iy * D_CONST + dy];
                        for (int ky = 0; ky < TILE_K; ky++) {
                            f += attn_tile[iy * TILE_J * TILE_K + jy * TILE_K + ky] * vqt * vs_tile[ky * D_CONST + dy];
                        }
                    }
                    yr_acc[n] += f;
                }
            }
            __syncthreads();
        }
    }

    // --- Write Output to Global Memory ---
    for (int n = 0; n < load_iters; n++) {
        int tid_n = tid + n * tpb;
        if (tid_n < TILE_J * D_CONST) {
            int jy = tid_n / D_CONST;
            int dy = tid_n % D_CONST;
            int j_global = j_start + jy;
            if (j_global < J) {
                Y_r_[r_bh_offset + j_global * D_CONST + dy] = yr_acc[n];
            }
        }
    }
}

template<int D_CONST>
void Yr_scatter_launcher_with_stats(
    const at::Tensor& Q, const at::Tensor& R, const at::Tensor& S,
    const at::Tensor& Vq_2, const at::Tensor& Vs_2,
    const at::Tensor& m_i, const at::Tensor& l_i,
    const at::Tensor& m_k, const at::Tensor& l_k,
    at::Tensor& Y_r_, float scale,
    cudaStream_t stream = 0
) {
    const auto B = Q.size(0);
    const auto H = Q.size(1);
    const auto I = Q.size(2);
    const auto J = R.size(2);
    const auto K = S.size(2);

    TORCH_CHECK(I == J && J == K, "Yr_scatter requires I == J == K");
    TORCH_CHECK(J % TILE_J == 0, "Yr_scatter requires J to be a multiple of TILE_J (16)");
    TORCH_CHECK(J >= TILE_J, "Yr_scatter requires J >= TILE_J (16)");

    const int TPB = 256;
    dim3 grid((J + TILE_J - 1) / TILE_J, H, B);
    dim3 block(TPB);

    size_t smem_size = sizeof(float) * (
        TILE_J * D_CONST +              // r_tile
        TILE_I * D_CONST +              // q_tile
        TILE_K * D_CONST +              // s_tile
        TILE_I * D_CONST +              // vq_tile
        TILE_K * D_CONST +              // vs_tile
        TILE_I * TILE_J * TILE_K +      // attn_tile
        TILE_I + TILE_I +               // mi_tile, li_tile
        TILE_K + TILE_K                 // mk_tile, lk_tile
    );

    Yr_scatter<D_CONST><<<grid, block, smem_size, stream>>>(
        Q.data_ptr<float>(), R.data_ptr<float>(), S.data_ptr<float>(),
        Vq_2.data_ptr<float>(), Vs_2.data_ptr<float>(),
        m_i.data_ptr<float>(), l_i.data_ptr<float>(),
        m_k.data_ptr<float>(), l_k.data_ptr<float>(),
        Y_r_.data_ptr<float>(),
        B, H, I, J, K, scale
    );
}


/**
 * @brief Computes Y_s_ = sum_{i,j} Ar[i,j,k] * As[i,j,k] * Vq_2[i,:] * Vr_2[j,:]
 *
 * Uses parallel outer-product approach for computing 3-way attention scores.
 * Each block processes a tile of K outputs, iterating over all I and J tiles.
 * This eliminates atomic operations by ensuring each output is owned by exactly
 * one block.
 *
 * Grid:  (ceil(K/TILE_K), H, B)
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
void Ys_scatter(
    const float* __restrict__ Q,
    const float* __restrict__ R,
    const float* __restrict__ S,
    const float* __restrict__ Vq_2,
    const float* __restrict__ Vr_2,
    const float* __restrict__ m_i_in,
    const float* __restrict__ l_i_in,
    const float* __restrict__ m_j_in,
    const float* __restrict__ l_j_in,
    float* __restrict__ Y_s_,
    int B, int H, int I, int J, int K, float scale
) {
    // --- Grid/Block Mapping ---
    const int b = blockIdx.z;
    const int h = blockIdx.y;
    const int k_start = blockIdx.x * TILE_K;

    const int tid = threadIdx.x;
    const int tpb = blockDim.x;

    // --- Memory Offsets ---
    const int64_t s_bh_offset = (int64_t)(b * H + h) * K * D_CONST;
    const int64_t q_bh_offset = (int64_t)(b * H + h) * I * D_CONST;
    const int64_t r_bh_offset = (int64_t)(b * H + h) * J * D_CONST;
    const int64_t mi_bh_offset = (int64_t)(b * H + h) * I;
    const int64_t mj_bh_offset = (int64_t)(b * H + h) * J;

    // --- Shared Memory Layout ---
    extern __shared__ float smem[];
    float* s_tile = smem;
    float* q_tile = s_tile + TILE_K * D_CONST;
    float* r_tile = q_tile + TILE_I * D_CONST;
    float* vq_tile = r_tile + TILE_J * D_CONST;
    float* vr_tile = vq_tile + TILE_I * D_CONST;
    float* attn_tile = vr_tile + TILE_J * D_CONST;
    float* mi_tile = attn_tile + TILE_I * TILE_J * TILE_K;
    float* li_tile = mi_tile + TILE_I;
    float* mj_tile = li_tile + TILE_I;
    float* lj_tile = mj_tile + TILE_J;

    // --- Thread Indexing for Cooperative Loads ---
    const int k_load = tid / D_CONST;
    const int d_load = tid % D_CONST;
    constexpr int load_iters = (TILE_K * D_CONST + 256 - 1) / 256;
    constexpr int load_step = 256 / D_CONST;

    // --- Per-Thread Output Accumulators (registers) ---
    float ys_acc[8];
    #pragma unroll
    for (int n = 0; n < 8; n++) {
        ys_acc[n] = 0.0f;
    }

    // --- Load S Tile (fixed for entire block) ---
    for (int n = 0; n < TILE_K; n += load_step) {
        int k_global = k_start + n + k_load;
        if (n + k_load < TILE_K && k_global < K) {
            s_tile[(n + k_load) * D_CONST + d_load] = S[s_bh_offset + k_global * D_CONST + d_load];
        }
    }
    __syncthreads();

    // --- Main Loop: Iterate Over I Tiles ---
    for (int it = 0; it < I; it += TILE_I) {
        for (int n = 0; n < TILE_I; n += load_step) {
            int i_global = it + n + k_load;
            if (n + k_load < TILE_I && i_global < I) {
                q_tile[(n + k_load) * D_CONST + d_load] = Q[q_bh_offset + i_global * D_CONST + d_load];
            }
        }
        for (int n = 0; n < TILE_I; n += load_step) {
            int i_global = it + n + k_load;
            if (n + k_load < TILE_I && i_global < I) {
                vq_tile[(n + k_load) * D_CONST + d_load] = Vq_2[q_bh_offset + i_global * D_CONST + d_load];
            }
        }
        if (tid < TILE_I && it + tid < I) {
            mi_tile[tid] = m_i_in[mi_bh_offset + it + tid];
            li_tile[tid] = 1.0f / l_i_in[mi_bh_offset + it + tid];
        }

        // --- Inner Loop: Iterate Over J Tiles ---
        for (int jt = 0; jt < J; jt += TILE_J) {
            for (int n = 0; n < TILE_J; n += load_step) {
                int j_global = jt + n + k_load;
                if (n + k_load < TILE_J && j_global < J) {
                    r_tile[(n + k_load) * D_CONST + d_load] = R[r_bh_offset + j_global * D_CONST + d_load];
                }
            }
            for (int n = 0; n < TILE_J; n += load_step) {
                int j_global = jt + n + k_load;
                if (n + k_load < TILE_J && j_global < J) {
                    vr_tile[(n + k_load) * D_CONST + d_load] = Vr_2[r_bh_offset + j_global * D_CONST + d_load];
                }
            }
            if (tid < TILE_J && jt + tid < J) {
                mj_tile[tid] = m_j_in[mj_bh_offset + jt + tid];
                lj_tile[tid] = 1.0f / l_j_in[mj_bh_offset + jt + tid];
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

            float qa[4], ra[4], sa[4];
            constexpr int d_per_slice = D_CONST / 4;

            for (int db = 0; db < d_per_slice; db++) {
                const int d_idx = da * d_per_slice + db;
                #pragma unroll
                for (int u = 0; u < 4; u++) {
                    qa[u] = q_tile[(ia * 4 + u) * D_CONST + d_idx];
                    ra[u] = r_tile[(ja * 4 + u) * D_CONST + d_idx];
                    sa[u] = s_tile[(ka * 4 + u) * D_CONST + d_idx];
                }
                #pragma unroll
                for (int i0 = 0; i0 < 4; i0++) {
                    #pragma unroll
                    for (int i1 = 0; i1 < 4; i1++) {
                        #pragma unroll
                        for (int i2 = 0; i2 < 4; i2++) {
                            acc[i0][i1][i2] += qa[i0] * ra[i1] * sa[i2];
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
                                attn_tile[(ia * 4 + i0) * TILE_J * TILE_K +
                                          (ja * 4 + i1) * TILE_K +
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
                                acc[i0][i1][i2] += attn_tile[(ia * 4 + i0) * TILE_J * TILE_K +
                                                             (ja * 4 + i1) * TILE_K +
                                                             (ka * 4 + i2)];
                            }
                        }
                    }
                }
                __syncthreads();
            }

            // --- Apply Softmax Scaling ---
            if (da == 0) {
                #pragma unroll
                for (int i0 = 0; i0 < 4; i0++) {
                    float mit = mi_tile[ia * 4 + i0];
                    float lit = li_tile[ia * 4 + i0];
                    #pragma unroll
                    for (int i1 = 0; i1 < 4; i1++) {
                        float mjt = mj_tile[ja * 4 + i1];
                        float ljt = lj_tile[ja * 4 + i1];
                        #pragma unroll
                        for (int i2 = 0; i2 < 4; i2++) {
                            float logit = acc[i0][i1][i2] * scale;
                            float ai = expf(logit - mit) * lit;
                            float aj = expf(logit - mjt) * ljt;
                            attn_tile[(ia * 4 + i0) * TILE_J * TILE_K +
                                      (ja * 4 + i1) * TILE_K +
                                      (ka * 4 + i2)] = ai * aj;
                        }
                    }
                }
            }
            __syncthreads();

            // --- Accumulate Output ---
            for (int n = 0; n < load_iters; n++) {
                int tid_n = tid + n * tpb;
                if (tid_n < TILE_K * D_CONST) {
                    int ky = tid_n / D_CONST;
                    int dy = tid_n % D_CONST;

                    float f = 0.0f;
                    for (int iy = 0; iy < TILE_I; iy++) {
                        float vqt = vq_tile[iy * D_CONST + dy];
                        for (int jy = 0; jy < TILE_J; jy++) {
                            f += attn_tile[iy * TILE_J * TILE_K + jy * TILE_K + ky] * vqt * vr_tile[jy * D_CONST + dy];
                        }
                    }
                    ys_acc[n] += f;
                }
            }
            __syncthreads();
        }
    }

    // --- Write Output to Global Memory ---
    for (int n = 0; n < load_iters; n++) {
        int tid_n = tid + n * tpb;
        if (tid_n < TILE_K * D_CONST) {
            int ky = tid_n / D_CONST;
            int dy = tid_n % D_CONST;
            int k_global = k_start + ky;
            if (k_global < K) {
                Y_s_[s_bh_offset + k_global * D_CONST + dy] = ys_acc[n];
            }
        }
    }
}


template<int D_CONST>
void Ys_scatter_launcher_with_stats(
    const at::Tensor& Q, const at::Tensor& R, const at::Tensor& S,
    const at::Tensor& Vq_2, const at::Tensor& Vr_2,
    const at::Tensor& m_i, const at::Tensor& l_i,
    const at::Tensor& m_j, const at::Tensor& l_j,
    at::Tensor& Y_s_, float scale,
    cudaStream_t stream = 0
) {
    const auto B = Q.size(0);
    const auto H = Q.size(1);
    const auto I = Q.size(2);
    const auto J = R.size(2);
    const auto K = S.size(2);

    TORCH_CHECK(I == J && J == K, "Ys_scatter requires I == J == K");
    TORCH_CHECK(K % TILE_K == 0, "Ys_scatter requires K to be a multiple of TILE_K (16)");
    TORCH_CHECK(K >= TILE_K, "Ys_scatter requires K >= TILE_K (16)");

    const int TPB = 256;
    dim3 grid((K + TILE_K - 1) / TILE_K, H, B);
    dim3 block(TPB);

    size_t smem_size = sizeof(float) * (
        TILE_K * D_CONST +              // s_tile
        TILE_I * D_CONST +              // q_tile
        TILE_J * D_CONST +              // r_tile
        TILE_I * D_CONST +              // vq_tile
        TILE_J * D_CONST +              // vr_tile
        TILE_I * TILE_J * TILE_K +      // attn_tile
        TILE_I + TILE_I +               // mi_tile, li_tile
        TILE_J + TILE_J                 // mj_tile, lj_tile
    );

    Ys_scatter<D_CONST><<<grid, block, smem_size, stream>>>(
        Q.data_ptr<float>(), R.data_ptr<float>(), S.data_ptr<float>(),
        Vq_2.data_ptr<float>(), Vr_2.data_ptr<float>(),
        m_i.data_ptr<float>(), l_i.data_ptr<float>(),
        m_j.data_ptr<float>(), l_j.data_ptr<float>(),
        Y_s_.data_ptr<float>(),
        B, H, I, J, K, scale
    );
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
           at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> forward_cuda(
    at::Tensor Q, at::Tensor R, at::Tensor S,
    at::Tensor Vq_1, at::Tensor Vq_2,
    at::Tensor Vr_1, at::Tensor Vr_2,
    at::Tensor Vs_1, at::Tensor Vs_2,
    double dropout_rate) {
    Q = Q.contiguous();  
    R = R.contiguous();  
    S = S.contiguous();
    Vq_1 = Vq_1.contiguous();
    Vq_2 = Vq_2.contiguous();
    Vr_1 = Vr_1.contiguous();
    Vr_2 = Vr_2.contiguous();
    Vs_1 = Vs_1.contiguous();
    Vs_2 = Vs_2.contiguous();

    const auto B = Q.size(0);
    const auto H = Q.size(1);
    const auto I = Q.size(2);
    const auto J = R.size(2);
    const auto K = S.size(2);
    const auto D = Q.size(3);

    const float scale = 1.0f / sqrtf((float)D); 

    // allocate outputs on GPU
    auto opts = Q.options();
    auto Y_q  = torch::zeros({B,H,I,D}, opts);
    auto Y_r  = torch::zeros({B,H,J,D}, opts);
    auto Y_s  = torch::zeros({B,H,K,D}, opts);
    auto Y_q_ = torch::zeros({B,H,I,D}, opts); 
    auto Y_r_ = torch::zeros({B,H,J,D}, opts);
    auto Y_s_ = torch::zeros({B,H,K,D}, opts);
    
    // Allocate softmax stats tensors - gather kernels will populate these
    // Stats are computed during gather and reused by scatter kernels + backward pass
    auto m_i = torch::zeros({B,H,I}, opts);
    auto l_i = torch::zeros({B,H,I}, opts);
    auto m_j = torch::zeros({B,H,J}, opts);
    auto l_j = torch::zeros({B,H,J}, opts);
    auto m_k = torch::zeros({B,H,K}, opts);
    auto l_k = torch::zeros({B,H,K}, opts);
    
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

    TORCH_CHECK(Q.scalar_type() == at::kFloat, "Only float32 is supported.");
    TORCH_CHECK(D % 4 == 0, "D must be multiple of 4 for FP32 float4 path.");
    TORCH_CHECK(D == 16 || D == 32 || D == 64, "forward: unsupported D=", D, ". Supported: 16, 32, 64");

    // =============================================================================
    // D-dispatch: routes to D_TMPL=16, 32 or 64 template instantiation
    // =============================================================================
    #define FWD_DISPATCH_D(D_VAL, ...) \
      [&] { \
        if ((D_VAL) == 16)      { constexpr int D_TMPL = 16; __VA_ARGS__; } \
        else if ((D_VAL) == 32) { constexpr int D_TMPL = 32; __VA_ARGS__; } \
        else if ((D_VAL) == 64) { constexpr int D_TMPL = 64; __VA_ARGS__; } \
      }()

    FWD_DISPATCH_D(D, {
    // GATHER: Y_q on stream 0
    {
        dim3 grid_yq((I + N_I_GATHER - 1) / N_I_GATHER, H, B);
        
        {
            constexpr int DP = D_TMPL + SMEM_PAD;
            size_t smem_size = sizeof(float) * (
                N_I_GATHER * D_TMPL +         // q_vecs
                TILE_J * DP +                 // r_tile (padded)
                TILE_K * DP +                 // s_tile (padded)
                TILE_J * DP +                 // v1_tile (padded)
                TILE_K * DP +                 // v2_tile (padded)
                N_I_GATHER * TILE_J * TILE_K +// p_tiles
                N_I_GATHER * 2 +              // m_l_sh
                N_I_GATHER * D_TMPL           // o_sh
            );

            Yq_gather<D_TMPL><<<grid_yq, block, smem_size, streams[0]>>>(
                reinterpret_cast<const float4*>(Q.data_ptr<float>()),
                reinterpret_cast<const float4*>(R.data_ptr<float>()),
                reinterpret_cast<const float4*>(S.data_ptr<float>()),
                reinterpret_cast<const float4*>(Vr_1.data_ptr<float>()),
                reinterpret_cast<const float4*>(Vs_1.data_ptr<float>()),
                Y_q.data_ptr<float>(),
                m_i.data_ptr<float>(),
                l_i.data_ptr<float>(),
                B, H, I, J, K, scale
            );
        }
    }
    AT_CUDA_CHECK(cudaEventRecord(gather_done[0], streams[0]));
    
    // GATHER: Y_r on stream 1
    {
        dim3 grid((J + N_I_GATHER - 1) / N_I_GATHER, H, B);
        constexpr int DP = D_TMPL + SMEM_PAD;
        size_t smem_size = sizeof(float) * (
            N_I_GATHER * D_TMPL + TILE_I * DP + TILE_K * DP +
            TILE_I * DP + TILE_K * DP +
            N_I_GATHER * TILE_I * TILE_K + N_I_GATHER * 2 + N_I_GATHER * D_TMPL
        );

        Yr_gather<D_TMPL><<<grid, block, smem_size, streams[1]>>>(
            reinterpret_cast<const float4*>(R.data_ptr<float>()),
            reinterpret_cast<const float4*>(Q.data_ptr<float>()),
            reinterpret_cast<const float4*>(S.data_ptr<float>()),
            reinterpret_cast<const float4*>(Vq_1.data_ptr<float>()),
            reinterpret_cast<const float4*>(Vs_1.data_ptr<float>()),
            Y_r.data_ptr<float>(),
            m_j.data_ptr<float>(),
            l_j.data_ptr<float>(),
            B, H, I, J, K, scale
        );
    }
    AT_CUDA_CHECK(cudaEventRecord(gather_done[1], streams[1]));

    // GATHER: Y_s on stream 2
    {
        dim3 grid((K + N_I_GATHER - 1) / N_I_GATHER, H, B);
        constexpr int DP = D_TMPL + SMEM_PAD;
        size_t smem_size = sizeof(float) * (
            N_I_GATHER * D_TMPL + TILE_I * DP + TILE_J * DP +
            TILE_I * DP + TILE_J * DP +
            N_I_GATHER * TILE_I * TILE_J + N_I_GATHER * 2 + N_I_GATHER * D_TMPL
        );

        Ys_gather<D_TMPL><<<grid, block, smem_size, streams[2]>>>(
            reinterpret_cast<const float4*>(S.data_ptr<float>()),
            reinterpret_cast<const float4*>(Q.data_ptr<float>()),
            reinterpret_cast<const float4*>(R.data_ptr<float>()),
            reinterpret_cast<const float4*>(Vq_1.data_ptr<float>()),
            reinterpret_cast<const float4*>(Vr_1.data_ptr<float>()),
            Y_s.data_ptr<float>(),
            m_k.data_ptr<float>(),
            l_k.data_ptr<float>(),
            B, H, I, J, K, scale
        );
    }
    AT_CUDA_CHECK(cudaEventRecord(gather_done[2], streams[2]));

    // Barrier: all scatter kernels wait for all gather kernels to complete
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            AT_CUDA_CHECK(cudaStreamWaitEvent(streams[i], gather_done[j], 0));
        }
    }

    // SCATTER - softmax stats were computed by gather kernels above, reuse them
    Yq_scatter_launcher_with_stats<D_TMPL>(Q, R, S, Vr_2, Vs_2, m_j, l_j, m_k, l_k, Y_q_, scale, streams[0]);
    Yr_scatter_launcher_with_stats<D_TMPL>(Q, R, S, Vq_2, Vs_2, m_i, l_i, m_k, l_k, Y_r_, scale, streams[1]);
    Ys_scatter_launcher_with_stats<D_TMPL>(Q, R, S, Vq_2, Vr_2, m_i, l_i, m_j, l_j, Y_s_, scale, streams[2]);
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




