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

#include "common.cuh"
#include "../cpp/cuda_bindings.h"

// ===================== gather kernels ======================

__global__
void Yq_gather(
    const float4* __restrict__ Q_f4,
    const float4* __restrict__ R_f4,
    const float4* __restrict__ S_f4,
    const float4* __restrict__ V1_f4, // Vr_1
    const float4* __restrict__ V2_f4, // Vs_1
    float*       __restrict__ Y,  // Y_q
    float*       __restrict__ m_i_out,  // softmax stats output (can be nullptr)
    float*       __restrict__ l_i_out,  // softmax stats output (can be nullptr)
    int B, int H, int I, int J, int K, int D, float scale)
{
    // --- Grid Mapping and Block Setup ---
    // Each block computes one output vector Y[b,h,i,:]
    const int i = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    // --- Shared Memory Allocation ---
    extern __shared__ float smem[];
    float* q_vec   = smem;                               // D
    float* r_tile  = q_vec + D;                          // TILE_J * D
    float* s_tile  = r_tile + TILE_J * D;                // TILE_K * D
    float* v1_tile = s_tile + TILE_K * D;                // TILE_J * D
    float* v2_tile = v1_tile + TILE_J * D;               // TILE_K * D
    float* p_tile  = v2_tile + TILE_K * D;               // TILE_J * TILE_K
    float* red_buf = p_tile + TILE_J * TILE_K;           // block_size
    // Shared memory for block-wide softmax stats and output vector
    float* m_l_sh  = red_buf + block_size;               // 2 floats for m_i, l_i
    float* o_sh    = m_l_sh + 2;                         // D floats for the output vector

    // --- Load Q[b,h,i,:] to Shared Memory using float4---
    const int64_t q_base_off = (((int64_t)b * H + h) * I + i) * D;
    const int64_t q_base_off_d4 = q_base_off / 4;
    for (int d4 = tid; d4 < D / 4; d4 += block_size) {
        ((float4*)q_vec)[d4] = Q_f4[q_base_off_d4 + d4];
    }

    // --- Initialize Online Softmax State in Shared Memory ---
    if (tid == 0) {
        m_l_sh[0] = NEG_INF; // m_i
        m_l_sh[1] = 0.0f;   // l_i
    }
    // Each thread initializes its portion of the output vector in shared memory
    for (int d = tid; d < D; d += block_size) {
        o_sh[d] = 0.0f;
    }
    __syncthreads();

    // --- Main Loop over R and S tiles ---
    for (int j0 = 0; j0 < J; j0 += TILE_J) {
        for (int k0 = 0; k0 < K; k0 += TILE_K) {
            
            // --- Load R, S, V1, V2 tiles into Shared Memory using float4 ---
            for (int j_loop_d4 = tid; j_loop_d4 < TILE_J * (D / 4); j_loop_d4 += block_size) {
                int jt = j_loop_d4 / (D / 4);
                int d4 = j_loop_d4 % (D / 4);
                int j = j0 + jt;
                if (j < J) {
                    const int64_t r_base_off_d4 = (((int64_t)b * H + h) * J + j) * (D / 4);
                    ((float4*)r_tile)[jt*(D/4) + d4] = R_f4[r_base_off_d4 + d4];
                    ((float4*)v1_tile)[jt*(D/4) + d4] = V1_f4[r_base_off_d4 + d4];
                }
            }
            for (int k_loop_d4 = tid; k_loop_d4 < TILE_K * (D / 4); k_loop_d4 += block_size) {
                int kt = k_loop_d4 / (D/4);
                int d4 = k_loop_d4 % (D/4);
                int k = k0 + kt;
                if (k < K) {
                    const int64_t s_base_off_d4 = (((int64_t)b * H + h) * K + k) * (D/4);
                    ((float4*)s_tile)[kt*(D/4) + d4] = S_f4[s_base_off_d4 + d4];
                    ((float4*)v2_tile)[kt*(D/4) + d4] = V2_f4[s_base_off_d4 + d4];
                }
            }
            __syncthreads();

            // --- Compute Dot Products for the Tile ---
           // Each thread computes (TILE_J*TILE_K)/block_size dots
            for (int flat_idx = tid; flat_idx < TILE_J * TILE_K; flat_idx += block_size) {
                int jt = flat_idx / TILE_K;
                int kt = flat_idx % TILE_K;
                float dot = 0.0f;
                if (j0 + jt < J && k0 + kt < K) {
                    // Unroll the dot product loop to increase ILP
                    float d_accum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    for (int d = 0; d < D; d += 4) {
                        d_accum[0] += q_vec[d+0] * r_tile[jt*D+(d+0)] * s_tile[kt*D+(d+0)];
                        d_accum[1] += q_vec[d+1] * r_tile[jt*D+(d+1)] * s_tile[kt*D+(d+1)];
                        d_accum[2] += q_vec[d+2] * r_tile[jt*D+(d+2)] * s_tile[kt*D+(d+2)];
                        d_accum[3] += q_vec[d+3] * r_tile[jt*D+(d+3)] * s_tile[kt*D+(d+3)];
                    }
                    dot = d_accum[0] + d_accum[1] + d_accum[2] + d_accum[3];
                    p_tile[flat_idx] = dot; // Use p_tile as temporary storage for dots
                } else {
                    p_tile[flat_idx] = NEG_INF; // Make sure padding doesn't affect max
                }
            }
            __syncthreads();

            // --- Find Tile Max (m_ij) ---
            float m_ij = NEG_INF;
            for (int flat_idx = tid; flat_idx < TILE_J * TILE_K; flat_idx += block_size) {
                m_ij = fmaxf(m_ij, p_tile[flat_idx] * scale);
            }
            red_buf[tid] = m_ij;
            __syncthreads();
            for (int s = block_size / 2; s > 0; s >>= 1) {
                if (tid < s) red_buf[tid] = fmaxf(red_buf[tid], red_buf[tid+s]);
                __syncthreads();
            }
            m_ij = red_buf[0];
            __syncthreads(); // Ensure all threads have read m_ij before red_buf is reused for sum

            // --- Compute Softmax Numerator (p_ij) and Denominator (l_ij) for the tile ---
            float l_ij = 0.0f;
            for (int flat_idx = tid; flat_idx < TILE_J * TILE_K; flat_idx += block_size) {
                int jt = flat_idx / TILE_K;
                int kt = flat_idx % TILE_K;
                if (j0 + jt < J && k0 + kt < K) {
                    float p_val = expf(p_tile[flat_idx] * scale - m_ij);
                    p_tile[flat_idx] = p_val;
                    l_ij += p_val;
                } else {
                    p_tile[flat_idx] = 0.0f;
                }
            }
            red_buf[tid] = l_ij;
            __syncthreads();
            for (int s = block_size / 2; s > 0; s >>= 1) {
                if (tid < s) red_buf[tid] += red_buf[tid+s];
                __syncthreads();
            }
            l_ij = red_buf[0];
            __syncthreads(); // Ensure all threads have read l_ij before red_buf is reused
            
            // --- Update Online Softmax State (Done by thread 0) ---
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
                // Store scaling factors for other threads to use
                red_buf[1] = alpha;
                red_buf[2] = beta;
                red_buf[3] = l_i_old;
            }
            __syncthreads();

            // All threads read the scaling factors
            alpha = red_buf[1];
            beta = red_buf[2];
            l_i_old = red_buf[3];
            l_new = m_l_sh[1];

            // --- Update Output Vector O ---
            // Each thread updates its portion of O in shared memory
            for(int d = tid; d < D; d += block_size) {
                float new_o_d = 0.0f;
                // Accumulate new value contribution from the tile
                for (int jt = 0; jt < TILE_J; ++jt) {
                    if (j0 + jt >= J) continue;
                    for (int kt = 0; kt < TILE_K; ++kt) {
                        if (k0 + kt >= K) continue;
                        new_o_d += p_tile[jt*TILE_K + kt] * v1_tile[jt*D + d] * v2_tile[kt*D + d];
                    }
                }
                
                // Rescale old O value and add new contribution
                if (l_new > 1e-20f) {
                    o_sh[d] = (alpha * l_i_old * o_sh[d] + beta * new_o_d) / l_new;
                }
            }
            __syncthreads();
        }
    }
    
    // --- Write Final Output from Shared to Global Memory ---
    for(int d = tid; d < D; d += block_size) {
        Y[q_base_off + d] = o_sh[d];
    }
    
    // --- Write Softmax Stats (if output pointers provided) ---
    if (tid == 0 && m_i_out != nullptr && l_i_out != nullptr) {
        int64_t stats_idx = ((int64_t)b * H + h) * I + i;
        m_i_out[stats_idx] = m_l_sh[0];
        l_i_out[stats_idx] = m_l_sh[1];
    }
}


__global__
void Yr_gather(
    const float4* __restrict__ R_query_f4,
    const float4* __restrict__ Q_f4,
    const float4* __restrict__ S_f4,
    const float4* __restrict__ V1_f4, // Vq_1
    const float4* __restrict__ V2_f4, // Vs_1
    float*       __restrict__ Y,      // Y_r
    float*       __restrict__ m_j_out,  // softmax stats output (can be nullptr)
    float*       __restrict__ l_j_out,  // softmax stats output (can be nullptr)
    int B, int H, int I, int J, int K, int D, float scale)
{
    // --- Grid Mapping and Block Setup ---
    // Each block computes one output vector Y[b,h,j,:]
    const int j = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    // --- Shared Memory Allocation ---
    extern __shared__ float smem[];
    float* query_vec = smem;                             // D
    float* i_tile    = query_vec + D;                    // TILE_I * D
    float* k_tile    = i_tile + TILE_I * D;              // TILE_K * D
    float* v1_tile   = k_tile + TILE_K * D;              // TILE_I * D
    float* v2_tile   = v1_tile + TILE_I * D;             // TILE_K * D
    float* p_tile    = v2_tile + TILE_K * D;             // TILE_I * TILE_K
    float* red_buf   = p_tile + TILE_I * TILE_K;         // block_size
    float* m_l_sh    = red_buf + block_size;             // 2 floats for m_i, l_i
    float* o_sh      = m_l_sh + 2;                       // D floats for the output vector

    // --- Load R_query[b,h,j,:] to Shared Memory ---
    const int64_t query_base_off = (((int64_t)b * H + h) * J + j) * D;
    const int64_t query_base_off_d4 = query_base_off / 4;
    for (int d4 = tid; d4 < D / 4; d4 += block_size) {
        ((float4*)query_vec)[d4] = R_query_f4[query_base_off_d4 + d4];
    }

    // --- Initialize ---
    if (tid == 0) {
        m_l_sh[0] = NEG_INF; // m_i
        m_l_sh[1] = 0.0f;   // l_i
    }
    for (int d = tid; d < D; d += block_size) o_sh[d] = 0.0f;
    __syncthreads();

    // --- Main Loop over Q and S tiles ---
    for (int i0 = 0; i0 < I; i0 += TILE_I) {
        for (int k0 = 0; k0 < K; k0 += TILE_K) {
            
            // --- Load Q, S, Vq_1, Vs_1 tiles ---
            for (int i_loop_d4 = tid; i_loop_d4 < TILE_I * (D / 4); i_loop_d4 += block_size) {
                int it = i_loop_d4 / (D / 4);
                int d4 = i_loop_d4 % (D / 4);
                int i_ = i0 + it;
                if (i_ < I) {
                    const int64_t i_base_off_d4 = (((int64_t)b * H + h) * I + i_) * (D / 4);
                    ((float4*)i_tile)[it*(D/4) + d4] = Q_f4[i_base_off_d4 + d4];
                    ((float4*)v1_tile)[it*(D/4) + d4] = V1_f4[i_base_off_d4 + d4];
                }
            }
            for (int k_loop_d4 = tid; k_loop_d4 < TILE_K * (D / 4); k_loop_d4 += block_size) {
                int kt = k_loop_d4 / (D/4);
                int d4 = k_loop_d4 % (D/4);
                int k = k0 + kt;
                if (k < K) {
                    const int64_t k_base_off_d4 = (((int64_t)b * H + h) * K + k) * (D/4);
                    ((float4*)k_tile)[kt*(D/4) + d4] = S_f4[k_base_off_d4 + d4];
                    ((float4*)v2_tile)[kt*(D/4) + d4] = V2_f4[k_base_off_d4 + d4];
                }
            }
            __syncthreads();

            // --- Compute Dot Products ---
            for (int flat_idx = tid; flat_idx < TILE_I * TILE_K; flat_idx += block_size) {
                int it = flat_idx / TILE_K;
                int kt = flat_idx % TILE_K;
                float dot = 0.0f;
                if (i0 + it < I && k0 + kt < K) {
                    for (int d = 0; d < D; ++d) dot += query_vec[d] * i_tile[it*D+d] * k_tile[kt*D+d];
                    p_tile[flat_idx] = dot;
                } else {
                    p_tile[flat_idx] = NEG_INF;
                }
            }
            __syncthreads();

            // --- Find Tile Max (m_ij) ---
            float m_ij = NEG_INF;
            for (int flat_idx = tid; flat_idx < TILE_I * TILE_K; flat_idx += block_size) {
                m_ij = fmaxf(m_ij, p_tile[flat_idx] * scale);
            }
            red_buf[tid] = m_ij;
            __syncthreads();
            for (int s = block_size / 2; s > 0; s >>= 1) {
                if (tid < s) red_buf[tid] = fmaxf(red_buf[tid], red_buf[tid+s]);
                __syncthreads();
            }
            m_ij = red_buf[0];
            __syncthreads(); // Ensure all threads have read m_ij before red_buf is reused

            // --- Compute Softmax Numerator & Denominator ---
            float l_ij = 0.0f;
            for (int flat_idx = tid; flat_idx < TILE_I * TILE_K; flat_idx += block_size) {
                if (i0 + (flat_idx/TILE_K) < I && k0 + (flat_idx%TILE_K) < K) {
                    float p_val = expf(p_tile[flat_idx] * scale - m_ij);
                    p_tile[flat_idx] = p_val;
                    l_ij += p_val;
                } else {
                    p_tile[flat_idx] = 0.0f;
                }
            }
            red_buf[tid] = l_ij;
            __syncthreads();
            for (int s = block_size / 2; s > 0; s >>= 1) {
                if (tid < s) red_buf[tid] += red_buf[tid+s];
                __syncthreads();
            }
            l_ij = red_buf[0];
            __syncthreads(); // Ensure all threads have read l_ij before red_buf is reused
            
            // --- Update Online Softmax State ---
            float m_i_old, l_i_old, m_new, alpha, beta, l_new;
            if (tid == 0) {
                m_i_old = m_l_sh[0]; l_i_old = m_l_sh[1];
                m_new = fmaxf(m_i_old, m_ij);
                alpha = expf(m_i_old - m_new); beta = expf(m_ij - m_new);
                l_new = alpha * l_i_old + beta * l_ij;
                m_l_sh[0] = m_new; m_l_sh[1] = l_new;
                red_buf[1] = alpha; red_buf[2] = beta; red_buf[3] = l_i_old;
            }
            __syncthreads();
            alpha = red_buf[1]; beta = red_buf[2]; l_i_old = red_buf[3]; l_new = m_l_sh[1];

            // --- Update Output Vector O ---
            for(int d = tid; d < D; d += block_size) {
                float new_o_d = 0.0f;
                for (int it = 0; it < TILE_I; ++it) {
                    if (i0+it >= I) continue;
                    for (int kt = 0; kt < TILE_K; ++kt) {
                        if (k0+kt < K) new_o_d += p_tile[it*TILE_K+kt] * v1_tile[it*D+d] * v2_tile[kt*D+d];
                    }
                }
                if (l_new > 1e-20f) o_sh[d] = (alpha * l_i_old * o_sh[d] + beta * new_o_d) / l_new;
            }
            __syncthreads();
        }
    }
    
    // --- Write Final Output to Global Memory ---
    for(int d = tid; d < D; d += block_size) {
        Y[query_base_off + d] = o_sh[d];
    }
    
    // --- Write Softmax Stats (if output pointers provided) ---
    if (tid == 0 && m_j_out != nullptr && l_j_out != nullptr) {
        int64_t stats_idx = ((int64_t)b * H + h) * J + j;
        m_j_out[stats_idx] = m_l_sh[0];
        l_j_out[stats_idx] = m_l_sh[1];
    }
}


__global__
void Ys_gather(
    const float4* __restrict__ S_query_f4,
    const float4* __restrict__ Q_f4,
    const float4* __restrict__ R_f4,
    const float4* __restrict__ V1_f4, // Vq_1
    const float4* __restrict__ V2_f4, // Vr_1
    float*       __restrict__ Y,      // Y_s
    float*       __restrict__ m_k_out,  // softmax stats output (can be nullptr)
    float*       __restrict__ l_k_out,  // softmax stats output (can be nullptr)
    int B, int H, int I, int J, int K, int D, float scale)
{
    // --- Grid Mapping and Block Setup ---
    // Each block computes one output vector Y[b,h,k,:]
    const int k = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    // --- Shared Memory Allocation ---
    extern __shared__ float smem[];
    float* query_vec = smem;                             // D
    float* i_tile    = query_vec + D;                    // TILE_I * D
    float* j_tile    = i_tile + TILE_I * D;              // TILE_J * D
    float* v1_tile   = j_tile + TILE_J * D;              // TILE_I * D
    float* v2_tile   = v1_tile + TILE_I * D;             // TILE_J * D
    float* p_tile    = v2_tile + TILE_J * D;             // TILE_I * TILE_J
    float* red_buf   = p_tile + TILE_I * TILE_J;         // block_size
    float* m_l_sh    = red_buf + block_size;             // 2 floats for m_i, l_i
    float* o_sh      = m_l_sh + 2;                       // D floats for the output vector

    // --- Load S_query[b,h,k,:] to Shared Memory ---
    const int64_t query_base_off = (((int64_t)b * H + h) * K + k) * D;
    const int64_t query_base_off_d4 = query_base_off / 4;
    for (int d4 = tid; d4 < D / 4; d4 += block_size) {
        ((float4*)query_vec)[d4] = S_query_f4[query_base_off_d4 + d4];
    }

    // --- Initialize ---
    if (tid == 0) {
        m_l_sh[0] = NEG_INF; // m_i
        m_l_sh[1] = 0.0f;   // l_i
    }
    for (int d = tid; d < D; d += block_size) o_sh[d] = 0.0f;
    __syncthreads();

    // --- Main Loop over Q and R tiles ---
    for (int i0 = 0; i0 < I; i0 += TILE_I) {
        for (int j0 = 0; j0 < J; j0 += TILE_J) {
            
            // --- Load Q, R, Vq_1, Vr_1 tiles ---
            for (int i_loop_d4 = tid; i_loop_d4 < TILE_I * (D / 4); i_loop_d4 += block_size) {
                int it = i_loop_d4 / (D / 4);
                int d4 = i_loop_d4 % (D / 4);
                int i_ = i0 + it;
                if (i_ < I) {
                    const int64_t i_base_off_d4 = (((int64_t)b * H + h) * I + i_) * (D / 4);
                    ((float4*)i_tile)[it*(D/4) + d4] = Q_f4[i_base_off_d4 + d4];
                    ((float4*)v1_tile)[it*(D/4) + d4] = V1_f4[i_base_off_d4 + d4];
                }
            }
            for (int j_loop_d4 = tid; j_loop_d4 < TILE_J * (D / 4); j_loop_d4 += block_size) {
                int jt = j_loop_d4 / (D/4);
                int d4 = j_loop_d4 % (D/4);
                int j = j0 + jt;
                if (j < J) {
                    const int64_t j_base_off_d4 = (((int64_t)b * H + h) * J + j) * (D/4);
                    ((float4*)j_tile)[jt*(D/4) + d4] = R_f4[j_base_off_d4 + d4];
                    ((float4*)v2_tile)[jt*(D/4) + d4] = V2_f4[j_base_off_d4 + d4];
                }
            }
            __syncthreads();

            // --- Compute Dot Products ---
            for (int flat_idx = tid; flat_idx < TILE_I * TILE_J; flat_idx += block_size) {
                int it = flat_idx / TILE_J;
                int jt = flat_idx % TILE_J;
                float dot = 0.0f;
                if (i0 + it < I && j0 + jt < J) {
                    for (int d = 0; d < D; ++d) dot += query_vec[d] * i_tile[it*D+d] * j_tile[jt*D+d];
                    p_tile[flat_idx] = dot;
                } else {
                    p_tile[flat_idx] = NEG_INF;
                }
            }
            __syncthreads();

            // --- Find Tile Max (m_ij) ---
            float m_ij = NEG_INF;
            for (int flat_idx = tid; flat_idx < TILE_I * TILE_J; flat_idx += block_size) {
                m_ij = fmaxf(m_ij, p_tile[flat_idx] * scale);
            }
            red_buf[tid] = m_ij;
            __syncthreads();
            for (int s = block_size / 2; s > 0; s >>= 1) {
                if (tid < s) red_buf[tid] = fmaxf(red_buf[tid], red_buf[tid+s]);
                __syncthreads();
            }
            m_ij = red_buf[0];
            __syncthreads(); // Ensure all threads have read m_ij before red_buf is reused

            // --- Compute Softmax Numerator & Denominator ---
            float l_ij = 0.0f;
            for (int flat_idx = tid; flat_idx < TILE_I * TILE_J; flat_idx += block_size) {
                if (i0 + (flat_idx/TILE_J) < I && j0 + (flat_idx%TILE_J) < J) {
                    float p_val = expf(p_tile[flat_idx] * scale - m_ij);
                    p_tile[flat_idx] = p_val;
                    l_ij += p_val;
                } else {
                    p_tile[flat_idx] = 0.0f;
                }
            }
            red_buf[tid] = l_ij;
            __syncthreads();
            for (int s = block_size / 2; s > 0; s >>= 1) {
                if (tid < s) red_buf[tid] += red_buf[tid+s];
                __syncthreads();
            }
            l_ij = red_buf[0];
            __syncthreads(); // Ensure all threads have read l_ij before red_buf is reused
            
            // --- Update Online Softmax State ---
            float m_i_old, l_i_old, m_new, alpha, beta, l_new;
            if (tid == 0) {
                m_i_old = m_l_sh[0]; l_i_old = m_l_sh[1];
                m_new = fmaxf(m_i_old, m_ij);
                alpha = expf(m_i_old - m_new); beta = expf(m_ij - m_new);
                l_new = alpha * l_i_old + beta * l_ij;
                m_l_sh[0] = m_new; m_l_sh[1] = l_new;
                red_buf[1] = alpha; red_buf[2] = beta; red_buf[3] = l_i_old;
            }
            __syncthreads();
            alpha = red_buf[1]; beta = red_buf[2]; l_i_old = red_buf[3]; l_new = m_l_sh[1];

            // --- Update Output Vector O ---
            for(int d = tid; d < D; d += block_size) {
                float new_o_d = 0.0f;
                for (int it = 0; it < TILE_I; ++it) {
                    if (i0+it >= I) continue;
                    for (int jt = 0; jt < TILE_J; ++jt) {
                        if (j0+jt < J) new_o_d += p_tile[it*TILE_J+jt] * v1_tile[it*D+d] * v2_tile[jt*D+d];
                    }
                }
                if (l_new > 1e-20f) o_sh[d] = (alpha * l_i_old * o_sh[d] + beta * new_o_d) / l_new;
            }
            __syncthreads();
        }
    }
    
    // --- Write Final Output to Global Memory ---
    for(int d = tid; d < D; d += block_size) {
        Y[query_base_off + d] = o_sh[d];
    }
    
    // --- Write Softmax Stats (if output pointers provided) ---
    if (tid == 0 && m_k_out != nullptr && l_k_out != nullptr) {
        int64_t stats_idx = ((int64_t)b * H + h) * K + k;
        m_k_out[stats_idx] = m_l_sh[0];
        l_k_out[stats_idx] = m_l_sh[1];
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
 * - D must be a multiple of 32
 * - I, J, K must be equal and multiples of TILE_I (16)
 * - I, J, K must be >= TILE_I
 */
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
    int B, int H, int I, int J, int K, int D, float scale
) {
    // --- Grid/Block Mapping ---
    const int b = blockIdx.z;
    const int h = blockIdx.y;
    const int i_start = blockIdx.x * TILE_I;

    const int tid = threadIdx.x;
    const int tpb = blockDim.x;

    // --- Memory Offsets ---
    const int64_t q_bh_offset = (int64_t)(b * H + h) * I * D;
    const int64_t r_bh_offset = (int64_t)(b * H + h) * J * D;
    const int64_t s_bh_offset = (int64_t)(b * H + h) * K * D;
    const int64_t mj_bh_offset = (int64_t)(b * H + h) * J;
    const int64_t mk_bh_offset = (int64_t)(b * H + h) * K;

    // --- Shared Memory Layout ---
    extern __shared__ float smem[];
    float* q_tile = smem;
    float* r_tile = q_tile + TILE_I * D;
    float* s_tile = r_tile + TILE_J * D;
    float* vr_tile = s_tile + TILE_K * D;
    float* vs_tile = vr_tile + TILE_J * D;
    float* attn_tile = vs_tile + TILE_K * D;
    float* mj_tile = attn_tile + TILE_I * TILE_J * TILE_K;
    float* lj_tile = mj_tile + TILE_J;
    float* mk_tile = lj_tile + TILE_J;
    float* lk_tile = mk_tile + TILE_K;

    // --- Thread Indexing for Cooperative Loads ---
    const int i_load = tid / D;
    const int d_load = tid % D;
    const int load_iters = (TILE_I * D + tpb - 1) / tpb;
    const int load_step = tpb / D;

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
            q_tile[(n + i_load) * D + d_load] = Q[q_bh_offset + i_global * D + d_load];
        }
    }

    // --- Main Loop: Iterate Over J Tiles ---
    for (int jt = 0; jt < J; jt += TILE_J) {
        // Load R tile
        for (int n = 0; n < TILE_J; n += load_step) {
            int j_global = jt + n + i_load;
            if (n + i_load < TILE_J && j_global < J) {
                r_tile[(n + i_load) * D + d_load] = R[r_bh_offset + j_global * D + d_load];
            }
        }
        // Load Vr_2 tile
        for (int n = 0; n < TILE_J; n += load_step) {
            int j_global = jt + n + i_load;
            if (n + i_load < TILE_J && j_global < J) {
                vr_tile[(n + i_load) * D + d_load] = Vr_2[r_bh_offset + j_global * D + d_load];
            }
        }
        // Load softmax stats for J (m_j, l_j)
        if (tid < TILE_J && jt + tid < J) {
            mj_tile[tid] = m_j_in[mj_bh_offset + jt + tid];
            lj_tile[tid] = 1.0f / l_j_in[mj_bh_offset + jt + tid];
        }

        // --- Inner Loop: Iterate Over K Tiles ---
        for (int kt = 0; kt < K; kt += TILE_K) {
            // Load S tile
            for (int n = 0; n < TILE_K; n += load_step) {
                int k_global = kt + n + i_load;
                if (n + i_load < TILE_K && k_global < K) {
                    s_tile[(n + i_load) * D + d_load] = S[s_bh_offset + k_global * D + d_load];
                }
            }
            // Load softmax stats for K (m_k, l_k)
            if (tid < TILE_K && kt + tid < K) {
                mk_tile[tid] = m_k_in[mk_bh_offset + kt + tid];
                lk_tile[tid] = 1.0f / l_k_in[mk_bh_offset + kt + tid];
            }
            __syncthreads();

            // --- Parallel Outer-Product Attention Computation ---
            // 256 threads compute a TILE_I × TILE_J × TILE_K attention cube
            // Thread mapping:
            //   da: dimension slice (0-3, each handles D/4 dimensions)
            //   ia: i sub-tile index (0-3)
            //   ja: j sub-tile index (0-3)
            //   ka: k sub-tile index (0-3)
            // Each thread computes a 4×4×4 sub-cube = 64 partial dot products
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
            const int d_per_slice = D / 4;

            // Accumulate partial dot products
            for (int db = 0; db < d_per_slice; db++) {
                const int d_idx = da * d_per_slice + db;
                #pragma unroll
                for (int u = 0; u < 4; u++) {
                    qa[u] = q_tile[(ia * 4 + u) * D + d_idx];
                    ra[u] = r_tile[(ja * 4 + u) * D + d_idx];
                    sa[u] = s_tile[(ka * 4 + u) * D + d_idx];
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

            // --- Apply Softmax Scaling (Ar * As) ---
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

            // Load Vs_2 tile
            for (int n = 0; n < TILE_K; n += load_step) {
                int k_global = kt + n + i_load;
                if (n + i_load < TILE_K && k_global < K) {
                    vs_tile[(n + i_load) * D + d_load] = Vs_2[s_bh_offset + k_global * D + d_load];
                }
            }
            __syncthreads();

            // --- Accumulate Output: attention × Vr_2 × Vs_2 ---
            for (int n = 0; n < load_iters; n++) {
                int tid_n = tid + n * tpb;
                if (tid_n < TILE_I * D) {
                    int iy = tid_n / D;
                    int dy = tid_n % D;

                    float f = 0.0f;
                    for (int jy = 0; jy < TILE_J; jy++) {
                        float vrt = vr_tile[jy * D + dy];
                        for (int ky = 0; ky < TILE_K; ky++) {
                            f += attn_tile[iy * TILE_J * TILE_K + jy * TILE_K + ky] * vrt * vs_tile[ky * D + dy];
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
        if (tid_n < TILE_I * D) {
            int iy = tid_n / D;
            int dy = tid_n % D;
            int i_global = i_start + iy;
            if (i_global < I) {
                Y_q_[q_bh_offset + i_global * D + dy] = yq_acc[n];
            }
        }
    }
}

void Yq_scatter_launcher_with_stats(
    const at::Tensor& Q, const at::Tensor& R, const at::Tensor& S,
    const at::Tensor& Vr_2, const at::Tensor& Vs_2,
    const at::Tensor& m_j, const at::Tensor& l_j,
    const at::Tensor& m_k, const at::Tensor& l_k,
    at::Tensor& Y_q_, float scale
) {
    const auto B = Q.size(0);
    const auto H = Q.size(1);
    const auto I = Q.size(2);
    const auto J = R.size(2);
    const auto K = S.size(2);
    const auto D = Q.size(3);

    TORCH_CHECK(D % 32 == 0, "Yq_scatter requires D to be a multiple of 32");
    TORCH_CHECK(I == J && J == K, "Yq_scatter requires I == J == K");
    TORCH_CHECK(I % TILE_I == 0, "Yq_scatter requires I to be a multiple of TILE_I (16)");
    TORCH_CHECK(I >= TILE_I, "Yq_scatter requires I >= TILE_I (16)");
    TORCH_CHECK((TILE_I * D) / 256 <= 8, "Yq_scatter: D too large (max 8 load iterations)");

    const int TPB = 256;
    dim3 grid((I + TILE_I - 1) / TILE_I, H, B);
    dim3 block(TPB);

    size_t smem_size = sizeof(float) * (
        TILE_I * D +                    // q_tile
        TILE_J * D +                    // r_tile
        TILE_K * D +                    // s_tile
        TILE_J * D +                    // vr_tile
        TILE_K * D +                    // vs_tile
        TILE_I * TILE_J * TILE_K +      // attn_tile
        TILE_J + TILE_J +               // mj_tile, lj_tile
        TILE_K + TILE_K                 // mk_tile, lk_tile
    );

    Yq_scatter<<<grid, block, smem_size>>>(
        Q.data_ptr<float>(), R.data_ptr<float>(), S.data_ptr<float>(),
        Vr_2.data_ptr<float>(), Vs_2.data_ptr<float>(),
        m_j.data_ptr<float>(), l_j.data_ptr<float>(),
        m_k.data_ptr<float>(), l_k.data_ptr<float>(),
        Y_q_.data_ptr<float>(),
        B, H, I, J, K, D, scale
    );
}


__global__ void Yr_scatter(
    const float* __restrict__ Q, const float* __restrict__ R, const float* __restrict__ S,
    const float* __restrict__ Vq_2, const float* __restrict__ Vs_2,
    const float* __restrict__ m_i_in, const float* __restrict__ l_i_in,
    const float* __restrict__ m_k_in, const float* __restrict__ l_k_in,
    float* __restrict__ Y_r_,
    int B, int H, int I, int J, int K, int D, float scale)
{
    // --- Grid/Block Mapping ---
    const int j_tile_idx_grid = blockIdx.x;
    const int i_tile_idx_grid = blockIdx.y;
    const int bh_idx = blockIdx.z;

    const int d_idx = threadIdx.x;
    const int j_local_idx = threadIdx.y;

    const int j_base = j_tile_idx_grid * TILE_J_SCATTER;
    const int j_idx = j_base + j_local_idx;

    const int i0 = i_tile_idx_grid * TILE_I;

    // NOTE: We cannot return early because all threads must participate in __syncthreads().
    const bool valid_thread = (j_idx < J) && (i0 < I);

    // --- Pointers ---
    const int64_t q_bh_offset = (int64_t)bh_idx * I * D;
    const int64_t r_bh_offset = (int64_t)bh_idx * J * D;
    const int64_t s_bh_offset = (int64_t)bh_idx * K * D;
    const int64_t vq_bh_offset = (int64_t)bh_idx * I * D;
    const int64_t vs_bh_offset = (int64_t)bh_idx * K * D;
    const int64_t mi_bh_offset = (int64_t)bh_idx * I;
    const int64_t mk_bh_offset = (int64_t)bh_idx * K;
    const int64_t yr_bh_offset = (int64_t)bh_idx * J * D;

    // --- Shared Memory Layout ---
    extern __shared__ float smem[];
    float* r_tile = (float*)smem;
    float* q_tile = r_tile + TILE_J_SCATTER * D;
    float* s_tile = q_tile + TILE_I * D;
    float* vq_tile = s_tile + TILE_K * D;
    float* vs_tile = vq_tile + TILE_I * D;
    float* mi_tile = (float*)(vs_tile + TILE_K * D);
    float* li_tile = mi_tile + TILE_I;
    float* mk_tile = li_tile + TILE_I;
    float* lk_tile = mk_tile + TILE_K;
    float* o_tile = lk_tile + TILE_K;
    float* attn_scores_tile = o_tile + TILE_J_SCATTER * D;

    int flat_thread_id_2d = threadIdx.y * blockDim.x + threadIdx.x;
    int threads_per_block = blockDim.x * blockDim.y;
    
    // Initialize the shared memory accumulator tile
    o_tile[j_local_idx * D + d_idx] = 0.0f;

    // --- Load R tile for this block ---
    for (int load_idx = flat_thread_id_2d; load_idx < TILE_J_SCATTER * D; load_idx += threads_per_block) {
        int row_in_tile = load_idx / D;
        int col_in_tile = load_idx % D;
        int j_global = j_base + row_in_tile;
        if (j_global < J) {
            r_tile[row_in_tile * D + col_in_tile] =
                R[r_bh_offset + j_global * D + col_in_tile];
        }
    }
    __syncthreads();

    // --- Cooperative loading for i-related tiles ---
    for (int load_idx = flat_thread_id_2d; load_idx < TILE_I * D; load_idx += threads_per_block) {
        int row_in_tile = load_idx / D;
        int col_in_tile = load_idx % D;
        int i_global = i0 + row_in_tile;
        if (i_global < I) {
            q_tile[row_in_tile * D + col_in_tile] = Q[q_bh_offset + i_global * D + col_in_tile];
            vq_tile[row_in_tile * D + col_in_tile] = Vq_2[vq_bh_offset + i_global * D + col_in_tile];
        }
    }

    for (int i_load = flat_thread_id_2d; i_load < TILE_I; i_load += threads_per_block) {
        if (i0 + i_load < I) {
             mi_tile[i_load] = m_i_in[mi_bh_offset + i0 + i_load];
             li_tile[i_load] = l_i_in[mi_bh_offset + i0 + i_load];
        }
    }

    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        // --- Cooperative loading for k-related tiles ---
        for (int load_idx = flat_thread_id_2d; load_idx < TILE_K * D; load_idx += threads_per_block) {
            int row_in_tile = load_idx / D;
            int col_in_tile = load_idx % D;
            int k_global = k0 + row_in_tile;
            if (k_global < K) {
                s_tile[row_in_tile * D + col_in_tile] = S[s_bh_offset + k_global * D + col_in_tile];
                vs_tile[row_in_tile * D + col_in_tile] = Vs_2[vs_bh_offset + k_global * D + col_in_tile];
             }
        }
        for (int k_load = flat_thread_id_2d; k_load < TILE_K; k_load += threads_per_block) {
            if (k0 + k_load < K) {
                mk_tile[k_load] = m_k_in[mk_bh_offset + k0 + k_load];
                lk_tile[k_load] = l_k_in[mk_bh_offset + k0 + k_load];
            }
        }
        __syncthreads();

        // --- Fused Computation ---
        const float* r_vec = r_tile + j_local_idx * D;

        for (int i_tile_idx = 0; i_tile_idx < TILE_I; ++i_tile_idx) {
            if (i0 + i_tile_idx >= I) continue;
            const float* q_vec = q_tile + i_tile_idx * D;
            float inv_li = 1.0f / li_tile[i_tile_idx];
            float current_mi = mi_tile[i_tile_idx];

            for (int k_tile_idx = 0; k_tile_idx < TILE_K; ++k_tile_idx) {
                if (k0 + k_tile_idx >= K) continue;
                
                if (d_idx == 0) {
                    const float* s_vec = s_tile + k_tile_idx * D;
                    float inv_lk = 1.0f / lk_tile[k_tile_idx];

                    const float4* q_vec_f4 = (const float4*)q_vec;
                    const float4* r_vec_f4 = (const float4*)r_vec;
                    const float4* s_vec_f4 = (const float4*)s_vec;  
                    float dot = 0.0f;
                    #pragma unroll
                    for (int d4 = 0; d4 < D / 4; ++d4) {
                        float4 q = q_vec_f4[d4];
                        float4 r = r_vec_f4[d4];
                        float4 s = s_vec_f4[d4];
                        dot += q.x * r.x * s.x + q.y * r.y * s.y + q.z * r.z * s.z + q.w * r.w * s.w;
                    }
                    float logit = dot * scale;
                    float aq_val = expf(logit - current_mi) * inv_li;
                    float as_val = expf(logit - mk_tile[k_tile_idx]) * inv_lk;
                    attn_scores_tile[j_local_idx] = aq_val * as_val;
                }
                __syncthreads();

                const float* vq_vec = vq_tile + i_tile_idx * D;
                const float* vs_vec = vs_tile + k_tile_idx * D;
                float combined_attn_val = attn_scores_tile[j_local_idx];
                
                if (d_idx < D) {
                    o_tile[j_local_idx * D + d_idx] += combined_attn_val * vq_vec[d_idx] * vs_vec[d_idx];
                }
                __syncthreads();
            }
        }
    }
    
    // --- Write final result to global memory ---
    if (valid_thread && d_idx < D) {
        float final_val = o_tile[j_local_idx * D + d_idx];
        atomicAdd(&Y_r_[yr_bh_offset + j_idx * D + d_idx], final_val);
    }
}

void Yr_scatter_launcher_with_stats(
    const at::Tensor& Q, const at::Tensor& R, const at::Tensor& S,
    const at::Tensor& Vq_2, const at::Tensor& Vs_2,
    const at::Tensor& m_i, const at::Tensor& l_i,
    const at::Tensor& m_k, const at::Tensor& l_k,
    at::Tensor& Y_r_, float scale
) {
    const auto B = Q.size(0);
    const auto H = Q.size(1);
    const auto I = Q.size(2);
    const auto J = R.size(2);
    const auto K = S.size(2);
    const auto D = Q.size(3);
    
    Y_r_.zero_();

    {
        TORCH_CHECK(D * TILE_J_SCATTER <= 1024, "D * TILE_J_SCATTER must be <= 1024 for the 2D block size.");
        dim3 grid(
            (J + TILE_J_SCATTER - 1) / TILE_J_SCATTER,
            (I + TILE_I - 1) / TILE_I,
            B * H
        );
        dim3 block(D, TILE_J_SCATTER);
        size_t smem_size = sizeof(float) * (
            TILE_J_SCATTER*D + TILE_I*D + TILE_K*D + 
            TILE_I*D + TILE_K*D +             
            TILE_I + TILE_I + TILE_K + TILE_K +
            TILE_J_SCATTER*D
        );
        Yr_scatter<<<grid, block, smem_size>>>(
            Q.data_ptr<float>(), R.data_ptr<float>(), S.data_ptr<float>(),
            Vq_2.data_ptr<float>(), Vs_2.data_ptr<float>(),
            m_i.data_ptr<float>(), l_i.data_ptr<float>(),
            m_k.data_ptr<float>(), l_k.data_ptr<float>(),
            Y_r_.data_ptr<float>(),
            B, H, I, J, K, D, scale
        );
    }
}


__global__ void Ys_scatter(
    const float* __restrict__ Q, const float* __restrict__ R, const float* __restrict__ S,
    const float* __restrict__ Vq_2, const float* __restrict__ Vr_2,
    const float* __restrict__ m_i_in, const float* __restrict__ l_i_in,
    const float* __restrict__ m_j_in, const float* __restrict__ l_j_in,
    float* __restrict__ Y_s_,
    int B, int H, int I, int J, int K, int D, float scale
) {
    // --- Grid/Block Mapping ---
    const int k_tile_idx_grid = blockIdx.x;
    const int i_tile_idx_grid = blockIdx.y;
    const int bh_idx = blockIdx.z;

    const int d_idx = threadIdx.x;
    const int k_local_idx = threadIdx.y;

    const int k_base = k_tile_idx_grid * TILE_K_SCATTER;
    const int k_idx = k_base + k_local_idx;

    const int i0 = i_tile_idx_grid * TILE_I;

    // NOTE: We cannot return early because all threads must participate in __syncthreads().
    const bool valid_thread = (k_idx < K) && (i0 < I);

    // --- Pointers ---
    const int64_t q_bh_offset = (int64_t)bh_idx * I * D;
    const int64_t r_bh_offset = (int64_t)bh_idx * J * D;
    const int64_t s_bh_offset = (int64_t)bh_idx * K * D;
    const int64_t vq_bh_offset = (int64_t)bh_idx * I * D;
    const int64_t vr_bh_offset = (int64_t)bh_idx * J * D;
    const int64_t mi_bh_offset = (int64_t)bh_idx * I;
    const int64_t mj_bh_offset = (int64_t)bh_idx * J;
    const int64_t ys_bh_offset = (int64_t)bh_idx * K * D;

    // --- Shared Memory Layout ---
    extern __shared__ float smem[];
    float* s_tile = (float*)smem;
    float* q_tile = s_tile + TILE_K_SCATTER * D;
    float* r_tile = q_tile + TILE_I * D;
    float* vq_tile = r_tile + TILE_J * D;
    float* vr_tile = vq_tile + TILE_I * D;
    float* mi_tile = (float*)(vr_tile + TILE_J * D);
    float* li_tile = mi_tile + TILE_I;
    float* mj_tile = li_tile + TILE_I;
    float* lj_tile = mj_tile + TILE_J;
    float* o_tile = lj_tile + TILE_J;
    float* attn_scores_tile = o_tile + TILE_K_SCATTER * D;

    int flat_thread_id_2d = threadIdx.y * blockDim.x + threadIdx.x;
    int threads_per_block = blockDim.x * blockDim.y;
    
    o_tile[k_local_idx * D + d_idx] = 0.0f;

    // --- Load S tile for this block ---
    for (int load_idx = flat_thread_id_2d; load_idx < TILE_K_SCATTER * D; load_idx += threads_per_block) {
        int row_in_tile = load_idx / D;
        int col_in_tile = load_idx % D;
        int k_global = k_base + row_in_tile;
        if (k_global < K) {
            s_tile[row_in_tile * D + col_in_tile] = S[s_bh_offset + k_global * D + col_in_tile];
        }
    }
    
    // --- Cooperative loading for i-related tiles ---
    for (int load_idx = flat_thread_id_2d; load_idx < TILE_I * D; load_idx += threads_per_block) {
        int row_in_tile = load_idx / D;
        int col_in_tile = load_idx % D;
        int i_global = i0 + row_in_tile;
        if (i_global < I) {
            q_tile[row_in_tile * D + col_in_tile] = Q[q_bh_offset + i_global * D + col_in_tile];
            vq_tile[row_in_tile * D + col_in_tile] = Vq_2[vq_bh_offset + i_global * D + col_in_tile];
        }
    }
    for (int i_load = flat_thread_id_2d; i_load < TILE_I; i_load += threads_per_block) {
        if (i0 + i_load < I) {
             mi_tile[i_load] = m_i_in[mi_bh_offset + i0 + i_load];
             li_tile[i_load] = l_i_in[mi_bh_offset + i0 + i_load];
        }
    }
    __syncthreads();

    for (int j0 = 0; j0 < J; j0 += TILE_J) {
        // --- Cooperative loading for j-related tiles ---
        for (int load_idx = flat_thread_id_2d; load_idx < TILE_J * D; load_idx += threads_per_block) {
            int row_in_tile = load_idx / D;
            int col_in_tile = load_idx % D;
            int j_global = j0 + row_in_tile;
            if (j_global < J) {
                r_tile[row_in_tile * D + col_in_tile] = R[r_bh_offset + j_global * D + col_in_tile];
                vr_tile[row_in_tile * D + col_in_tile] = Vr_2[vr_bh_offset + j_global * D + col_in_tile];
             }
        }
        for (int j_load = flat_thread_id_2d; j_load < TILE_J; j_load += threads_per_block) {
            if (j0 + j_load < J) {
                mj_tile[j_load] = m_j_in[mj_bh_offset + j0 + j_load];
                lj_tile[j_load] = l_j_in[mj_bh_offset + j0 + j_load];
            }
        }
        __syncthreads();

        // --- Fused Computation ---
        const float* s_vec = s_tile + k_local_idx * D;

        for (int i_tile_idx = 0; i_tile_idx < TILE_I; ++i_tile_idx) {
            if (i0 + i_tile_idx >= I) continue;
            const float* q_vec = q_tile + i_tile_idx * D;
            float inv_li = 1.0f / li_tile[i_tile_idx];
            float current_mi = mi_tile[i_tile_idx];

            for (int j_tile_idx = 0; j_tile_idx < TILE_J; ++j_tile_idx) {
                if (j0 + j_tile_idx >= J) continue;
                
                if (d_idx == 0) {
                    const float* r_vec = r_tile + j_tile_idx * D;
                    float inv_lj = 1.0f / lj_tile[j_tile_idx];

                    const float4* q_vec_f4 = (const float4*)q_vec;
                    const float4* r_vec_f4 = (const float4*)r_vec;
                    const float4* s_vec_f4 = (const float4*)s_vec;
                    float dot = 0.0f;
                    #pragma unroll
                    for (int d4 = 0; d4 < D / 4; ++d4) {
                        float4 q = q_vec_f4[d4];
                        float4 r = r_vec_f4[d4];
                        float4 s = s_vec_f4[d4];
                        dot += q.x * r.x * s.x + q.y * r.y * s.y + q.z * r.z * s.z + q.w * r.w * s.w;
                    }
                    float logit = dot * scale;
                    float aq_val = expf(logit - current_mi) * inv_li;
                    float ar_val = expf(logit - mj_tile[j_tile_idx]) * inv_lj;
                    attn_scores_tile[k_local_idx] = aq_val * ar_val;
                }
                __syncthreads();

                const float* vq_vec = vq_tile + i_tile_idx * D;
                const float* vr_vec = vr_tile + j_tile_idx * D;
                float combined_attn_val = attn_scores_tile[k_local_idx];
                
                if (d_idx < D) {
                    o_tile[k_local_idx * D + d_idx] += combined_attn_val * vq_vec[d_idx] * vr_vec[d_idx];
                }
                __syncthreads();
            }
        }
    }
    
    // --- Write final result to global memory ---
    if (valid_thread && d_idx < D) {
        float final_val = o_tile[k_local_idx * D + d_idx];
        atomicAdd(&Y_s_[ys_bh_offset + k_idx * D + d_idx], final_val);
    }
}


void Ys_scatter_launcher_with_stats(
    const at::Tensor& Q, const at::Tensor& R, const at::Tensor& S,
    const at::Tensor& Vq_2, const at::Tensor& Vr_2,
    const at::Tensor& m_i, const at::Tensor& l_i,
    const at::Tensor& m_j, const at::Tensor& l_j,
    at::Tensor& Y_s_, float scale
) {
    const auto B = Q.size(0);
    const auto H = Q.size(1);
    const auto I = Q.size(2);
    const auto J = R.size(2);
    const auto K = S.size(2);
    const auto D = Q.size(3);
    
    Y_s_.zero_();

    {
        TORCH_CHECK(D * TILE_K_SCATTER <= 1024, "D * TILE_K_SCATTER must be <= 1024 for the 2D block size.");
        dim3 grid(
            (K + TILE_K_SCATTER - 1) / TILE_K_SCATTER,
            (I + TILE_I - 1) / TILE_I,
            B * H
        );
        dim3 block(D, TILE_K_SCATTER);
        size_t smem_size = sizeof(float) * (
            TILE_K_SCATTER*D + TILE_I*D + TILE_J*D + 
            TILE_I*D + TILE_J*D +             
            TILE_I + TILE_I + TILE_J + TILE_J +
            TILE_K_SCATTER*D 
        );
        Ys_scatter<<<grid, block, smem_size>>>(
            Q.data_ptr<float>(), R.data_ptr<float>(), S.data_ptr<float>(),
            Vq_2.data_ptr<float>(), Vr_2.data_ptr<float>(),
            m_i.data_ptr<float>(), l_i.data_ptr<float>(),
            m_j.data_ptr<float>(), l_j.data_ptr<float>(),
            Y_s_.data_ptr<float>(),
            B, H, I, J, K, D, scale
        );
    }
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

    // GATHER: Y_q (also computes m_i, l_i stats)
    {
        dim3 grid(I, H, B);
        size_t smem_size = sizeof(float) * (D + TILE_J*D + TILE_K*D + TILE_J*D + TILE_K*D + TILE_J*TILE_K + TpB + 2 + D);

        TORCH_CHECK(Q.scalar_type() == at::kFloat, "Only float32 is supported.");
        TORCH_CHECK(D % 4 == 0, "D must be multiple of 4 for FP32 float4 path.");
        Yq_gather<<<grid, block, smem_size>>>(
            reinterpret_cast<const float4*>(Q.data_ptr<float>()),
            reinterpret_cast<const float4*>(R.data_ptr<float>()),
            reinterpret_cast<const float4*>(S.data_ptr<float>()),
            reinterpret_cast<const float4*>(Vr_1.data_ptr<float>()),
            reinterpret_cast<const float4*>(Vs_1.data_ptr<float>()),
            Y_q.data_ptr<float>(),
            m_i.data_ptr<float>(),  // output: softmax stats
            l_i.data_ptr<float>(),  // output: softmax stats
            B, H, I, J, K, D, scale
        );
        AT_CUDA_CHECK(cudaGetLastError());
    }
    
   // GATHER: Y_r (also computes m_j, l_j stats)
    {
        dim3 grid(J, H, B);
        size_t smem_size = sizeof(float) * (D + TILE_I*D + TILE_K*D + TILE_I*D + TILE_K*D + TILE_I*TILE_K + TpB + 2 + D);

        TORCH_CHECK(Q.scalar_type() == at::kFloat, "Only float32 is supported.");
        TORCH_CHECK(D % 4 == 0, "D must be multiple of 4 for FP32 float4 path.");
        Yr_gather<<<grid, block, smem_size>>>(
            reinterpret_cast<const float4*>(R.data_ptr<float>()),
            reinterpret_cast<const float4*>(Q.data_ptr<float>()),
            reinterpret_cast<const float4*>(S.data_ptr<float>()),
            reinterpret_cast<const float4*>(Vq_1.data_ptr<float>()),
            reinterpret_cast<const float4*>(Vs_1.data_ptr<float>()),
            Y_r.data_ptr<float>(),
            m_j.data_ptr<float>(),  // output: softmax stats
            l_j.data_ptr<float>(),  // output: softmax stats
            B, H, I, J, K, D, scale
        );
        AT_CUDA_CHECK(cudaGetLastError());
    }
    // GATHER: Y_s (also computes m_k, l_k stats)
    {
        dim3 grid(K, H, B);
        size_t smem_size = sizeof(float) * (D + TILE_I*D + TILE_J*D + TILE_I*D + TILE_J*D + TILE_I*TILE_J + TpB + 2 + D);

        TORCH_CHECK(Q.scalar_type() == at::kFloat, "Only float32 is supported.");
        TORCH_CHECK(D % 4 == 0, "D must be multiple of 4 for FP32 float4 path.");
        Ys_gather<<<grid, block, smem_size>>>(
            reinterpret_cast<const float4*>(S.data_ptr<float>()),
            reinterpret_cast<const float4*>(Q.data_ptr<float>()),
            reinterpret_cast<const float4*>(R.data_ptr<float>()),
            reinterpret_cast<const float4*>(Vq_1.data_ptr<float>()),
            reinterpret_cast<const float4*>(Vr_1.data_ptr<float>()),
            Y_s.data_ptr<float>(),
            m_k.data_ptr<float>(),  // output: softmax stats
            l_k.data_ptr<float>(),  // output: softmax stats
            B, H, I, J, K, D, scale
        );
        AT_CUDA_CHECK(cudaGetLastError());
    }

    // SCATTER - softmax stats were computed by gather kernels above, reuse them
    Yq_scatter_launcher_with_stats(Q, R, S, Vr_2, Vs_2, m_j, l_j, m_k, l_k, Y_q_, scale);
    Yr_scatter_launcher_with_stats(Q, R, S, Vq_2, Vs_2, m_i, l_i, m_k, l_k, Y_r_, scale);
    Ys_scatter_launcher_with_stats(Q, R, S, Vq_2, Vr_2, m_i, l_i, m_j, l_j, Y_s_, scale);

    cudaDeviceSynchronize(); 
    // Return outputs + softmax stats (for reuse in backward pass)
    return std::make_tuple(Y_q, Y_r, Y_s, Y_q_, Y_r_, Y_s_,
                           m_i, l_i, m_j, l_j, m_k, l_k);}




