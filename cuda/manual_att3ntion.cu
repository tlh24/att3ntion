#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>      
#include <cuda.h>
#include <cuda_runtime.h>
#include "../cpp/manual_att3ntion.h"

// -- Forward Pass --
#define TILE_J 16
#define TILE_K 16
#define TILE_I 16

extern "C" __global__
void Yq_gather_flash_kernel(
    const float4* __restrict__ Q_f4,
    const float4* __restrict__ R_f4,
    const float4* __restrict__ S_f4,
    const float4* __restrict__ V1_f4, // Vr_1
    const float4* __restrict__ V2_f4, // Vs_1
    float*       __restrict__ Y,  // Y_q
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
        m_l_sh[0] = -1e30f; // m_i
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
                    p_tile[flat_idx] = -1e30f; // Make sure padding doesn't affect max
                }
            }
            __syncthreads();

            // --- Find Tile Max (m_ij) ---
            float m_ij = -1e30f;
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
            // Each thread updates its portion of O in shared memory
            for(int d = tid; d < D; d += block_size) {
                float new_o_d = 0.0f;
                // Accumulate new value contribution from the tile. Unroll accumulation.
                float o_accum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                for (int jt = 0; jt < TILE_J; ++jt) {
                    if (j0+jt >= J) continue;
                    for (int kt = 0; kt < TILE_K; kt+=4) { // Unroll innermost loop
                        if (k0+kt >= K) continue;
                        o_accum[0] += p_tile[jt*TILE_K+kt+0] * v1_tile[jt*D+d] * v2_tile[(kt+0)*D+d];
                        o_accum[1] += p_tile[jt*TILE_K+kt+1] * v1_tile[jt*D+d] * v2_tile[(kt+1)*D+d];
                        o_accum[2] += p_tile[jt*TILE_K+kt+2] * v1_tile[jt*D+d] * v2_tile[(kt+2)*D+d];
                        o_accum[3] += p_tile[jt*TILE_K+kt+3] * v1_tile[jt*D+d] * v2_tile[(kt+3)*D+d];
                    }
                }
                new_o_d = o_accum[0] + o_accum[1] + o_accum[2] + o_accum[3];
                
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
}

extern "C" __global__
void Yr_gather_flash_kernel(
    const float4* __restrict__ R_query_f4,
    const float4* __restrict__ Q_f4,
    const float4* __restrict__ S_f4,
    const float4* __restrict__ V1_f4, // Vq_1
    const float4* __restrict__ V2_f4, // Vs_1
    float*       __restrict__ Y,      // Y_r
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
        m_l_sh[0] = -1e30f; // m_i
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
                    p_tile[flat_idx] = -1e30f;
                }
            }
            __syncthreads();

            // --- Find Tile Max (m_ij) ---
            float m_ij = -1e30f;
            for (int flat_idx = tid; flat_idx < TILE_I * TILE_K; flat_idx += block_size) {
                m_ij = fmaxf(m_ij, p_tile[flat_idx] * scale);
            }
            red_buf[tid] = m_ij;
            __syncthreads();
            for (int s = block_size / 2; s > 0; s >>= 1) if (tid < s) red_buf[tid] = fmaxf(red_buf[tid], red_buf[tid+s]);
            __syncthreads();
            m_ij = red_buf[0];

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
            for (int s = block_size / 2; s > 0; s >>= 1) if (tid < s) red_buf[tid] += red_buf[tid+s];
            __syncthreads();
            l_ij = red_buf[0];
            
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
}
extern "C" __global__
void Ys_gather_flash_kernel(
    const float4* __restrict__ S_query_f4,
    const float4* __restrict__ Q_f4,
    const float4* __restrict__ R_f4,
    const float4* __restrict__ V1_f4, // Vq_1
    const float4* __restrict__ V2_f4, // Vr_1
    float*       __restrict__ Y,      // Y_s
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
        m_l_sh[0] = -1e30f; // m_i
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
                    p_tile[flat_idx] = -1e30f;
                }
            }
            __syncthreads();

            // --- Find Tile Max (m_ij) ---
            float m_ij = -1e30f;
            for (int flat_idx = tid; flat_idx < TILE_I * TILE_J; flat_idx += block_size) {
                m_ij = fmaxf(m_ij, p_tile[flat_idx] * scale);
            }
            red_buf[tid] = m_ij;
            __syncthreads();
            for (int s = block_size / 2; s > 0; s >>= 1) if (tid < s) red_buf[tid] = fmaxf(red_buf[tid], red_buf[tid+s]);
            __syncthreads();
            m_ij = red_buf[0];

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
            for (int s = block_size / 2; s > 0; s >>= 1) if (tid < s) red_buf[tid] += red_buf[tid+s];
            __syncthreads();
            l_ij = red_buf[0];
            
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
}

// Ar = softmax_{i,k}(A)
__global__ void Ar_tiled_softmax(
    const float* __restrict__ Q, const float* __restrict__ R, const float* __restrict__ S,
    float* __restrict__ m_j_out, float* __restrict__ l_j_out,
    int B, int H, int I, int J, int K, int D, float scale
) {
    // --- Grid/Block Mapping ---
    // Each block computes the stats for a single j_idx, for a single batch/head.
    const int j_idx = blockIdx.x;
    const int bh_idx = blockIdx.y;
    
    if (j_idx >= J) return;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    // --- Pointers ---
    const int64_t q_bh_offset = (int64_t)bh_idx * I * D;
    const int64_t r_bh_offset = (int64_t)bh_idx * J * D;
    const int64_t s_bh_offset = (int64_t)bh_idx * K * D;

    const float* Q_slice = Q + q_bh_offset;
    const float* R_slice = R + r_bh_offset;
    const float* S_slice = S + s_bh_offset;

    // --- Shared Memory ---
    extern __shared__ float smem[];
    float* r_vec_sh = smem;                              // D
    float* q_tile = r_vec_sh + D;                        // TILE_I * D
    float* s_tile = q_tile + TILE_I * D;                 // TILE_K * D
    float* p_tile = s_tile + TILE_K * D;                 // TILE_I * TILE_K (for intermediate dots)
    float* red_buf = p_tile + TILE_I * TILE_K;           // block_size (for reduction)

    // --- Load R[j] into shared memory ---
    for(int d=tid; d<D; d+=block_size) {
        r_vec_sh[d] = R_slice[j_idx * D + d];
    }
    
    // Each thread holds a local copy of the block-wide stats
    float m_block = -1e30f;
    float l_block = 0.0f;

    __syncthreads();

    // --- Main Loop: Iterate over all i and k tiles ---
    for (int i0 = 0; i0 < I; i0 += TILE_I) {
        for (int k0 = 0; k0 < K; k0 += TILE_K) {
            // Cooperatively load Q and S tiles
            for (int idx = tid; idx < TILE_I * D; idx += block_size) {
                int i_load = i0 + (idx/D);
                if(i_load < I) q_tile[idx] = Q_slice[i_load * D + (idx%D)];
            }
            for (int idx = tid; idx < TILE_K * D; idx += block_size) {
                int k_load = k0 + (idx/D);
                if(k_load < K) s_tile[idx] = S_slice[k_load * D + (idx%D)];
            }
            __syncthreads();

            // --- Compute dot products for this tile and store in p_tile ---
            for (int flat_idx = tid; flat_idx < TILE_I * TILE_K; flat_idx += block_size) {
                int i_tile_idx = flat_idx / TILE_K;
                int k_tile_idx = flat_idx % TILE_K;
                
                float dot = 0.0f;
                if (i0 + i_tile_idx < I && k0 + k_tile_idx < K) {
                    const float* q_vec_sh = q_tile + i_tile_idx * D;
                    const float* s_vec_sh = s_tile + k_tile_idx * D;
                    for(int d=0; d<D; ++d) dot += q_vec_sh[d] * r_vec_sh[d] * s_vec_sh[d];
                    p_tile[flat_idx] = dot * scale;
                } else {
                    p_tile[flat_idx] = -1e30f; // Padding
                }
            }
            __syncthreads();

            // --- Parallel reduction to find tile max (m_tile) ---
            float m_tile_thread = -1e30f;
            for (int flat_idx = tid; flat_idx < TILE_I * TILE_K; flat_idx += block_size) {
                m_tile_thread = fmaxf(m_tile_thread, p_tile[flat_idx]);
            }
            red_buf[tid] = m_tile_thread;
            __syncthreads();
            for (int s = block_size / 2; s > 0; s >>= 1) {
                if (tid < s) red_buf[tid] = fmaxf(red_buf[tid], red_buf[tid + s]);
                __syncthreads();
            }
            float m_tile = red_buf[0];

            // --- Parallel reduction to find tile sum_exp (l_tile) ---
            float l_tile_thread = 0.0f;
            for (int flat_idx = tid; flat_idx < TILE_I * TILE_K; flat_idx += block_size) {
                if (p_tile[flat_idx] > -1e29f) { // Check if it's not padding
                    l_tile_thread += expf(p_tile[flat_idx] - m_tile);
                }
            }
            red_buf[tid] = l_tile_thread;
            __syncthreads();
            for (int s = block_size / 2; s > 0; s >>= 1) {
                if (tid < s) red_buf[tid] += red_buf[tid + s];
                __syncthreads();
            }
            float l_tile = red_buf[0];
            
            // --- Online update of block-wide stats (each thread updates its copy) ---
            float m_new = fmaxf(m_block, m_tile);
            l_block = expf(m_block - m_new) * l_block + expf(m_tile - m_new) * l_tile;
            m_block = m_new;
            __syncthreads(); // Sync to ensure all threads are ready for the next tile
        }
    }
    
    // --- Write final stats to global memory (only thread 0) ---
    if (tid == 0) {
        m_j_out[bh_idx * J + j_idx] = m_block;
        l_j_out[bh_idx * J + j_idx] = l_block;
    }
}

// As = softmax_{i,j}(A)
__global__ void As_tiled_softmax(
    const float* __restrict__ Q, const float* __restrict__ R, const float* __restrict__ S,
    float* __restrict__ m_k_out, float* __restrict__ l_k_out,
    int B, int H, int I, int J, int K, int D, float scale
) {
    // --- Grid/Block Mapping ---
    const int k_idx = blockIdx.x;
    const int bh_idx = blockIdx.y;
    if (k_idx >= K) return;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    // --- Pointers ---
    const int64_t q_bh_offset = (int64_t)bh_idx * I * D;
    const int64_t r_bh_offset = (int64_t)bh_idx * J * D;
    const int64_t s_bh_offset = (int64_t)bh_idx * K * D;

    const float* Q_slice = Q + q_bh_offset;
    const float* R_slice = R + r_bh_offset;
    const float* S_slice = S + s_bh_offset;

    // --- Shared Memory ---
    extern __shared__ float smem[];
    float* s_vec_sh = smem;                              // D
    float* q_tile = s_vec_sh + D;                        // TILE_I * D
    float* r_tile = q_tile + TILE_I * D;                 // TILE_J * D
    float* p_tile = r_tile + TILE_J * D;                 // TILE_I * TILE_J
    float* red_buf = p_tile + TILE_I * TILE_J;           // block_size

    for(int d=tid; d<D; d+=block_size) s_vec_sh[d] = S_slice[k_idx * D + d];
    
    float m_block = -1e30f;
    float l_block = 0.0f;
    __syncthreads();

    // --- Main Loop: Iterate over all i and j tiles ---
    for (int i0 = 0; i0 < I; i0 += TILE_I) {
        for (int j0 = 0; j0 < J; j0 += TILE_J) {
            // Cooperatively load Q and R tiles
            for (int idx = tid; idx < TILE_I * D; idx += block_size) {
                int i_load = i0 + (idx/D);
                if(i_load < I) q_tile[idx] = Q_slice[i_load * D + (idx%D)];
            }
            for (int idx = tid; idx < TILE_J * D; idx += block_size) {
                int j_load = j0 + (idx/D);
                if(j_load < J) r_tile[idx] = R_slice[j_load * D + (idx%D)];
            }
            __syncthreads();

            // Compute dot products
            for (int flat_idx = tid; flat_idx < TILE_I * TILE_J; flat_idx += block_size) {
                int i_tile_idx = flat_idx / TILE_J;
                int j_tile_idx = flat_idx % TILE_J;
                float dot = 0.0f;
                if (i0 + i_tile_idx < I && j0 + j_tile_idx < J) {
                    const float* q_vec_sh = q_tile + i_tile_idx * D;
                    const float* r_vec_sh = r_tile + j_tile_idx * D;
                    for(int d=0; d<D; ++d) dot += q_vec_sh[d] * r_vec_sh[d] * s_vec_sh[d];
                    p_tile[flat_idx] = dot * scale;
                } else {
                    p_tile[flat_idx] = -1e30f;
                }
            }
            __syncthreads();

            // Find tile max (m_tile)
            float m_tile_thread = -1e30f;
            for (int i = tid; i < TILE_I*TILE_J; i+=block_size) m_tile_thread = fmaxf(m_tile_thread, p_tile[i]);
            red_buf[tid] = m_tile_thread;
            __syncthreads();
            for (int s=block_size/2; s>0; s>>=1) if (tid<s) red_buf[tid] = fmaxf(red_buf[tid], red_buf[tid+s]);
            __syncthreads();
            float m_tile = red_buf[0];

            // Find tile sum_exp (l_tile)
            float l_tile_thread = 0.0f;
            for (int i = tid; i < TILE_I*TILE_J; i+=block_size) {
                if(p_tile[i] > -1e29f) l_tile_thread += expf(p_tile[i] - m_tile);
            }
            red_buf[tid] = l_tile_thread;
            __syncthreads();
            for (int s=block_size/2; s>0; s>>=1) if (tid<s) red_buf[tid] += red_buf[tid+s];
            __syncthreads();
            float l_tile = red_buf[0];

            // Online update
            float m_new = fmaxf(m_block, m_tile);
            l_block = expf(m_block - m_new) * l_block + expf(m_tile - m_new) * l_tile;
            m_block = m_new;
            __syncthreads();
        }
    }
    
    if (tid == 0) {
        m_k_out[bh_idx * K + k_idx] = m_block;
        l_k_out[bh_idx * K + k_idx] = l_block;
    }
}

// Aq = softmax_{j,k}(A)
__global__ void Aq_tiled_softmax(
    const float* __restrict__ Q, const float* __restrict__ R, const float* __restrict__ S,
    float* __restrict__ m_i_out, float* __restrict__ l_i_out,
    int B, int H, int I, int J, int K, int D, float scale
) {
    // --- Grid/Block Mapping ---
    // Each block computes the stats for a single i_idx, for a single batch/head.
    const int i_idx = blockIdx.x;
    const int bh_idx = blockIdx.y;
    
    if (i_idx >= I) return;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    // --- Pointers ---
    const int64_t q_bh_offset = (int64_t)bh_idx * I * D;
    const int64_t r_bh_offset = (int64_t)bh_idx * J * D;
    const int64_t s_bh_offset = (int64_t)bh_idx * K * D;

    const float* Q_slice = Q + q_bh_offset;
    const float* R_slice = R + r_bh_offset;
    const float* S_slice = S + s_bh_offset;

    // --- Shared Memory ---
    extern __shared__ float smem[];
    float* q_vec_sh = smem;                              // D
    float* r_tile = q_vec_sh + D;                        // TILE_J * D
    float* s_tile = r_tile + TILE_J * D;                 // TILE_K * D
    float* p_tile = s_tile + TILE_K * D;                 // TILE_J * TILE_K
    float* red_buf = p_tile + TILE_J * TILE_K;           // block_size

    // --- Load Q[i] into shared memory ---
    for(int d=tid; d<D; d+=block_size) {
        q_vec_sh[d] = Q_slice[i_idx * D + d];
    }
    
    // Each thread holds a local copy of the block-wide stats
    float m_block = -1e30f;
    float l_block = 0.0f;

    __syncthreads();

    // --- Main Loop: Iterate over all j and k tiles ---
    for (int j0 = 0; j0 < J; j0 += TILE_J) {
        for (int k0 = 0; k0 < K; k0 += TILE_K) {
            // Cooperatively load R and S tiles
            for (int idx = tid; idx < TILE_J * D; idx += block_size) {
                int j_load = j0 + (idx/D);
                if(j_load < J) r_tile[idx] = R_slice[j_load * D + (idx%D)];
            }
            for (int idx = tid; idx < TILE_K * D; idx += block_size) {
                int k_load = k0 + (idx/D);
                if(k_load < K) s_tile[idx] = S_slice[k_load * D + (idx%D)];
            }
            __syncthreads();

            // --- Compute dot products for this tile and store in p_tile ---
            for (int flat_idx = tid; flat_idx < TILE_J * TILE_K; flat_idx += block_size) {
                int j_tile_idx = flat_idx / TILE_K;
                int k_tile_idx = flat_idx % TILE_K;
                
                float dot = 0.0f;
                if (j0 + j_tile_idx < J && k0 + k_tile_idx < K) {
                    const float* r_vec_sh = r_tile + j_tile_idx * D;
                    const float* s_vec_sh = s_tile + k_tile_idx * D;
                    for(int d=0; d<D; ++d) dot += q_vec_sh[d] * r_vec_sh[d] * s_vec_sh[d];
                    p_tile[flat_idx] = dot * scale;
                } else {
                    p_tile[flat_idx] = -1e30f; // Padding
                }
            }
            __syncthreads();

            // --- Parallel reduction to find tile max (m_tile) ---
            float m_tile_thread = -1e30f;
            for (int flat_idx = tid; flat_idx < TILE_J * TILE_K; flat_idx += block_size) {
                m_tile_thread = fmaxf(m_tile_thread, p_tile[flat_idx]);
            }
            red_buf[tid] = m_tile_thread;
            __syncthreads();
            for (int s = block_size / 2; s > 0; s >>= 1) {
                if (tid < s) red_buf[tid] = fmaxf(red_buf[tid], red_buf[tid + s]);
                __syncthreads();
            }
            float m_tile = red_buf[0];

            // --- Parallel reduction to find tile sum_exp (l_tile) ---
            float l_tile_thread = 0.0f;
            for (int flat_idx = tid; flat_idx < TILE_J * TILE_K; flat_idx += block_size) {
                if (p_tile[flat_idx] > -1e29f) { // Check if it's not padding
                    l_tile_thread += expf(p_tile[flat_idx] - m_tile);
                }
            }
            red_buf[tid] = l_tile_thread;
            __syncthreads();
            for (int s = block_size / 2; s > 0; s >>= 1) {
                if (tid < s) red_buf[tid] += red_buf[tid + s];
                __syncthreads();
            }
            float l_tile = red_buf[0];
            
            // --- Online update of block-wide stats (each thread updates its copy) ---
            float m_new = fmaxf(m_block, m_tile);
            l_block = expf(m_block - m_new) * l_block + expf(m_tile - m_new) * l_tile;
            m_block = m_new;
            __syncthreads(); 
        }
    }
    
    // --- Write final stats to global memory (only thread 0) ---
    if (tid == 0) {
        m_i_out[bh_idx * I + i_idx] = m_block;
        l_i_out[bh_idx * I + i_idx] = l_block;
    }
}

__global__ void Yq_scatter_flash(
    const float* __restrict__ Q, const float* __restrict__ R, const float* __restrict__ S,
    const float* __restrict__ Vr_2, const float* __restrict__ Vs_2,
    const float* __restrict__ m_j_in, const float* __restrict__ l_j_in,
    const float* __restrict__ m_k_in, const float* __restrict__ l_k_in,
    float* __restrict__ Y_q_,
    int B, int H, int I, int J, int K, int D, float scale
) {
    // --- Grid/Block Mapping ---
    const int i_tile_idx_grid = blockIdx.x;
    const int j_tile_idx_grid = blockIdx.y;
    const int bh_idx = blockIdx.z;

    const int d_idx = threadIdx.x;
    const int i_local_idx = threadIdx.y;

    const int i_base = i_tile_idx_grid * TILE_I;
    const int i_idx = i_base + i_local_idx;

    const int j0 = j_tile_idx_grid * TILE_J;

    // Early exit for blocks outside the valid problem space
    if (i_idx >= I || j0 >= J) return;

    // --- Pointers ---
    const int64_t q_bh_offset = (int64_t)bh_idx * I * D;
    const int64_t r_bh_offset = (int64_t)bh_idx * J * D;
    const int64_t s_bh_offset = (int64_t)bh_idx * K * D;
    const int64_t vr_bh_offset = (int64_t)bh_idx * J * D;
    const int64_t vs_bh_offset = (int64_t)bh_idx * K * D;
    const int64_t mj_bh_offset = (int64_t)bh_idx * J;
    const int64_t mk_bh_offset = (int64_t)bh_idx * K;
    const int64_t yq_bh_offset = (int64_t)bh_idx * I * D;

    // --- Shared Memory Layout ---
    extern __shared__ float smem[];

    float* q_tile = (float*)smem;
    float* r_tile = q_tile + TILE_I * D;
    float* s_tile = r_tile + TILE_J * D;
    float* vr_tile = s_tile + TILE_K * D;
    float* vs_tile = vr_tile + TILE_J * D;

    float* mj_tile = vs_tile + TILE_K * D;
    float* lj_tile = mj_tile + TILE_J;
    float* mk_tile = lj_tile + TILE_J;
    float* lk_tile = mk_tile + TILE_K;

    float* o_tile = lk_tile + TILE_K;
    float* attn_scores_tile = o_tile + TILE_I * D; // New: one score per 'i' in the tile
    
    int flat_thread_id_2d = threadIdx.y * blockDim.x + threadIdx.x;
    int threads_per_block = blockDim.x * blockDim.y;
    
    o_tile[i_local_idx * D + d_idx] = 0.0f;

    // --- Load Q tile for this block shaped (TILE_I, D)---
    for (int load_idx = flat_thread_id_2d; load_idx < TILE_I * D; load_idx += threads_per_block) {
        int row_in_tile = load_idx / D;
        int col_in_tile = load_idx % D;
        int i_global = i_base + row_in_tile;
        if (i_global < I) { // ensure threads don't read data out of bounds
            q_tile[row_in_tile * D + col_in_tile] = 
                Q[q_bh_offset + i_global * D + col_in_tile];
        }
    }
    __syncthreads();

    // --- Cooperative loading for j-related tiles shaped (TILE_J, D) ---
    for (int load_idx = flat_thread_id_2d; load_idx < TILE_J * D; load_idx += threads_per_block) {
        int row_in_tile = load_idx / D;
        int col_in_tile = load_idx % D;
        int j_global = j0 + row_in_tile;
        if (j_global < J) {
            r_tile[row_in_tile * D + col_in_tile] = R[r_bh_offset + j_global * D + col_in_tile];
            vr_tile[row_in_tile * D + col_in_tile] = Vr_2[vr_bh_offset + j_global * D + col_in_tile];
        }
    }

    // load mj and lj tiles shaped (TILE_J)
    for (int j_load = flat_thread_id_2d; j_load < TILE_J; j_load += threads_per_block) {
        if (j0 + j_load < J) {
             mj_tile[j_load] = m_j_in[mj_bh_offset + j0 + j_load];
             lj_tile[j_load] = l_j_in[mj_bh_offset + j0 + j_load];
        }
    }

    for (int k0 = 0; k0 < K; k0 += TILE_K) { // for each S of length tile K
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
        const float* q_vec = q_tile + i_local_idx * D;

        for (int j_tile_idx = 0; j_tile_idx < TILE_J; ++j_tile_idx) {
            if (j0 + j_tile_idx >= J) continue;
            const float* r_vec = r_tile + j_tile_idx * D;
            float inv_lj = 1.0f / lj_tile[j_tile_idx];
            float current_mj = mj_tile[j_tile_idx];

            for (int k_tile_idx = 0; k_tile_idx < TILE_K; ++k_tile_idx) {
                if (k0 + k_tile_idx >= K) continue;
                
                // --- Step 2.1: Scalar Computation by a Single Thread per 'i' ---
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

                    float ar_val = expf(logit - current_mj) * inv_lj;
                float as_val = expf(logit - mk_tile[k_tile_idx]) * inv_lk;
                
                    // Store the combined scalar attention score in our new shared memory buffer
                    attn_scores_tile[i_local_idx] = ar_val * as_val;
                }
                
                // --- Step 2.2: Synchronize Block ---
                // Wait for all scalar computations to finish and be visible to all threads.
                __syncthreads();
                
                // --- Step 2.3: Parallel Vector Update ---
                // All D threads for this 'i' read the same score and update their slice of the output.
                const float* vr_vec = vr_tile + j_tile_idx * D;
                const float* vs_vec = vs_tile + k_tile_idx * D;
                float combined_attn_val = attn_scores_tile[i_local_idx];
                
                if (d_idx < D) {
                    o_tile[i_local_idx * D + d_idx] += combined_attn_val * vr_vec[d_idx] * vs_vec[d_idx];
            }

                // --- Step 2.4: Synchronize Block Again ---
                // This is crucial to prevent a race condition where the next 'k' iteration
                // overwrites attn_scores_tile before all threads from the current 'k' are done reading it.
        __syncthreads();
            }
        }
        // __syncthreads(); // This one from your original code is now redundant because of the sync inside the k-loop
    }
    
    // --- Write final result to global memory ---
    // Use atomicAdd to safely accumulate results from different j-tile blocks
    float final_val = o_tile[i_local_idx * D + d_idx];
    if (d_idx < D) atomicAdd(&Y_q_[yq_bh_offset + i_idx * D + d_idx], final_val);
}

void Yq_scatter_flash_launcher(
    const at::Tensor& Q, const at::Tensor& R, const at::Tensor& S,
    const at::Tensor& Vr_2, const at::Tensor& Vs_2,
    at::Tensor& Y_q_, float scale
) {
    const auto B = Q.size(0);
    const auto H = Q.size(1);
    const auto I = Q.size(2);
    const auto J = R.size(2);
    const auto K = S.size(2);
    const auto D = Q.size(3);
    auto opts = Q.options();

    auto m_j = torch::zeros({B, H, J}, opts);
    auto l_j = torch::zeros({B, H, J}, opts);
    auto m_k = torch::zeros({B, H, K}, opts);
    auto l_k = torch::zeros({B, H, K}, opts);
    
    // Zero the output tensor for atomic accumulation
    Y_q_.zero_();

    const int stats_threads = 256;
    { // Ar stats
        dim3 grid(J, B * H);
        dim3 block(stats_threads);
        size_t smem_size = sizeof(float) * (D + TILE_I*D + TILE_K*D + TILE_I*TILE_K + stats_threads);
        Ar_tiled_softmax<<<grid, block, smem_size>>>(
            Q.data_ptr<float>(), R.data_ptr<float>(), S.data_ptr<float>(),
            m_j.data_ptr<float>(), l_j.data_ptr<float>(),
            B, H, I, J, K, D, scale);
    }
    { // As stats
        dim3 grid(K, B * H);
        dim3 block(stats_threads);
        size_t smem_size = sizeof(float) * (D + TILE_I*D + TILE_J*D + TILE_I*TILE_J + stats_threads);
        As_tiled_softmax<<<grid, block, smem_size>>>(
            Q.data_ptr<float>(), R.data_ptr<float>(), S.data_ptr<float>(),
            m_k.data_ptr<float>(), l_k.data_ptr<float>(),
            B, H, I, J, K, D, scale);
    }

    {
        TORCH_CHECK(D * TILE_I <= 1024, "D * TILE_I must be <= 1024 for the 2D block size.");
        // Change grid to be 3D: (I_tiles, J_tiles, B*H)
        dim3 grid(
            (I + TILE_I - 1) / TILE_I,
            (J + TILE_J - 1) / TILE_J,
            B * H
        );
        dim3 block(D, TILE_I);
        size_t smem_size = sizeof(float) * (
            TILE_I*D + TILE_J*D + TILE_K*D + 
            TILE_J*D + TILE_K*D +             
            TILE_J + TILE_J + TILE_K + TILE_K +
            TILE_I*D // Add memory for the o_tile accumulator
        );
        Yq_scatter_flash<<<grid, block, smem_size>>>(
            Q.data_ptr<float>(), R.data_ptr<float>(), S.data_ptr<float>(),
            Vr_2.data_ptr<float>(), Vs_2.data_ptr<float>(),
            m_j.data_ptr<float>(), l_j.data_ptr<float>(),
            m_k.data_ptr<float>(), l_k.data_ptr<float>(),
            Y_q_.data_ptr<float>(),
            B, H, I, J, K, D, scale
        );
    }
}


__global__ void Yr_scatter_flash(
    const float* __restrict__ Q, const float* __restrict__ R, const float* __restrict__ S,
    const float* __restrict__ Vq_2, const float* __restrict__ Vs_2,
    const float* __restrict__ m_i_in, const float* __restrict__ l_i_in,
    const float* __restrict__ m_k_in, const float* __restrict__ l_k_in,
    float* __restrict__ Y_r_,
    int B, int H, int I, int J, int K, int D, float scale
) {
    // --- Grid/Block Mapping ---
    const int j_tile_idx_grid = blockIdx.x;
    const int bh_idx = blockIdx.y;

    const int d_idx = threadIdx.x;
    const int j_local_idx = threadIdx.y;

    const int j_base = j_tile_idx_grid * TILE_J;
    const int j_idx = j_base + j_local_idx;

    if (j_idx >= J) return;

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
    float4* r_tile_f4 = (float4*)smem;
    float4* q_tile_f4 = r_tile_f4 + TILE_J * (D / 4);
    float4* s_tile_f4 = q_tile_f4 + TILE_I * (D / 4);
    float4* vq_tile_f4 = s_tile_f4 + TILE_K * (D / 4);
    float4* vs_tile_f4 = vq_tile_f4 + TILE_I * (D / 4);
    float* mi_tile = (float*)(vs_tile_f4 + TILE_K * (D / 4));
    float* li_tile = mi_tile + TILE_I;
    float* mk_tile = li_tile + TILE_I;
    float* lk_tile = mk_tile + TILE_K;
    float* o_tile = lk_tile + TILE_K; 

    int flat_thread_id_2d = threadIdx.y * blockDim.x + threadIdx.x;
    int threads_per_block = blockDim.x * blockDim.y;
    
    // Initialize the shared memory accumulator tile
    o_tile[j_local_idx * D + d_idx] = 0.0f;

    // --- Load R tile for this block using float4 ---
    for (int load_idx_f4 = flat_thread_id_2d; load_idx_f4 < TILE_J * (D / 4); load_idx_f4 += threads_per_block) {
        int row_in_tile = load_idx_f4 / (D / 4);
        int col_in_f4 = load_idx_f4 % (D / 4);
        int j_global = j_base + row_in_tile;
        if (j_global < J) {
            r_tile_f4[row_in_tile * (D / 4) + col_in_f4] = 
                ((const float4*)(R + r_bh_offset))[j_global * (D / 4) + col_in_f4];
        }
    }
    __syncthreads();

    for (int i0 = 0; i0 < I; i0 += TILE_I) {
        // --- Cooperative loading for i-related tiles ---
        for (int load_idx_f4 = flat_thread_id_2d; load_idx_f4 < TILE_I * (D / 4); load_idx_f4 += threads_per_block) {
            int row_in_tile = load_idx_f4 / (D / 4);
            int col_in_f4 = load_idx_f4 % (D / 4);
            int i_global = i0 + row_in_tile;
            if (i_global < I) {
                q_tile_f4[row_in_tile * (D / 4) + col_in_f4] = ((const float4*)(Q + q_bh_offset))[i_global * (D / 4) + col_in_f4];
                vq_tile_f4[row_in_tile * (D / 4) + col_in_f4] = ((const float4*)(Vq_2 + vq_bh_offset))[i_global * (D / 4) + col_in_f4];
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
            for (int load_idx_f4 = flat_thread_id_2d; load_idx_f4 < TILE_K * (D / 4); load_idx_f4 += threads_per_block) {
                int row_in_tile = load_idx_f4 / (D / 4);
                int col_in_f4 = load_idx_f4 % (D / 4);
                int k_global = k0 + row_in_tile;
                if (k_global < K) {
                    s_tile_f4[row_in_tile * (D / 4) + col_in_f4] = ((const float4*)(S + s_bh_offset))[k_global * (D / 4) + col_in_f4];
                    vs_tile_f4[row_in_tile * (D / 4) + col_in_f4] = ((const float4*)(Vs_2 + vs_bh_offset))[k_global * (D / 4) + col_in_f4];
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
            const float4* r_vec_f4 = r_tile_f4 + j_local_idx * (D / 4);

            for (int i_tile_idx = 0; i_tile_idx < TILE_I; ++i_tile_idx) {
                if (i0 + i_tile_idx >= I) continue;
                const float4* q_vec_f4 = q_tile_f4 + i_tile_idx * (D / 4);
                float inv_li = 1.0f / li_tile[i_tile_idx];

                for (int k_tile_idx = 0; k_tile_idx < TILE_K; ++k_tile_idx) {
                    if (k0 + k_tile_idx >= K) continue;
                    const float4* s_vec_f4 = s_tile_f4 + k_tile_idx * (D / 4);
                    float inv_lk = 1.0f / lk_tile[k_tile_idx];

                    float dot = 0.0f;
                    #pragma unroll
                    for (int d4 = 0; d4 < D / 4; ++d4) {
                        float4 q = q_vec_f4[d4];
                        float4 r = r_vec_f4[d4];
                        float4 s = s_vec_f4[d4];
                        dot += q.x * r.x * s.x + q.y * r.y * s.y + q.z * r.z * s.z + q.w * r.w * s.w;
                    }
                    float logit = dot * scale;

                    float aq_val = expf(logit - mi_tile[i_tile_idx]) * inv_li;
                    float as_val = expf(logit - mk_tile[k_tile_idx]) * inv_lk;
                    
                    o_tile[j_local_idx * D + d_idx] += aq_val * as_val * ((float*)vq_tile_f4)[i_tile_idx*D + d_idx] * ((float*)vs_tile_f4)[k_tile_idx*D + d_idx];
                }
            }
            __syncthreads();
        }
    }
    
    // --- Write final result to global memory ---
    if (d_idx < D) Y_r_[yr_bh_offset + j_idx * D + d_idx] = o_tile[j_local_idx * D + d_idx];
}


void Yr_scatter_flash_launcher(
    const at::Tensor& Q, const at::Tensor& R, const at::Tensor& S,
    const at::Tensor& Vq_2, const at::Tensor& Vs_2,
    at::Tensor& Y_r_, float scale
) {
    const auto B = Q.size(0);
    const auto H = Q.size(1);
    const auto I = Q.size(2);
    const auto J = R.size(2);
    const auto K = S.size(2);
    const auto D = Q.size(3);
    auto opts = Q.options();

    auto m_i = torch::zeros({B, H, I}, opts);
    auto l_i = torch::zeros({B, H, I}, opts);
    auto m_k = torch::zeros({B, H, K}, opts);
    auto l_k = torch::zeros({B, H, K}, opts);
    
    const int stats_threads = 256;
    { // Aq stats
        dim3 grid(I, B * H);
        dim3 block(stats_threads);
        size_t smem_size = sizeof(float) * (D + TILE_J*D + TILE_K*D + TILE_J*TILE_K + stats_threads);
        Aq_tiled_softmax<<<grid, block, smem_size>>>(
            Q.data_ptr<float>(), R.data_ptr<float>(), S.data_ptr<float>(),
            m_i.data_ptr<float>(), l_i.data_ptr<float>(),
            B, H, I, J, K, D, scale);
    }
    { // As stats
        dim3 grid(K, B * H);
        dim3 block(stats_threads);
        size_t smem_size = sizeof(float) * (D + TILE_I*D + TILE_J*D + TILE_I*TILE_J + stats_threads);
        As_tiled_softmax<<<grid, block, smem_size>>>(
            Q.data_ptr<float>(), R.data_ptr<float>(), S.data_ptr<float>(),
            m_k.data_ptr<float>(), l_k.data_ptr<float>(),
            B, H, I, J, K, D, scale);
    }

    {
        TORCH_CHECK(D * TILE_J <= 1024, "D * TILE_J must be <= 1024 for the 2D block size.");
        dim3 grid((J + TILE_J - 1) / TILE_J, B * H);
        dim3 block(D, TILE_J);
        size_t smem_size = sizeof(float) * (
            TILE_J*D + TILE_I*D + TILE_K*D + 
            TILE_I*D + TILE_K*D +             
            TILE_I + TILE_I + TILE_K + TILE_K +
            TILE_J*D 
        );
        Yr_scatter_flash<<<grid, block, smem_size>>>(
            Q.data_ptr<float>(), R.data_ptr<float>(), S.data_ptr<float>(),
            Vq_2.data_ptr<float>(), Vs_2.data_ptr<float>(),
            m_i.data_ptr<float>(), l_i.data_ptr<float>(),
            m_k.data_ptr<float>(), l_k.data_ptr<float>(),
            Y_r_.data_ptr<float>(),
            B, H, I, J, K, D, scale
        );
    }
}


__global__ void Ys_scatter_flash(
    const float* __restrict__ Q, const float* __restrict__ R, const float* __restrict__ S,
    const float* __restrict__ Vq_2, const float* __restrict__ Vr_2,
    const float* __restrict__ m_i_in, const float* __restrict__ l_i_in,
    const float* __restrict__ m_j_in, const float* __restrict__ l_j_in,
    float* __restrict__ Y_s_,
    int B, int H, int I, int J, int K, int D, float scale
) {
    // --- Grid/Block Mapping ---
    const int k_tile_idx_grid = blockIdx.x;
    const int bh_idx = blockIdx.y;

    const int d_idx = threadIdx.x;
    const int k_local_idx = threadIdx.y;

    const int k_base = k_tile_idx_grid * TILE_K;
    const int k_idx = k_base + k_local_idx;

    if (k_idx >= K) return;

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
    float4* s_tile_f4 = (float4*)smem;
    float4* q_tile_f4 = s_tile_f4 + TILE_K * (D / 4);
    float4* r_tile_f4 = q_tile_f4 + TILE_I * (D / 4);
    float4* vq_tile_f4 = r_tile_f4 + TILE_J * (D / 4);
    float4* vr_tile_f4 = vq_tile_f4 + TILE_I * (D / 4);
    float* mi_tile = (float*)(vr_tile_f4 + TILE_J * (D / 4));
    float* li_tile = mi_tile + TILE_I;
    float* mj_tile = li_tile + TILE_I;
    float* lj_tile = mj_tile + TILE_J;
    float* o_tile = lj_tile + TILE_J; 

    int flat_thread_id_2d = threadIdx.y * blockDim.x + threadIdx.x;
    int threads_per_block = blockDim.x * blockDim.y;
    
    o_tile[k_local_idx * D + d_idx] = 0.0f;

    // --- Load S tile for this block using float4 ---
    for (int load_idx_f4 = flat_thread_id_2d; load_idx_f4 < TILE_K * (D / 4); load_idx_f4 += threads_per_block) {
        int row_in_tile = load_idx_f4 / (D / 4);
        int col_in_f4 = load_idx_f4 % (D / 4);
        int k_global = k_base + row_in_tile;
        if (k_global < K) {
            s_tile_f4[row_in_tile * (D / 4) + col_in_f4] = 
                ((const float4*)(S + s_bh_offset))[k_global * (D / 4) + col_in_f4];
        }
    }
    __syncthreads();

    for (int i0 = 0; i0 < I; i0 += TILE_I) {
        // --- Cooperative loading for i-related tiles ---
        for (int load_idx_f4 = flat_thread_id_2d; load_idx_f4 < TILE_I * (D / 4); load_idx_f4 += threads_per_block) {
            int row_in_tile = load_idx_f4 / (D / 4);
            int i_global = i0 + row_in_tile;
            if (i_global < I) {
                q_tile_f4[load_idx_f4] = ((const float4*)(Q + q_bh_offset))[i_global * (D / 4) + (load_idx_f4 % (D/4))];
                vq_tile_f4[load_idx_f4] = ((const float4*)(Vq_2 + vq_bh_offset))[i_global * (D / 4) + (load_idx_f4 % (D/4))];
            }
        }
        for (int i_load = flat_thread_id_2d; i_load < TILE_I; i_load += threads_per_block) {
            if (i0 + i_load < I) {
                 mi_tile[i_load] = m_i_in[mi_bh_offset + i0 + i_load];
                 li_tile[i_load] = l_i_in[mi_bh_offset + i0 + i_load];
            }
        }

        for (int j0 = 0; j0 < J; j0 += TILE_J) {
            // --- Cooperative loading for j-related tiles ---
            for (int load_idx_f4 = flat_thread_id_2d; load_idx_f4 < TILE_J * (D / 4); load_idx_f4 += threads_per_block) {
                int row_in_tile = load_idx_f4 / (D / 4);
                int j_global = j0 + row_in_tile;
                if (j_global < J) {
                    r_tile_f4[load_idx_f4] = ((const float4*)(R + r_bh_offset))[j_global * (D / 4) + (load_idx_f4 % (D/4))];
                    vr_tile_f4[load_idx_f4] = ((const float4*)(Vr_2 + vr_bh_offset))[j_global * (D / 4) + (load_idx_f4 % (D/4))];
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
            const float4* s_vec_f4 = s_tile_f4 + k_local_idx * (D / 4);

            for (int i_tile_idx = 0; i_tile_idx < TILE_I; ++i_tile_idx) {
                if (i0 + i_tile_idx >= I) continue;
                const float4* q_vec_f4 = q_tile_f4 + i_tile_idx * (D / 4);
                float inv_li = 1.0f / li_tile[i_tile_idx];

                for (int j_tile_idx = 0; j_tile_idx < TILE_J; ++j_tile_idx) {
                    if (j0 + j_tile_idx >= J) continue;
                    const float4* r_vec_f4 = r_tile_f4 + j_tile_idx * (D / 4);
                    float inv_lj = 1.0f / lj_tile[j_tile_idx];

                    float dot = 0.0f;
                    #pragma unroll
                    for (int d4 = 0; d4 < D / 4; ++d4) {
                        float4 q = q_vec_f4[d4];
                        float4 r = r_vec_f4[d4];
                        float4 s = s_vec_f4[d4];
                        dot += q.x * r.x * s.x + q.y * r.y * s.y + q.z * r.z * s.z + q.w * r.w * s.w;
                    }
                    float logit = dot * scale;

                    float aq_val = expf(logit - mi_tile[i_tile_idx]) * inv_li;
                    float ar_val = expf(logit - mj_tile[j_tile_idx]) * inv_lj;
                    
                    o_tile[k_local_idx * D + d_idx] += aq_val * ar_val * ((float*)vq_tile_f4)[i_tile_idx*D + d_idx] * ((float*)vr_tile_f4)[j_tile_idx*D + d_idx];
                }
            }
            __syncthreads();
        }
    }
    
    // --- Write final result to global memory ---
    if (d_idx < D) Y_s_[ys_bh_offset + k_idx * D + d_idx] = o_tile[k_local_idx * D + d_idx];
}


void Ys_scatter_flash_launcher(
    const at::Tensor& Q, const at::Tensor& R, const at::Tensor& S,
    const at::Tensor& Vq_2, const at::Tensor& Vr_2,
    at::Tensor& Y_s_, float scale
) {
    const auto B = Q.size(0);
    const auto H = Q.size(1);
    const auto I = Q.size(2);
    const auto J = R.size(2);
    const auto K = S.size(2);
    const auto D = Q.size(3);
    auto opts = Q.options();

    auto m_i = torch::zeros({B, H, I}, opts);
    auto l_i = torch::zeros({B, H, I}, opts);
    auto m_j = torch::zeros({B, H, J}, opts);
    auto l_j = torch::zeros({B, H, J}, opts);
    
    const int stats_threads = 256;
    { // Aq stats
        dim3 grid(I, B * H);
        dim3 block(stats_threads);
        size_t smem_size = sizeof(float) * (D + TILE_J*D + TILE_K*D + TILE_J*TILE_K + stats_threads);
        Aq_tiled_softmax<<<grid, block, smem_size>>>(
            Q.data_ptr<float>(), R.data_ptr<float>(), S.data_ptr<float>(),
            m_i.data_ptr<float>(), l_i.data_ptr<float>(),
            B, H, I, J, K, D, scale);
    }
    { // Ar stats
        dim3 grid(J, B * H);
        dim3 block(stats_threads);
        size_t smem_size = sizeof(float) * (D + TILE_I*D + TILE_K*D + TILE_I*TILE_K + stats_threads);
        Ar_tiled_softmax<<<grid, block, smem_size>>>(
            Q.data_ptr<float>(), R.data_ptr<float>(), S.data_ptr<float>(),
            m_j.data_ptr<float>(), l_j.data_ptr<float>(),
            B, H, I, J, K, D, scale);
    }

    {
        TORCH_CHECK(D * TILE_K <= 1024, "D * TILE_K must be <= 1024 for the 2D block size.");
        dim3 grid((K + TILE_K - 1) / TILE_K, B * H);
        dim3 block(D, TILE_K);
        size_t smem_size = sizeof(float) * (
            TILE_K*D + TILE_I*D + TILE_J*D + 
            TILE_I*D + TILE_J*D +             
            TILE_I + TILE_I + TILE_J + TILE_J +
            TILE_K*D 
        );
        Ys_scatter_flash<<<grid, block, smem_size>>>(
            Q.data_ptr<float>(), R.data_ptr<float>(), S.data_ptr<float>(),
            Vq_2.data_ptr<float>(), Vr_2.data_ptr<float>(),
            m_i.data_ptr<float>(), l_i.data_ptr<float>(),
            m_j.data_ptr<float>(), l_j.data_ptr<float>(),
            Y_s_.data_ptr<float>(),
            B, H, I, J, K, D, scale
        );
    }
}

__global__ void Yq_gather_kernel_naive(
    const float* __restrict__ Q, 
    const float* __restrict__ R, 
    const float* __restrict__ S,
    const float* __restrict__ V1, 
    const float* __restrict__ V2,
    float*       __restrict__ Y,
    int B, int H, int I, int J, int K, int D,
    float scale){
    // global thread index [0 .. B*H*I*D)
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)B*H*I*D) return;
    
    // decode (b,h,i,d) from idx 
    int d = idx % D; 
    int tmp = idx / D;
    int i = tmp % I;
    tmp = tmp / I;
    int h = tmp % H;
    int b = tmp / H;

    // pointers into the 4D tensors:
    // Q[b,h,i,:]   at Q + (((b*H + h)*I + i)*D)
    const float* q_base = Q + (((b*H + h)*I + i) * D);
    float y_val = 0.0f;

    // --- 1) find max for numerical‐stable softmax over (j,k) ---
    float max_dot = -1e30f;
    for(int j=0; j < J; ++j){
      const float* r_base = R + (((b*H + h)*J + j) * D);
      for(int k=0; k < K; ++k){
        const float* s_base = S + (((b*H + h)*K + k) * D);
        // dot = sum_d q_base[d]*r_base[d]*s_base[d]
        float dot = 0.0f;
        #pragma unroll 4
        for(int dd=0; dd < D; ++dd)
          dot += q_base[dd] * r_base[dd] * s_base[dd];
        max_dot = fmaxf(max_dot, dot * scale);
      }
    }

    // --- 2) compute denom ---
    float sum_exp = 0.0f;
    for(int j=0; j < J; ++j){
      const float* r_base = R + (((b*H + h)*J + j) * D);
      for(int k=0; k < K; ++k){
        const float* s_base = S + (((b*H + h)*K + k) * D);
        float dot = 0.0f;
        #pragma unroll 4
        for(int dd=0; dd < D; ++dd)
          dot += q_base[dd] * r_base[dd] * s_base[dd];
        sum_exp += expf(dot*scale - max_dot);
      }
    }

    // --- 3) accumulate Y[b,h,i,d] = sum_jk softmax * V1[b,h,j,d]*V2[b,h,k,d] ---
    float inv_denom = 1.0f / sum_exp;
    for(int j=0; j < J; ++j){
      const float* r_base = R + (((b*H + h)*J + j) * D);
      const float* v1_base = V1 + (((b*H + h)*J + j) * D);
      for(int k=0; k < K; ++k){
        const float* s_base  = S  + (((b*H + h)*K + k) * D);
        const float* v2_base = V2 + (((b*H + h)*K + k) * D);
        float dot = 0.0f;
        #pragma unroll 4
        for(int dd=0; dd < D; ++dd)
          dot += q_base[dd] * r_base[dd] * s_base[dd];
        float attn = expf(dot*scale - max_dot) * inv_denom;
        y_val += attn * v1_base[d] * v2_base[d];
      }
    }
    Y[idx] = y_val;}




__global__ void Yr_gather_kernel_naive(
    const float* __restrict__ R_query, // R is the 'query' for this dimension
    const float* __restrict__ Q,
    const float* __restrict__ S,
    const float* __restrict__ V1,      // Vq_1
    const float* __restrict__ V2,      // Vs_1
    float*       __restrict__ Y,       // Y_r
    int B, int H, int I, int J, int K, int D,
    float scale){
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)B*H*J*D) return;

    int d = idx % D;
    int tmp = idx / D;
    int j = tmp % J;
    tmp = tmp / J;
    int h = tmp % H;
    int b = tmp / H;

    // pointers into the 4D tensors:
    //   R_query[b,h,j,:] at R_query + (((b*H + h)*J + j)*D)
    const float* r_base_query = R_query + (((b*H + h)*J + j) * D);
    float y_val = 0.0f;

    // --- 1) find max softmax over (i,k) ---
    float max_dot = -1e30f;
    for(int i=0; i < I; ++i){
      const float* q_base = Q + (((b*H + h)*I + i) * D);
      for(int k=0; k < K; ++k){
        const float* s_base = S + (((b*H + h)*K + k) * D);
        // dot = sum_d r_base_query[d]*q_base[d]*s_base[d]
        float dot = 0.0f;
        #pragma unroll 4
        for(int dd=0; dd < D; ++dd)
          dot += r_base_query[dd] * q_base[dd] * s_base[dd];
        max_dot = fmaxf(max_dot, dot * scale);
      }
    }

    // --- 2) compute denom ---
    float sum_exp = 0.0f;
    for(int i=0; i < I; ++i){
      const float* q_base = Q + (((b*H + h)*I + i) * D);
      for(int k=0; k < K; ++k){
        const float* s_base = S + (((b*H + h)*K + k) * D);
        float dot = 0.0f;
        #pragma unroll 4
        for(int dd=0; dd < D; ++dd)
          dot += r_base_query[dd] * q_base[dd] * s_base[dd];
        sum_exp += expf(dot*scale - max_dot);
      }
    }

    // --- 3) accumulate Y[b,h,j,d] = sum_ik softmax * V1[b,h,i,d]*V2[b,h,k,d] ---
    float inv_denom = 1.0f / sum_exp;
    for(int i=0; i < I; ++i){
      const float* q_base = Q + (((b*H + h)*I + i) * D);
      const float* v1_base = V1 + (((b*H + h)*I + i) * D); // V1 corresponds to Vq_1 (dim I)
      for(int k=0; k < K; ++k){
        const float* s_base  = S  + (((b*H + h)*K + k) * D);
        const float* v2_base = V2 + (((b*H + h)*K + k) * D); // V2 corresponds to Vs_1 (dim K)
        float dot = 0.0f;
        #pragma unroll 4
        for(int dd=0; dd < D; ++dd)
          dot += r_base_query[dd] * q_base[dd] * s_base[dd];
        float attn = expf(dot*scale - max_dot) * inv_denom;
        y_val += attn * v1_base[d] * v2_base[d];
      }
    }

    Y[idx] = y_val;}
__global__ void Ys_gather_kernel_naive(
    const float* __restrict__ S_query, // S is the 'query' for this dimension
    const float* __restrict__ Q,
    const float* __restrict__ R,
    const float* __restrict__ V1,      // Vq_1
    const float* __restrict__ V2,      // Vr_1
    float*       __restrict__ Y,       // Y_s
    int B, int H, int I, int J, int K, int D,
    float scale){
    // global thread index [0 .. B*H*K*D)
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)B*H*K*D) return;

    // decode (b,h,k,d) from idx
    int d = idx % D;
    int tmp = idx / D;
    int k = tmp % K;
    tmp = tmp / K;
    int h = tmp % H;
    int b = tmp / H;

    // pointers into the 4D tensors:
    //   S_query[b,h,k,:] at S_query + (((b*H + h)*K + k)*D)
    const float* s_base_query = S_query + (((b*H + h)*K + k) * D);
    float y_val = 0.0f;

    // --- 1) find max for softmax over (i,j) ---
    float max_dot = -1e30f;
    for(int i=0; i < I; ++i){
      const float* q_base = Q + (((b*H + h)*I + i) * D);
      for(int j=0; j < J; ++j){
        const float* r_base = R + (((b*H + h)*J + j) * D);
        // dot = sum_d s_base_query[d]*q_base[d]*r_base[d]
        float dot = 0.0f;
        #pragma unroll 4
        for(int dd=0; dd < D; ++dd)
          dot += s_base_query[dd] * q_base[dd] * r_base[dd];
        max_dot = fmaxf(max_dot, dot * scale);
      }
    }

    // --- 2) compute denom ---
    float sum_exp = 0.0f;
    for(int i=0; i < I; ++i){
      const float* q_base = Q + (((b*H + h)*I + i) * D);
      for(int j=0; j < J; ++j){
        const float* r_base = R + (((b*H + h)*J + j) * D);
        float dot = 0.0f;
        #pragma unroll 4
        for(int dd=0; dd < D; ++dd)
          dot += s_base_query[dd] * q_base[dd] * r_base[dd];
        sum_exp += expf(dot*scale - max_dot);
      }
    }

    // --- 3) accumulate Y[b,h,k,d] = sum_ij softmax * V1[b,h,i,d]*V2[b,h,j,d] ---
    float inv_denom = 1.0f / sum_exp;
    for(int i=0; i < I; ++i){
      const float* q_base = Q + (((b*H + h)*I + i) * D);
      const float* v1_base = V1 + (((b*H + h)*I + i) * D); // V1 corresponds to Vq_1 (dim I)
      for(int j=0; j < J; ++j){
        const float* r_base  = R  + (((b*H + h)*J + j) * D);
        const float* v2_base = V2 + (((b*H + h)*J + j) * D); // V2 corresponds to Vr_1 (dim J)
        float dot = 0.0f;
        #pragma unroll 4
        for(int dd=0; dd < D; ++dd)
          dot += s_base_query[dd] * q_base[dd] * r_base[dd];
        float attn = expf(dot*scale - max_dot) * inv_denom;
        y_val += attn * v1_base[d] * v2_base[d];
      }
    }

    Y[idx] = y_val;}

__device__ inline float compute_dot_product_cuda(
    const float* __restrict__ Q, 
    const float* __restrict__ R, 
    const float* __restrict__ S,
    int b, int h, int i, int j, int k, 
    int B, int H, int I, int J, int K, int D){
    // Calculate base pointers using passed dimensions
    const float* q_vec = Q + (((int64_t)b * H + h) * I + i) * D;
    const float* r_vec = R + (((int64_t)b * H + h) * J + j) * D;
    const float* s_vec = S + (((int64_t)b * H + h) * K + k) * D;

    float dot = 0.0f;
    #pragma unroll 4
    for (int d = 0; d < D; ++d) {
        dot += q_vec[d] * r_vec[d] * s_vec[d];
    }
    return dot;}

__device__ inline float compute_single_softmax_attn_cuda(
    const float* __restrict__ Q, 
    const float* __restrict__ R, 
    const float* __restrict__ S,
    int b, int h, int i_target, int j_target, int k_target,
    int B, int H, int I, int J, int K, int D,
    float scale,
    int fixed_dim ){
    float max_val = -1.0e30f;
    float sum_exp = 0.0f;

    // --- First Pass: Find Max --- 
    if (fixed_dim == 0) { //Aq
        for (int j_idx = 0; j_idx < J; ++j_idx) {
            for (int k_idx = 0; k_idx < K; ++k_idx) {
                float dot = compute_dot_product_cuda(Q, R, S, b, h, i_target, j_idx, k_idx, B, H, I, J, K, D);
                max_val = fmaxf(max_val, dot * scale);
            }
        }
    }
    else if (fixed_dim == 1) { // Ar
        for (int i_idx = 0; i_idx < I; ++i_idx) {
            for (int k_idx = 0; k_idx < K; ++k_idx) {
                float dot = compute_dot_product_cuda(Q, R, S, b, h, i_idx, j_target, k_idx, B, H, I, J, K, D);
                max_val = fmaxf(max_val, dot * scale);
            }
        }
    } else { // fixed_dim == 2
        for (int i_idx = 0; i_idx < I; ++i_idx) {
            for (int j_idx = 0; j_idx < J; ++j_idx) {
                float dot = compute_dot_product_cuda(Q, R, S, b, h, i_idx, j_idx, k_target, B, H, I, J, K, D);
                max_val = fmaxf(max_val, dot * scale);
            }
        }
    }

    // --- Second Pass: Compute Sum Exp ---
    if (fixed_dim == 0) { // Aq
        for (int j_idx = 0; j_idx < J; ++j_idx) {
            for (int k_idx = 0; k_idx < K; ++k_idx) {
                float dot = compute_dot_product_cuda(Q, R, S, b, h, i_target, j_idx, k_idx, B, H, I, J, K, D);
                sum_exp += expf(dot * scale - max_val);
            }
        }
    }
    else if (fixed_dim == 1) { // Ar
        for (int i_idx = 0; i_idx < I; ++i_idx) {
            for (int k_idx = 0; k_idx < K; ++k_idx) {
                float dot = compute_dot_product_cuda(Q, R, S, b, h, i_idx, j_target, k_idx, B, H, I, J, K, D);
                sum_exp += expf(dot * scale - max_val);
            }
        }
    } else { // As
        for (int i_idx = 0; i_idx < I; ++i_idx) {
            for (int j_idx = 0; j_idx < J; ++j_idx) {
                float dot = compute_dot_product_cuda(Q, R, S, b, h, i_idx, j_idx, k_target, B, H, I, J, K, D);
                sum_exp += expf(dot * scale - max_val);
            }
        }
    }

    // --- Compute final value for the target indices ---
    float target_dot = compute_dot_product_cuda(Q, R, S, b, h, i_target, j_target, k_target, B, H, I, J, K, D);
    
    // Handle potential division by zero if sum_exp is very small
    if (sum_exp <= 1e-20f) {
        int num_elements;
        if (fixed_dim == 0) num_elements = J * K;
        else if (fixed_dim == 1) num_elements = I * K;
        else /* fixed_dim == 2 */ num_elements = I * J;
        // If the target dot was also the max (or close), return uniform prob, else 0.
        return (fabsf(target_dot * scale - max_val) < 1e-5f) ? (1.0f / (float)num_elements) : 0.0f;
    }
    
    return expf(target_dot * scale - max_val) / sum_exp;}




__global__ void Yq_scatter_kernel_naive(
    const float* Q,
    const float* R,
    const float* S,
    const float* Vr_2, 
    const float* Vs_2, 
    float*       Y_q_, 
    int B, int H, int I, int J, int K, int D,
    float scale){
    // global thread index [0 .. B*H*I*D)
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)B*H*I*D) return;

    // decode (b,h,i,d) from idx - this thread computes Y_q_[b,h,i,d]
    int d = idx % D;
    int tmp = idx / D;
    int i = tmp % I;
    tmp = tmp / I;
    int h = tmp % H;
    int b = tmp / H;

    float accum_val = 0.0f;

    // Iterate over the dimensions we sum over (j and k)
    for (int j = 0; j < J; ++j) {
        const float* vr2_vec = Vr_2 + (((int64_t)b * H + h) * J + j) * D;
        for (int k = 0; k < K; ++k) {
            const float* vs2_vec = Vs_2 + (((int64_t)b * H + h) * K + k) * D;

            // Ar[b,h,i,j,k] (softmax over i',k' for fixed j)
            float attn_ar = compute_single_softmax_attn_cuda(Q, R, S, b, h, i, j, k, B, H, I, J, K, D, scale, 1);
            
            //As[b,h,i,j,k] (softmax over i',j' for fixed k)
            float attn_as = compute_single_softmax_attn_cuda(Q, R, S, b, h, i, j, k, B, H, I, J, K, D, scale, 2);

            // Get value components for the specific d
            float vr2_val = vr2_vec[d];
            float vs2_val = vs2_vec[d];

            // Ar * As * Vr_2 * Vs_2
            accum_val += attn_ar * attn_as * vr2_val * vs2_val;
        }
    }

    Y_q_[idx] = accum_val;}
__global__ void Yq_scatter_kernel_optimized(
    const float* __restrict__ Ar_slice, 
    const float* __restrict__ As_slice,
    const float* __restrict__ Vr_2_slice, 
    const float* __restrict__ Vs_2_slice, 
    float*       __restrict__ Y_q__slice, 
    int I, int J, int K, int D) {
    // Each thread computes one element of the output slice Y_q_'. Grid is [I*D].
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)I * D) return;

    int d = idx % D;
    int i = idx / D;

    float accum_val = 0.0f;

    // Sum over j and k
    for (int j = 0; j < J; ++j) {
        const float* vr2_vec = Vr_2_slice + (int64_t)j * D;
        for (int k = 0; k < K; ++k) {
            const float* vs2_vec = Vs_2_slice + (int64_t)k * D;
            
            int64_t ijk_idx = (int64_t)i * J * K + (int64_t)j * K + k;
            float attn_ar = Ar_slice[ijk_idx];
            float attn_as = As_slice[ijk_idx];

            float vr2_val = vr2_vec[d];
            float vs2_val = vs2_vec[d];

            accum_val += attn_ar * attn_as * vr2_val * vs2_val;
        }
    }
    Y_q__slice[idx] = accum_val;}

__global__ void Yr_scatter_kernel_optimized(
    const float* __restrict__ Aq_slice, 
    const float* __restrict__ As_slice,
    const float* __restrict__ Vq_2_slice, 
    const float* __restrict__ Vs_2_slice, 
    float*       __restrict__ Y_r__slice, 
    int I, int J, int K, int D){
    // Each thread computes one element of the output slice Y_r_'. Grid is [J*D].
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)J * D) return;

    int d = idx % D;
    int j = idx / D;

    float accum_val = 0.0f;

    // Sum over i and k
    for (int i = 0; i < I; ++i) {
        const float* vq2_vec = Vq_2_slice + (int64_t)i * D;
        for (int k = 0; k < K; ++k) {
            const float* vs2_vec = Vs_2_slice + (int64_t)k * D;

            int64_t ijk_idx = (int64_t)i * J * K + (int64_t)j * K + k;
            float attn_aq = Aq_slice[ijk_idx];
            float attn_as = As_slice[ijk_idx];

            float vq2_val = vq2_vec[d];
            float vs2_val = vs2_vec[d];

            accum_val += attn_aq * attn_as * vq2_val * vs2_val;
        }
    }
    Y_r__slice[idx] = accum_val;}

__global__ void Ys_scatter_kernel_optimized(
    const float* __restrict__ Aq_slice, 
    const float* __restrict__ Ar_slice,
    const float* __restrict__ Vq_2_slice, 
    const float* __restrict__ Vr_2_slice, 
    float*       __restrict__ Y_s__slice, 
    int I, int J, int K, int D){
    // Each thread computes one element of the output slice Y_s_'. Grid is [K*D].
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)K * D) return;

    int d = idx % D;
    int k = idx / D;

    float accum_val = 0.0f;

    // Sum over i and j
    for (int i = 0; i < I; ++i) {
        const float* vq2_vec = Vq_2_slice + (int64_t)i * D;
        for (int j = 0; j < J; ++j) {
            const float* vr2_vec = Vr_2_slice + (int64_t)j * D;

            int64_t ijk_idx = (int64_t)i * J * K + (int64_t)j * K + k;
            float attn_aq = Aq_slice[ijk_idx];
            float attn_ar = Ar_slice[ijk_idx];

            float vq2_val = vq2_vec[d];
            float vr2_val = vr2_vec[d];

            accum_val += attn_aq * attn_ar * vq2_val * vr2_val;
        }
    }
    Y_s__slice[idx] = accum_val;}
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> forward_cuda(
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

    const int threads = 256;

    // GATHER
    TORCH_CHECK(D % 4 == 0, "Head dimension D must be a multiple of 4 for the flash kernel.");
    const int TpB = 128; // Threads per block
    // Y_q Gather
    {
        dim3 grid(I, H, B);
        dim3 block(TpB);
        size_t smem_size = sizeof(float) * (D + TILE_J*D + TILE_K*D + TILE_J*D + TILE_K*D + TILE_J*TILE_K + TpB + 2 + D);

        Yq_gather_flash_kernel<<<grid, block, smem_size>>>(
            reinterpret_cast<const float4*>(Q.data_ptr<float>()), 
            reinterpret_cast<const float4*>(R.data_ptr<float>()), 
            reinterpret_cast<const float4*>(S.data_ptr<float>()),
            reinterpret_cast<const float4*>(Vr_1.data_ptr<float>()), 
            reinterpret_cast<const float4*>(Vs_1.data_ptr<float>()),
            Y_q.data_ptr<float>(), 
            B, H, I, J, K, D, scale);
    }
    // Y_r Gather
    {
        dim3 grid(J, H, B);
        dim3 block(TpB);
        size_t smem_size = sizeof(float) * (D + TILE_I*D + TILE_K*D + TILE_I*D + TILE_K*D + TILE_I*TILE_K + TpB + 2 + D);
        Yr_gather_flash_kernel<<<grid, block, smem_size>>>(
            reinterpret_cast<const float4*>(R.data_ptr<float>()), 
            reinterpret_cast<const float4*>(Q.data_ptr<float>()), 
            reinterpret_cast<const float4*>(S.data_ptr<float>()), 
            reinterpret_cast<const float4*>(Vq_1.data_ptr<float>()), 
            reinterpret_cast<const float4*>(Vs_1.data_ptr<float>()), 
            Y_r.data_ptr<float>(), 
            B, H, I, J, K, D, scale);
    }
    // Y_s Gather
    {
        dim3 grid(K, H, B);
        dim3 block(TpB);
        size_t smem_size = sizeof(float) * (D + TILE_I*D + TILE_J*D + TILE_I*D + TILE_J*D + TILE_I*TILE_J + TpB + 2 + D);
        Ys_gather_flash_kernel<<<grid, block, smem_size>>>(
            reinterpret_cast<const float4*>(S.data_ptr<float>()), 
            reinterpret_cast<const float4*>(Q.data_ptr<float>()), 
            reinterpret_cast<const float4*>(R.data_ptr<float>()),
            reinterpret_cast<const float4*>(Vq_1.data_ptr<float>()), 
            reinterpret_cast<const float4*>(Vr_1.data_ptr<float>()), 
            Y_s.data_ptr<float>(), 
            B, H, I, J, K, D, scale);
    }

    // SCATTER 
    Yq_scatter_flash_launcher(Q, R, S, Vr_2, Vs_2, Y_q_, scale);
    Yr_scatter_flash_launcher(Q, R, S, Vq_2, Vs_2, Y_r_, scale);
    Ys_scatter_flash_launcher(Q, R, S, Vq_2, Vr_2, Y_s_, scale);

    cudaDeviceSynchronize(); 
    return std::make_tuple(Y_q, Y_r, Y_s, Y_q_, Y_r_, Y_s_);}




// --- Backward Pass ---

__global__ void gather_grad_Vq1_kernel_optimized(
    const float* __restrict__ gradY_slice,     // [N, D]
    const float* __restrict__ Vr_1_slice,      // [J, D]
    const float* __restrict__ Vs_1_slice,      // [K, D]
    const float* __restrict__ Ar_slice,        // [I, J, K]
    const float* __restrict__ As_slice,        // [I, J, K]
    float*       __restrict__ gradVq1_slice_out, // [I, D]
    int I, int J, int K, int D, int N_grad){
    // Grid of I*D threads, each computes one element of the output gradVq1_slice
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)I * D) return;

    int d = idx % D;
    int i_target = idx / D;

    float grad_accum = 0.0f;

    // --- 1. Contribution from Y_r path (dL/dY_r) ---
    // dL/dVq_1[i] += sum_{j} [ dL/dY_r[j,d] * sum_{k} ( Ar[i,j,k] * Vs_1[k,d] ) ]
    for (int j = 0; j < J; ++j) {
        if (j >= N_grad) continue;
        float dy_r = gradY_slice[(int64_t)j * D + d];
        if (dy_r == 0.0f) continue;

        float inner_sum_k = 0.0f;
        for (int k = 0; k < K; ++k) {
            int64_t ijk_idx = (int64_t)i_target * J * K + (int64_t)j * K + k;
            inner_sum_k += Ar_slice[ijk_idx] * Vs_1_slice[(int64_t)k * D + d];
        }
        grad_accum += dy_r * inner_sum_k;
    }

    // --- 2. Contribution from Y_s path (dL/dY_s) ---
    // dL/dVq_1[i] += sum_{k} [ dL/dY_s[k,d] * sum_{j} ( As[i,j,k] * Vr_1[j,d] ) ]
    for (int k = 0; k < K; ++k) {
        if (k >= N_grad) continue;
        float dy_s = gradY_slice[(int64_t)k * D + d];
        if (dy_s == 0.0f) continue;
        
        float inner_sum_j = 0.0f;
        for (int j = 0; j < J; ++j) {
            int64_t ijk_idx = (int64_t)i_target * J * K + (int64_t)j * K + k;
            inner_sum_j += As_slice[ijk_idx] * Vr_1_slice[(int64_t)j * D + d];
        }
        grad_accum += dy_s * inner_sum_j;
    }

    gradVq1_slice_out[idx] = grad_accum;}

__global__ void gather_grad_Vr1_kernel_optimized(
    const float* __restrict__ gradY_slice,       // [N, D]
    const float* __restrict__ Vq_1_slice,        // [I, D]
    const float* __restrict__ Vs_1_slice,        // [K, D]
    const float* __restrict__ Aq_slice,          // [I, J, K]
    const float* __restrict__ As_slice,          // [I, J, K]
    float*       __restrict__ gradVr1_slice_out,   // [J, D]
    int I, int J, int K, int D, int N_grad){
    // Grid of J*D threads
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)J * D) return;

    int d = idx % D;
    int j_target = idx / D;

    float grad_accum = 0.0f;

    // --- 1. Contribution from Y_q path (dL/dY_q) ---
    for (int i = 0; i < I; ++i) {
        if (i >= N_grad) continue;
        float dy_q = gradY_slice[(int64_t)i * D + d];
        if (dy_q == 0.0f) continue;
        
        float inner_sum_k = 0.0f;
        for (int k = 0; k < K; ++k) {
            int64_t ijk_idx = (int64_t)i * J * K + (int64_t)j_target * K + k;
            inner_sum_k += Aq_slice[ijk_idx] * Vs_1_slice[(int64_t)k * D + d];
        }
        grad_accum += dy_q * inner_sum_k;
    }

    // --- 2. Contribution from Y_s path (dL/dY_s) ---
    for (int k = 0; k < K; ++k) {
        if (k >= N_grad) continue;
        float dy_s = gradY_slice[(int64_t)k * D + d];
        if (dy_s == 0.0f) continue;
        
        float inner_sum_i = 0.0f;
        for (int i = 0; i < I; ++i) {
            int64_t ijk_idx = (int64_t)i * J * K + (int64_t)j_target * K + k;
            inner_sum_i += As_slice[ijk_idx] * Vq_1_slice[(int64_t)i * D + d];
        }
        grad_accum += dy_s * inner_sum_i;
    }

    gradVr1_slice_out[idx] = grad_accum;}

__global__ void gather_grad_Vs1_kernel_optimized(
    const float* __restrict__ gradY_slice,     // [N, D]
    const float* __restrict__ Vq_1_slice,      // [I, D]
    const float* __restrict__ Vr_1_slice,      // [J, D]
    const float* __restrict__ Aq_slice,        // [I, J, K]
    const float* __restrict__ Ar_slice,        // [I, J, K]
    float*       __restrict__ gradVs1_slice_out, // [K, D]
    int I, int J, int K, int D, int N_grad){
    // Grid of K*D threads
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)K * D) return;

    int d = idx % D;
    int k_target = idx / D;

    float grad_accum = 0.0f;

    // --- 1. Contribution from Y_q path (dL/dY_q) ---
    for (int i = 0; i < I; ++i) {
        if (i >= N_grad) continue;
        float dy_q = gradY_slice[(int64_t)i * D + d];
        if (dy_q == 0.0f) continue;

        float inner_sum_j = 0.0f;
        for (int j = 0; j < J; ++j) {
            int64_t ijk_idx = (int64_t)i * J * K + (int64_t)j * K + k_target;
            inner_sum_j += Aq_slice[ijk_idx] * Vr_1_slice[(int64_t)j * D + d];
        }
        grad_accum += dy_q * inner_sum_j;
    }

    // --- 2. Contribution from Y_r path (dL/dY_r) ---
    for (int j = 0; j < J; ++j) {
        if (j >= N_grad) continue;
        float dy_r = gradY_slice[(int64_t)j * D + d];
        if (dy_r == 0.0f) continue;

        float inner_sum_i = 0.0f;
        for (int i = 0; i < I; ++i) {
            int64_t ijk_idx = (int64_t)i * J * K + (int64_t)j * K + k_target;
            inner_sum_i += Ar_slice[ijk_idx] * Vq_1_slice[(int64_t)i * D + d];
        }
        grad_accum += dy_r * inner_sum_i;
    }

    gradVs1_slice_out[idx] = grad_accum;}

__global__ void scatter_grad_Vq2_kernel_optimized(
    const float* __restrict__ gradY_slice,
    const float* __restrict__ Aq_slice,
    const float* __restrict__ Ar_slice,
    const float* __restrict__ As_slice,
    const float* __restrict__ Vr_2_slice,
    const float* __restrict__ Vs_2_slice,
    float*       __restrict__ gradVq2_slice_out,
    int I, int J, int K, int D, int N_grad) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)I * D) return;

    int d = idx % D;
    int i_target = idx / D;

    float grad_accum = 0.0f;

    // Term 1: from dL/dY_r'
    for (int j = 0; j < J; ++j) {
        if (j >= N_grad) continue;
        float dy_r = gradY_slice[(int64_t)j * D + d];
        if (dy_r == 0.0f) continue;

        for (int k = 0; k < K; ++k) {
            int64_t ijk_idx = (int64_t)i_target * J * K + (int64_t)j * K + k;
            grad_accum += dy_r * Aq_slice[ijk_idx] * As_slice[ijk_idx] * Vs_2_slice[(int64_t)k * D + d];
        }
    }

    // Term 2: from dL/dY_s'
    for (int k = 0; k < K; ++k) {
        if (k >= N_grad) continue;
        float dy_s = gradY_slice[(int64_t)k * D + d];
        if (dy_s == 0.0f) continue;
        
        for (int j = 0; j < J; ++j) {
            int64_t ijk_idx = (int64_t)i_target * J * K + (int64_t)j * K + k;
            grad_accum += dy_s * Aq_slice[ijk_idx] * Ar_slice[ijk_idx] * Vr_2_slice[(int64_t)j * D + d];
        }
    }

    gradVq2_slice_out[idx] = grad_accum;}

__global__ void scatter_grad_Vr2_kernel_optimized(
    const float* __restrict__ gradY_slice,
    const float* __restrict__ Aq_slice,
    const float* __restrict__ Ar_slice,
    const float* __restrict__ As_slice,
    const float* __restrict__ Vq_2_slice,
    const float* __restrict__ Vs_2_slice,
    float*       __restrict__ gradVr2_slice_out,
    int I, int J, int K, int D, int N_grad) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)J * D) return;

    int d = idx % D;
    int j_target = idx / D;

    float grad_accum = 0.0f;

    // Term 1: from dL/dY_q'
    for (int i = 0; i < I; ++i) {
        if (i >= N_grad) continue;
        float dy_q = gradY_slice[(int64_t)i * D + d];
        if (dy_q == 0.0f) continue;

        for (int k = 0; k < K; ++k) {
            int64_t ijk_idx = (int64_t)i * J * K + (int64_t)j_target * K + k;
            grad_accum += dy_q * Ar_slice[ijk_idx] * As_slice[ijk_idx] * Vs_2_slice[(int64_t)k * D + d];
        }
    }

    // Term 2: from dL/dY_s'
    for (int k = 0; k < K; ++k) {
        if (k >= N_grad) continue;
        float dy_s = gradY_slice[(int64_t)k * D + d];
        if (dy_s == 0.0f) continue;
        
        for (int i = 0; i < I; ++i) {
            int64_t ijk_idx = (int64_t)i * J * K + (int64_t)j_target * K + k;
            grad_accum += dy_s * Aq_slice[ijk_idx] * Ar_slice[ijk_idx] * Vq_2_slice[(int64_t)i * D + d];
        }
    }

    gradVr2_slice_out[idx] = grad_accum;}

__global__ void scatter_grad_Vs2_kernel_optimized(
    const float* __restrict__ gradY_slice,
    const float* __restrict__ Aq_slice,
    const float* __restrict__ Ar_slice,
    const float* __restrict__ As_slice,
    const float* __restrict__ Vq_2_slice,
    const float* __restrict__ Vr_2_slice,
    float*       __restrict__ gradVs2_slice_out,
    int I, int J, int K, int D, int N_grad) {
    
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)K * D) return;

    int d = idx % D;
    int k_target = idx / D;

    float grad_accum = 0.0f;

    // Term 1: from dL/dY_q'
    for (int i = 0; i < I; ++i) {
        if (i >= N_grad) continue;
        float dy_q = gradY_slice[(int64_t)i * D + d];
        if (dy_q == 0.0f) continue;

        for (int j = 0; j < J; ++j) {
            int64_t ijk_idx = (int64_t)i * J * K + (int64_t)j * K + k_target;
            grad_accum += dy_q * Ar_slice[ijk_idx] * As_slice[ijk_idx] * Vr_2_slice[(int64_t)j * D + d];
        }
    }

    // Term 2: from dL/dY_r'
    for (int j = 0; j < J; ++j) {
        if (j >= N_grad) continue;
        float dy_r = gradY_slice[(int64_t)j * D + d];
        if (dy_r == 0.0f) continue;

        for (int i = 0; i < I; ++i) {
            int64_t ijk_idx = (int64_t)i * J * K + (int64_t)j * K + k_target;
            grad_accum += dy_r * Aq_slice[ijk_idx] * As_slice[ijk_idx] * Vq_2_slice[(int64_t)i * D + d];
        }
    }

    gradVs2_slice_out[idx] = grad_accum;}

// -- needed for grad_Q, grad_R, grad_S : multi-kernel implementation --
// -- grad A batched gpu helpers --


__global__ void compute_A_slice_kernel(
    const float* __restrict__ Q_slice_global, // Input Q_slice [I,D] on GPU
    const float* __restrict__ R_slice_global, // Input R_slice [J,D] on GPU
    const float* __restrict__ S_slice_global, // Input S_slice [K,D] on GPU
    float*       __restrict__ A_out_global,   // Output A_slice [I,J,K] on GPU
    int I, int J, int K, int D,
    float scale
) {
    // Map 3D block and thread indices to I, J, K
    // This kernel is intended to be launched with a 3D grid/block structure
    // that covers all elements of A_slice [I,J,K].
    // E.g., GridDim could be ( (I+BlockDim.x-1)/BlockDim.x, (J+BlockDim.y-1)/BlockDim.y, (K+BlockDim.z-1)/BlockDim.z )
    // BlockDim could be (DIM_I_THREADS, DIM_J_THREADS, DIM_K_THREADS)
    
    int i_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int j_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int k_idx = blockIdx.z * blockDim.z + threadIdx.z;

    // Boundary check
    if (i_idx >= I || j_idx >= J || k_idx >= K) {
        return;
    }

    // Compute dot product Q_slice[i_idx,:] * R_slice[j_idx,:] * S_slice[k_idx,:]
    float dot_product = 0.0f;
    for (int d_loop = 0; d_loop < D; ++d_loop) {
        // Q_slice_global is [I, D], so access Q_slice_global[i_idx * D + d_loop]
        // R_slice_global is [J, D], so access R_slice_global[j_idx * D + d_loop]
        // S_slice_global is [K, D], so access S_slice_global[k_idx * D + d_loop]
        dot_product += Q_slice_global[i_idx * D + d_loop] * \
                       R_slice_global[j_idx * D + d_loop] * \
                       S_slice_global[k_idx * D + d_loop];
    }

    // Apply scale and store in A_out_global
    // A_out_global is [I,J,K]. Index is i_idx * (J*K) + j_idx * K + k_idx
    A_out_global[i_idx * J * K + j_idx * K + k_idx] = dot_product * scale;
}
// C++ Wrapper for compute_A_slice_kernel
torch::Tensor compute_A_slice_cuda_wrapper(
    const torch::Tensor& Q_slice_gpu, // Assumed to be on GPU, shape [I,D]
    const torch::Tensor& R_slice_gpu, // Assumed to be on GPU, shape [J,D]
    const torch::Tensor& S_slice_gpu, // Assumed to be on GPU, shape [K,D]
    float scale
) {
    TORCH_CHECK(Q_slice_gpu.is_cuda(), "Q_slice_gpu must be a CUDA tensor");
    TORCH_CHECK(R_slice_gpu.is_cuda(), "R_slice_gpu must be a CUDA tensor");
    TORCH_CHECK(S_slice_gpu.is_cuda(), "S_slice_gpu must be a CUDA tensor");

    TORCH_CHECK(Q_slice_gpu.dim() == 2, "Q_slice_gpu must be 2D");
    TORCH_CHECK(R_slice_gpu.dim() == 2, "R_slice_gpu must be 2D");
    TORCH_CHECK(S_slice_gpu.dim() == 2, "S_slice_gpu must be 2D");

    const int I = Q_slice_gpu.size(0);
    const int D_q = Q_slice_gpu.size(1);
    const int J = R_slice_gpu.size(0);
    const int D_r = R_slice_gpu.size(1);
    const int K = S_slice_gpu.size(0);
    const int D_s = S_slice_gpu.size(1);

    TORCH_CHECK(D_q == D_r && D_r == D_s, "Dimension D must match for Q, R, S slices");
    const int D = D_q;

    // Allocate output tensor A_slice_out_gpu [I,J,K] on GPU
    auto options = Q_slice_gpu.options(); // Inherit dtype and device
    torch::Tensor A_slice_out_gpu = torch::zeros({I, J, K}, options);

    // Define block dimensions (e.g., 8x8x8 = 512 threads, adjust as needed)
    // These are example values, can be tuned.
    // Max threads per block is 1024.
    // Block dimensions should ideally be multiples of warp size (32 for NVIDIA GPUs) for one dim.
    constexpr int BLOCK_DIM_I = 8;
    constexpr int BLOCK_DIM_J = 8;
    constexpr int BLOCK_DIM_K = 8; 
    
    dim3 blockDim(BLOCK_DIM_I, BLOCK_DIM_J, BLOCK_DIM_K);

    // Calculate grid dimensions
    dim3 gridDim(
        (I + BLOCK_DIM_I - 1) / BLOCK_DIM_I,
        (J + BLOCK_DIM_J - 1) / BLOCK_DIM_J,
        (K + BLOCK_DIM_K - 1) / BLOCK_DIM_K
    );
    
    // Ensure inputs are contiguous (important for direct pointer access in kernel)
    auto Q_cont = Q_slice_gpu.contiguous();
    auto R_cont = R_slice_gpu.contiguous();
    auto S_cont = S_slice_gpu.contiguous();

    // Launch Kernel
    compute_A_slice_kernel<<<gridDim, blockDim>>>(
        Q_cont.data_ptr<float>(),
        R_cont.data_ptr<float>(),
        S_cont.data_ptr<float>(),
        A_slice_out_gpu.data_ptr<float>(),
        I, J, K, D,
        scale
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in compute_A_slice_cuda_wrapper: %s\n", cudaGetErrorString(err));
        // Optionally throw an exception
        // throw std::runtime_error(std::string("CUDA kernel launch failed in compute_A_slice_cuda_wrapper: ") + cudaGetErrorString(err));
    }
    // cudaDeviceSynchronize(); // For debugging, to ensure kernel finishes before wrapper returns

    return A_slice_out_gpu;
}
// Kernel to compute Aq_slice (softmax over j, k for each fixed i)
// Each block processes one 'i' plane.
__global__ void compute_Aq_slice_kernel(
    const float* __restrict__ A_slice_global, // Input A_slice [I,J,K] (global mem)
    float*       __restrict__ Aq_out_global,  // Output Aq_slice [I,J,K] (global mem)
    int I_dim, int J_dim, int K_dim // Use _dim to avoid conflict with loop vars
) {
    // Shared memory is now only for reduction, not for the entire data plane.
    extern __shared__ float s_reduction_pad[]; 
    
    // Current 'i' index this block is responsible for
    int i_current = blockIdx.x;
    if (i_current >= I_dim) {
        return;
    }

    const float* current_A_plane_global = A_slice_global + (int64_t)i_current * J_dim * K_dim;
    float* current_Aq_plane_global = Aq_out_global + (int64_t)i_current * J_dim * K_dim;

    int plane_size = J_dim * K_dim;
    int tid_in_block = threadIdx.x;
    int threads_in_block = blockDim.x;

    // --- Pass 1: Find max_val in the plane using a parallel reduction ---
    float thread_max_val = -FLT_MAX;
    for (int idx = tid_in_block; idx < plane_size; idx += threads_in_block) {
        thread_max_val = fmaxf(thread_max_val, current_A_plane_global[idx]);
    }
    s_reduction_pad[tid_in_block] = thread_max_val;
    __syncthreads();

    // Reduce to find the max value for the whole plane
    for (int offset = threads_in_block / 2; offset > 0; offset >>= 1) {
        if (tid_in_block < offset) {
            s_reduction_pad[tid_in_block] = fmaxf(s_reduction_pad[tid_in_block], s_reduction_pad[tid_in_block + offset]);
        }
        __syncthreads();
    }
    float plane_max_val = s_reduction_pad[0];
    __syncthreads(); // Ensure all threads see the correct max value

    // --- Pass 2: Compute sum_exp for the plane ---
    float thread_sum_exp = 0.0f;
    for (int idx = tid_in_block; idx < plane_size; idx += threads_in_block) {
        thread_sum_exp += expf(current_A_plane_global[idx] - plane_max_val);
    }
    s_reduction_pad[tid_in_block] = thread_sum_exp;
    __syncthreads();

    // Reduce to find the sum of exponents for the whole plane
    for (int offset = threads_in_block / 2; offset > 0; offset >>= 1) {
        if (tid_in_block < offset) {
            s_reduction_pad[tid_in_block] += s_reduction_pad[tid_in_block + offset];
        }
        __syncthreads();
    }
    float plane_sum_exp = s_reduction_pad[0];
    __syncthreads();

    // --- Pass 3: Compute softmax values and write to global memory ---
    if (plane_sum_exp == 0.0f) plane_sum_exp = 1e-20f; // Avoid division by zero
    for (int idx = tid_in_block; idx < plane_size; idx += threads_in_block) {
        current_Aq_plane_global[idx] = expf(current_A_plane_global[idx] - plane_max_val) / plane_sum_exp;
    }
}


torch::Tensor compute_Aq_slice_cuda_wrapper(
    const torch::Tensor& A_slice_gpu // Assumed to be on GPU, shape [I,J,K]
) {
    TORCH_CHECK(A_slice_gpu.is_cuda(), "A_slice_gpu must be a CUDA tensor");
    TORCH_CHECK(A_slice_gpu.dim() == 3, "A_slice_gpu must be 3D [I,J,K]");

    const int I = A_slice_gpu.size(0);
    const int J = A_slice_gpu.size(1);
    const int K = A_slice_gpu.size(2);

    auto options = A_slice_gpu.options();
    torch::Tensor Aq_slice_out_gpu = torch::zeros({I, J, K}, options);

    dim3 gridDim(I);
    int threads_per_block = 256;
    dim3 blockDim(threads_per_block);

    // Shared memory size is now fixed and small, proportional to block size
    size_t shared_mem_size = threads_per_block * sizeof(float);
    
    // The explicit call to cudaFuncSetAttribute is no longer needed, as the requested
    // shared memory size is small and constant. Passing it in the kernel launch is sufficient.

    auto A_cont = A_slice_gpu.contiguous();

    // Launch Kernel
    compute_Aq_slice_kernel<<<gridDim, blockDim, shared_mem_size>>>(
        A_cont.data_ptr<float>(),
        Aq_slice_out_gpu.data_ptr<float>(),
        I, J, K
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in compute_Aq_slice_cuda_wrapper: %s\n", cudaGetErrorString(err));
    }
    return Aq_slice_out_gpu;
}

// Kernel to compute Ar_slice (softmax over i, k for each fixed j)
// Each block processes one 'j' plane.
__global__ void compute_Ar_slice_kernel(
    const float* __restrict__ A_slice_global, // Input A_slice [I,J,K] (global mem)
    float*       __restrict__ Ar_out_global,  // Output Ar_slice [I,J,K] (global mem)
    int I_dim, int J_dim, int K_dim
) {
    extern __shared__ float s_reduction_pad[]; 
    
    int j_current = blockIdx.x;
    if (j_current >= J_dim) return;

    int plane_size = I_dim * K_dim;
    int tid_in_block = threadIdx.x;
    int threads_in_block = blockDim.x;

    // --- Pass 1: Find max_val ---
    float thread_max_val = -FLT_MAX;
    for (int idx = tid_in_block; idx < plane_size; idx += threads_in_block) {
        int i_load = idx / K_dim;
        int k_load = idx % K_dim;
        int64_t global_idx = (int64_t)i_load * J_dim * K_dim + (int64_t)j_current * K_dim + k_load;
        thread_max_val = fmaxf(thread_max_val, A_slice_global[global_idx]);
    }
    s_reduction_pad[tid_in_block] = thread_max_val;
    __syncthreads();
    for (int offset = threads_in_block / 2; offset > 0; offset >>= 1) {
        if (tid_in_block < offset) s_reduction_pad[tid_in_block] = fmaxf(s_reduction_pad[tid_in_block], s_reduction_pad[tid_in_block + offset]);
        __syncthreads();
    }
    float plane_max_val = s_reduction_pad[0];
    __syncthreads();

    // --- Pass 2: Compute sum_exp ---
    float thread_sum_exp = 0.0f;
    for (int idx = tid_in_block; idx < plane_size; idx += threads_in_block) {
        int i_load = idx / K_dim;
        int k_load = idx % K_dim;
        int64_t global_idx = (int64_t)i_load * J_dim * K_dim + (int64_t)j_current * K_dim + k_load;
        thread_sum_exp += expf(A_slice_global[global_idx] - plane_max_val);
    }
    s_reduction_pad[tid_in_block] = thread_sum_exp;
    __syncthreads();
    for (int offset = threads_in_block / 2; offset > 0; offset >>= 1) {
        if (tid_in_block < offset) s_reduction_pad[tid_in_block] += s_reduction_pad[tid_in_block + offset];
        __syncthreads();
    }
    float plane_sum_exp = s_reduction_pad[0];
    __syncthreads();

    // --- Pass 3: Compute softmax and write ---
    if (plane_sum_exp == 0.0f) plane_sum_exp = 1e-20f;
    for (int idx = tid_in_block; idx < plane_size; idx += threads_in_block) {
        int i_write = idx / K_dim;
        int k_write = idx % K_dim;
        int64_t global_idx = (int64_t)i_write * J_dim * K_dim + (int64_t)j_current * K_dim + k_write;
        Ar_out_global[global_idx] = expf(A_slice_global[global_idx] - plane_max_val) / plane_sum_exp;
    }
}


// Kernel to compute As_slice (softmax over i, j for each fixed k)
// Each block processes one 'k' plane.
__global__ void compute_As_slice_kernel(
    const float* __restrict__ A_slice_global, // Input A_slice [I,J,K] (global mem)
    float*       __restrict__ As_out_global,  // Output As_slice [I,J,K] (global mem)
    int I_dim, int J_dim, int K_dim
) {
    extern __shared__ float s_reduction_pad[];

    int k_current = blockIdx.x;
    if (k_current >= K_dim) return;

    int plane_size = I_dim * J_dim;
    int tid_in_block = threadIdx.x;
    int threads_in_block = blockDim.x;

    // --- Pass 1: Find max_val ---
    float thread_max_val = -FLT_MAX;
    for (int idx = tid_in_block; idx < plane_size; idx += threads_in_block) {
        int i_load = idx / J_dim;
        int j_load = idx % J_dim;
        int64_t global_idx = (int64_t)i_load * J_dim * K_dim + (int64_t)j_load * K_dim + k_current;
        thread_max_val = fmaxf(thread_max_val, A_slice_global[global_idx]);
    }
    s_reduction_pad[tid_in_block] = thread_max_val;
    __syncthreads();
    for (int offset = threads_in_block / 2; offset > 0; offset >>= 1) {
        if (tid_in_block < offset) s_reduction_pad[tid_in_block] = fmaxf(s_reduction_pad[tid_in_block], s_reduction_pad[tid_in_block + offset]);
        __syncthreads();
    }
    float plane_max_val = s_reduction_pad[0];
    __syncthreads();

    // --- Pass 2: Compute sum_exp ---
    float thread_sum_exp = 0.0f;
    for (int idx = tid_in_block; idx < plane_size; idx += threads_in_block) {
        int i_load = idx / J_dim;
        int j_load = idx % J_dim;
        int64_t global_idx = (int64_t)i_load * J_dim * K_dim + (int64_t)j_load * K_dim + k_current;
        thread_sum_exp += expf(A_slice_global[global_idx] - plane_max_val);
    }
    s_reduction_pad[tid_in_block] = thread_sum_exp;
    __syncthreads();
    for (int offset = threads_in_block / 2; offset > 0; offset >>= 1) {
        if (tid_in_block < offset) s_reduction_pad[tid_in_block] += s_reduction_pad[tid_in_block + offset];
        __syncthreads();
    }
    float plane_sum_exp = s_reduction_pad[0];
    __syncthreads();

    // --- Pass 3: Compute softmax and write ---
    if (plane_sum_exp == 0.0f) plane_sum_exp = 1e-20f;
    for (int idx = tid_in_block; idx < plane_size; idx += threads_in_block) {
        int i_write = idx / J_dim;
        int j_write = idx % J_dim;
        int64_t global_idx = (int64_t)i_write * J_dim * K_dim + (int64_t)j_write * K_dim + k_current;
        As_out_global[global_idx] = expf(A_slice_global[global_idx] - plane_max_val) / plane_sum_exp;
    }
}

// C++ Wrapper for compute_Ar_slice_kernel
torch::Tensor compute_Ar_slice_cuda_wrapper(
    const torch::Tensor& A_slice_gpu // Assumed to be on GPU, shape [I,J,K]
) {
    TORCH_CHECK(A_slice_gpu.is_cuda(), "A_slice_gpu must be a CUDA tensor");
    TORCH_CHECK(A_slice_gpu.dim() == 3, "A_slice_gpu must be 3D [I,J,K]");

    const int I = A_slice_gpu.size(0);
    const int J = A_slice_gpu.size(1);
    const int K = A_slice_gpu.size(2);

    auto options = A_slice_gpu.options(); 
    torch::Tensor Ar_slice_out_gpu = torch::zeros({I, J, K}, options);

    dim3 gridDim(J); // J blocks, one for each j-plane

    int threads_per_block = 256;
    dim3 blockDim(threads_per_block); 

    size_t shared_mem_size = threads_per_block * sizeof(float);
    
    auto A_cont = A_slice_gpu.contiguous();

    compute_Ar_slice_kernel<<<gridDim, blockDim, shared_mem_size>>>(
        A_cont.data_ptr<float>(),
        Ar_slice_out_gpu.data_ptr<float>(),
        I, J, K
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in compute_Ar_slice_cuda_wrapper: %s\n", cudaGetErrorString(err));
    }
    return Ar_slice_out_gpu;
}


// C++ Wrapper for compute_As_slice_kernel
torch::Tensor compute_As_slice_cuda_wrapper(
    const torch::Tensor& A_slice_gpu // Assumed to be on GPU, shape [I,J,K]
) {
    TORCH_CHECK(A_slice_gpu.is_cuda(), "A_slice_gpu must be a CUDA tensor");
    TORCH_CHECK(A_slice_gpu.dim() == 3, "A_slice_gpu must be 3D [I,J,K]");

    const int I = A_slice_gpu.size(0);
    const int J = A_slice_gpu.size(1);
    const int K = A_slice_gpu.size(2);

    auto options = A_slice_gpu.options(); 
    torch::Tensor As_slice_out_gpu = torch::zeros({I, J, K}, options);

    dim3 gridDim(K); // K blocks, one for each k-plane

    int threads_per_block = 256;
    dim3 blockDim(threads_per_block); 

    size_t shared_mem_size = threads_per_block * sizeof(float);
    
    auto A_cont = A_slice_gpu.contiguous();

    compute_As_slice_kernel<<<gridDim, blockDim, shared_mem_size>>>(
        A_cont.data_ptr<float>(),
        As_slice_out_gpu.data_ptr<float>(),
        I, J, K
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in compute_As_slice_cuda_wrapper: %s\n", cudaGetErrorString(err));
    }
    return As_slice_out_gpu;
}

// Kernel to compute grad_Aq_slice, grad_Ar_slice, grad_As_slice (Phase 1 of C++ compute_grad_A_single)
__global__ void compute_interim_grads_kernel(
    // Inputs
    const float* __restrict__ grad_output_slice, // [N, D]
    const float* __restrict__ Vq_1_slice,        // [I, D]
    const float* __restrict__ Vq_2_slice,        // [I, D]
    const float* __restrict__ Vr_1_slice,        // [J, D]
    const float* __restrict__ Vr_2_slice,        // [J, D]
    const float* __restrict__ Vs_1_slice,        // [K, D]
    const float* __restrict__ Vs_2_slice,        // [K, D]
    const float* __restrict__ Aq_slice,          // [I, J, K]
    const float* __restrict__ Ar_slice,          // [I, J, K]
    const float* __restrict__ As_slice,          // [I, J, K]
    // Outputs (Temporary)
    float* __restrict__ grad_Aq_slice_out,     // [I, J, K]
    float* __restrict__ grad_Ar_slice_out,     // [I, J, K]
    float* __restrict__ grad_As_slice_out,     // [I, J, K]
    // Dimensions
    int I_dim, int J_dim, int K_dim, int D_dim, int N_dim // Use _dim suffix
) {
    // Map 3D thread indices to (i, j, k)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    // Boundary check
    if (i >= I_dim || j >= J_dim || k >= K_dim) {
        return;
    }
    
    // Calculate linear index for output arrays [I,J,K]
    int64_t ijk_idx = (int64_t)i * J_dim * K_dim + (int64_t)j * K_dim + k;

    float grad_aq_val = 0.0f;
    float grad_ar_val = 0.0f;
    float grad_as_val = 0.0f;

    // Get pointers to relevant rows in V matrices (based on i, j, k)
    const float* vq1_row_i = Vq_1_slice + (int64_t)i * D_dim;
    const float* vq2_row_i = Vq_2_slice + (int64_t)i * D_dim;
    const float* vr1_row_j = Vr_1_slice + (int64_t)j * D_dim;
    const float* vr2_row_j = Vr_2_slice + (int64_t)j * D_dim;
    const float* vs1_row_k = Vs_1_slice + (int64_t)k * D_dim;
    const float* vs2_row_k = Vs_2_slice + (int64_t)k * D_dim;
    
    // Pre-fetch Aq, Ar, As values for this (i,j,k)
    float aq_ijk = Aq_slice[ijk_idx];
    float ar_ijk = Ar_slice[ijk_idx];
    float as_ijk = As_slice[ijk_idx];


    // --- Term 1.a: grad_Aq from Yq (gather) ---
    if (i < N_dim) {
        const float* grad_out_row_i = grad_output_slice + (int64_t)i * D_dim;
        for (int d = 0; d < D_dim; ++d) {
            grad_aq_val += grad_out_row_i[d] * vr1_row_j[d] * vs1_row_k[d];
        }
    }

    // --- Term 1.b: grad_Aq from Yr' (scatter) ---
    if (j < N_dim) {
        float grad_x_term_j = 0.0f;
        const float* grad_out_row_j = grad_output_slice + (int64_t)j * D_dim;
        for (int d = 0; d < D_dim; ++d) {
            grad_x_term_j += grad_out_row_j[d] * vq2_row_i[d] * vs2_row_k[d];
        }
        grad_aq_val += grad_x_term_j * as_ijk; // Multiply by As[i,j,k]
    }

    // --- Term 1.c: grad_Aq from Ys' (scatter) ---
    if (k < N_dim) {
        float grad_x_term_k = 0.0f;
        const float* grad_out_row_k = grad_output_slice + (int64_t)k * D_dim;
        for (int d = 0; d < D_dim; ++d) {
            grad_x_term_k += grad_out_row_k[d] * vq2_row_i[d] * vr2_row_j[d];
        }
        grad_aq_val += grad_x_term_k * ar_ijk; // Multiply by Ar[i,j,k]
    }

    // --- Term 2.a: grad_Ar from Yr (gather) ---
     if (j < N_dim) {
        const float* grad_out_row_j = grad_output_slice + (int64_t)j * D_dim;
        for (int d = 0; d < D_dim; ++d) {
            grad_ar_val += grad_out_row_j[d] * vq1_row_i[d] * vs1_row_k[d];
        }
    }

    // --- Term 2.b: grad_Ar from Yq' (scatter) ---
     if (i < N_dim) {
        float grad_x_term_i = 0.0f;
        const float* grad_out_row_i = grad_output_slice + (int64_t)i * D_dim;
        for (int d = 0; d < D_dim; ++d) {
            grad_x_term_i += grad_out_row_i[d] * vr2_row_j[d] * vs2_row_k[d];
        }
        grad_ar_val += grad_x_term_i * as_ijk; // Multiply by As[i,j,k]
    }
   
    // --- Term 2.c: grad_Ar from Ys' (scatter) ---
    if (k < N_dim) {
        float grad_x_term_k = 0.0f;
        const float* grad_out_row_k = grad_output_slice + (int64_t)k * D_dim;
         for (int d = 0; d < D_dim; ++d) {
             grad_x_term_k += grad_out_row_k[d] * vq2_row_i[d] * vr2_row_j[d];
         }
        grad_ar_val += grad_x_term_k * aq_ijk; // Multiply by Aq[i,j,k]
    }

    // --- Term 3.a: grad_As from Ys (gather) ---
     if (k < N_dim) {
        const float* grad_out_row_k = grad_output_slice + (int64_t)k * D_dim;
        for (int d = 0; d < D_dim; ++d) {
            grad_as_val += grad_out_row_k[d] * vq1_row_i[d] * vr1_row_j[d];
        }
    }

    // --- Term 3.b: grad_As from Yq' (scatter) ---
     if (i < N_dim) {
        float grad_x_term_i = 0.0f;
        const float* grad_out_row_i = grad_output_slice + (int64_t)i * D_dim;
        for (int d = 0; d < D_dim; ++d) {
            grad_x_term_i += grad_out_row_i[d] * vr2_row_j[d] * vs2_row_k[d];
        }
        grad_as_val += grad_x_term_i * ar_ijk; // Multiply by Ar[i,j,k]
    }
    
    // --- Term 3.c: grad_As from Yr' (scatter) ---
    if (j < N_dim) {
        float grad_x_term_j = 0.0f;
        const float* grad_out_row_j = grad_output_slice + (int64_t)j * D_dim;
        for (int d = 0; d < D_dim; ++d) {
            grad_x_term_j += grad_out_row_j[d] * vq2_row_i[d] * vs2_row_k[d];
        }
        grad_as_val += grad_x_term_j * aq_ijk; // Multiply by Aq[i,j,k]
    }

    // Write results to output tensors
    grad_Aq_slice_out[ijk_idx] = grad_aq_val;
    grad_Ar_slice_out[ijk_idx] = grad_ar_val;
    grad_As_slice_out[ijk_idx] = grad_as_val;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
compute_interim_grads_cuda_wrapper(
    // Inputs (GPU tensors)
    const torch::Tensor& grad_output_slice, // [N, D]
    const torch::Tensor& Vq_1_slice,        // [I, D]
    const torch::Tensor& Vq_2_slice,        // [I, D]
    const torch::Tensor& Vr_1_slice,        // [J, D]
    const torch::Tensor& Vr_2_slice,        // [J, D]
    const torch::Tensor& Vs_1_slice,        // [K, D]
    const torch::Tensor& Vs_2_slice,        // [K, D]
    const torch::Tensor& Aq_slice,          // [I, J, K]
    const torch::Tensor& Ar_slice,          // [I, J, K]
    const torch::Tensor& As_slice           // [I, J, K]
) {
    // Input validation (basic checks)
    TORCH_CHECK(grad_output_slice.is_cuda(), "Input grad_output_slice must be CUDA tensor");
    // ... check other inputs are CUDA ...
    TORCH_CHECK(Aq_slice.dim() == 3, "Aq_slice must be 3D");
    // ... check other dimensions ...

    const int I = Vq_1_slice.size(0);
    const int J = Vr_1_slice.size(0);
    const int K = Vs_1_slice.size(0);
    const int D = Vq_1_slice.size(1);
    const int N = grad_output_slice.size(0);

    // Check dimension consistency
    TORCH_CHECK(Vq_2_slice.sizes() == Vq_1_slice.sizes(), "Vq shapes mismatch");
    TORCH_CHECK(Vr_2_slice.sizes() == Vr_1_slice.sizes(), "Vr shapes mismatch");
    TORCH_CHECK(Vs_2_slice.sizes() == Vs_1_slice.sizes(), "Vs shapes mismatch");
    TORCH_CHECK(Aq_slice.sizes() == torch::IntArrayRef({I, J, K}), "Aq shape mismatch");
    TORCH_CHECK(Ar_slice.sizes() == torch::IntArrayRef({I, J, K}), "Ar shape mismatch");
    TORCH_CHECK(As_slice.sizes() == torch::IntArrayRef({I, J, K}), "As shape mismatch");
    TORCH_CHECK(grad_output_slice.size(1) == D, "grad_output D mismatch");
    // ... add more checks as needed ...

    // Allocate output tensors (temporary grads)
    auto options = Aq_slice.options();
    torch::Tensor grad_Aq_slice_out = torch::zeros({I, J, K}, options);
    torch::Tensor grad_Ar_slice_out = torch::zeros({I, J, K}, options);
    torch::Tensor grad_As_slice_out = torch::zeros({I, J, K}, options);

    // Kernel launch configuration (similar to compute_A_slice_kernel)
    constexpr int BLOCK_DIM_I = 8;
    constexpr int BLOCK_DIM_J = 8;
    constexpr int BLOCK_DIM_K = 8; 
    dim3 blockDim(BLOCK_DIM_I, BLOCK_DIM_J, BLOCK_DIM_K);
    dim3 gridDim(
        (I + BLOCK_DIM_I - 1) / BLOCK_DIM_I,
        (J + BLOCK_DIM_J - 1) / BLOCK_DIM_J,
        (K + BLOCK_DIM_K - 1) / BLOCK_DIM_K
    );

    auto grad_output_cont = grad_output_slice.contiguous();
    auto Vq_1_cont = Vq_1_slice.contiguous();
    auto Vq_2_cont = Vq_2_slice.contiguous();
    auto Vr_1_cont = Vr_1_slice.contiguous();
    auto Vr_2_cont = Vr_2_slice.contiguous();
    auto Vs_1_cont = Vs_1_slice.contiguous();
    auto Vs_2_cont = Vs_2_slice.contiguous();
    auto Aq_cont = Aq_slice.contiguous();
    auto Ar_cont = Ar_slice.contiguous();
    auto As_cont = As_slice.contiguous();

    // Launch Kernel
    compute_interim_grads_kernel<<<gridDim, blockDim>>>(
        grad_output_cont.data_ptr<float>(),
        Vq_1_cont.data_ptr<float>(), Vq_2_cont.data_ptr<float>(),
        Vr_1_cont.data_ptr<float>(), Vr_2_cont.data_ptr<float>(),
        Vs_1_cont.data_ptr<float>(), Vs_2_cont.data_ptr<float>(),
        Aq_cont.data_ptr<float>(), Ar_cont.data_ptr<float>(), As_cont.data_ptr<float>(),
        grad_Aq_slice_out.data_ptr<float>(),
        grad_Ar_slice_out.data_ptr<float>(),
        grad_As_slice_out.data_ptr<float>(),
        I, J, K, D, N
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in compute_interim_grads_cuda_wrapper: %s\n", cudaGetErrorString(err));
    }

    return std::make_tuple(grad_Aq_slice_out, grad_Ar_slice_out, grad_As_slice_out);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
get_interim_grads_cpu(
    const torch::Tensor& grad_output_slice_cpu, // [N, D]
    const torch::Tensor& Q_slice_cpu,           // [I, D]
    const torch::Tensor& R_slice_cpu,           // [J, D]
    const torch::Tensor& S_slice_cpu,           // [K, D]
    const torch::Tensor& Vq_1_slice_cpu,        // [I, D]
    const torch::Tensor& Vq_2_slice_cpu,        // [I, D]
    const torch::Tensor& Vr_1_slice_cpu,        // [J, D]
    const torch::Tensor& Vr_2_slice_cpu,        // [J, D]
    const torch::Tensor& Vs_1_slice_cpu,        // [K, D]
    const torch::Tensor& Vs_2_slice_cpu,        // [K, D]
    const torch::Tensor& Aq_slice_cpu,          // [I, J, K]
    const torch::Tensor& Ar_slice_cpu,          // [I, J, K]
    const torch::Tensor& As_slice_cpu           // [I, J, K]
) {
    const int I = Q_slice_cpu.size(0);
    const int J = R_slice_cpu.size(0);
    const int K = S_slice_cpu.size(0);
    const int D = Q_slice_cpu.size(1);
    const int N = grad_output_slice_cpu.size(0); 
    auto options = Aq_slice_cpu.options(); 

    auto grad_Aq_slice_ref = torch::zeros({I, J, K}, options);
    auto grad_Ar_slice_ref = torch::zeros({I, J, K}, options);
    auto grad_As_slice_ref = torch::zeros({I, J, K}, options);

    auto grad_Aq_acc = grad_Aq_slice_ref.accessor<float, 3>();
    auto grad_Ar_acc = grad_Ar_slice_ref.accessor<float, 3>();
    auto grad_As_acc = grad_As_slice_ref.accessor<float, 3>();
    
    auto grad_output_acc = grad_output_slice_cpu.accessor<float, 2>();
    auto Vq_1_acc = Vq_1_slice_cpu.accessor<float, 2>();
    auto Vq_2_acc = Vq_2_slice_cpu.accessor<float, 2>();
    auto Vr_1_acc = Vr_1_slice_cpu.accessor<float, 2>();
    auto Vr_2_acc = Vr_2_slice_cpu.accessor<float, 2>();
    auto Vs_1_acc = Vs_1_slice_cpu.accessor<float, 2>();
    auto Vs_2_acc = Vs_2_slice_cpu.accessor<float, 2>();
    auto Aq_acc = Aq_slice_cpu.accessor<float, 3>();
    auto Ar_acc = Ar_slice_cpu.accessor<float, 3>();
    auto As_acc = As_slice_cpu.accessor<float, 3>();

    // Replicate Phase 1 logic from compute_grad_A_single here
    // Term 1.a
    if (I <= N) {
        for (int i = 0; i < I; ++i) { for (int j = 0; j < J; ++j) { for (int k = 0; k < K; ++k) { for (int d = 0; d < D; ++d) {
            grad_Aq_acc[i][j][k] += grad_output_acc[i][d] * Vr_1_acc[j][d] * Vs_1_acc[k][d];
        }}}}
    }
    // Term 1.b
    if (J <= N) {
        for (int i = 0; i < I; ++i) { for (int j = 0; j < J; ++j) { for (int k = 0; k < K; ++k) {
            float grad_x_vals = 0.0f;
            for (int d = 0; d < D; ++d) { grad_x_vals += grad_output_acc[j][d] * Vq_2_acc[i][d] * Vs_2_acc[k][d]; }
            grad_Aq_acc[i][j][k] += grad_x_vals * As_acc[i][j][k];
        }}}
    }
    // Term 1.c
    if (K <= N) {
         for (int i = 0; i < I; ++i) { for (int j = 0; j < J; ++j) { for (int k = 0; k < K; ++k) {
             float grad_x_vals = 0.0f;
             for (int d = 0; d < D; ++d) { grad_x_vals += grad_output_acc[k][d] * Vq_2_acc[i][d] * Vr_2_acc[j][d]; }
             grad_Aq_acc[i][j][k] += grad_x_vals * Ar_acc[i][j][k];
         }}}
     }

    // Term 2.a
     if (J <= N) {
         for (int i = 0; i < I; ++i) { for (int j = 0; j < J; ++j) { for (int k = 0; k < K; ++k) { for (int d = 0; d < D; ++d) {
             grad_Ar_acc[i][j][k] += grad_output_acc[j][d] * Vq_1_acc[i][d] * Vs_1_acc[k][d];
         }}}}
     }
    // Term 2.b
     if (I <= N) {
         for (int i = 0; i < I; ++i) { for (int j = 0; j < J; ++j) { for (int k = 0; k < K; ++k) {
             float grad_x_vals = 0.0f;
             for (int d = 0; d < D; ++d) { grad_x_vals += grad_output_acc[i][d] * Vr_2_acc[j][d] * Vs_2_acc[k][d]; }
             grad_Ar_acc[i][j][k] += grad_x_vals * As_acc[i][j][k];
         }}}
     }
    // Term 2.c
     if (K <= N) {
          for (int i = 0; i < I; ++i) { for (int j = 0; j < J; ++j) { for (int k = 0; k < K; ++k) {
              float grad_x_vals = 0.0f;
              for (int d = 0; d < D; ++d) { grad_x_vals += grad_output_acc[k][d] * Vq_2_acc[i][d] * Vr_2_acc[j][d]; }
              grad_Ar_acc[i][j][k] += grad_x_vals * Aq_acc[i][j][k];
          }}}
      }

    // Term 3.a
      if (K <= N) {
          for (int i = 0; i < I; ++i) { for (int j = 0; j < J; ++j) { for (int k = 0; k < K; ++k) { for (int d = 0; d < D; ++d) {
              grad_As_acc[i][j][k] += grad_output_acc[k][d] * Vq_1_acc[i][d] * Vr_1_acc[j][d];
          }}}}
      }
    // Term 3.b
      if (I <= N) {
          for (int i = 0; i < I; ++i) { for (int j = 0; j < J; ++j) { for (int k = 0; k < K; ++k) {
              float grad_x_vals = 0.0f;
              for (int d = 0; d < D; ++d) { grad_x_vals += grad_output_acc[i][d] * Vr_2_acc[j][d] * Vs_2_acc[k][d]; }
              grad_As_acc[i][j][k] += grad_x_vals * Ar_acc[i][j][k];
          }}}
      }
    // Term 3.c
      if (J <= N) {
          for (int i = 0; i < I; ++i) { for (int j = 0; j < J; ++j) { for (int k = 0; k < K; ++k) {
              float grad_x_vals = 0.0f;
              for (int d = 0; d < D; ++d) { grad_x_vals += grad_output_acc[j][d] * Vq_2_acc[i][d] * Vs_2_acc[k][d]; }
              grad_As_acc[i][j][k] += grad_x_vals * Aq_acc[i][j][k];
          }}}
      }

    return std::make_tuple(grad_Aq_slice_ref, grad_Ar_slice_ref, grad_As_slice_ref);
}

//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess) {
//         fprintf(stderr, "CUDA error in apply_softmax_backward_cuda_wrapper: %s\n", cudaGetErrorString(err));
//     }

//     return grad_A_slice_out_gpu;
// }

// --- Optimized Softmax Backward Implementation ---

// Stage 1, Kernel 1: Compute sum_q[i] = sum_{j,k} (grad_Aq[i,j,k] * Aq[i,j,k])
__global__ void compute_softmax_backward_sum_q_kernel(
    const float* __restrict__ grad_Aq_slice_in,
    const float* __restrict__ Aq_slice_in,
    float* __restrict__ sum_q_vec_out, // [I]
    int I_dim, int J_dim, int K_dim)
{
    extern __shared__ float s_reduction_pad[];
    int i_current = blockIdx.x;
    if (i_current >= I_dim) return;

    const float* grad_Aq_plane = grad_Aq_slice_in + (int64_t)i_current * J_dim * K_dim;
    const float* Aq_plane = Aq_slice_in + (int64_t)i_current * J_dim * K_dim;

    int plane_size = J_dim * K_dim;
    int tid_in_block = threadIdx.x;
    int threads_in_block = blockDim.x;

    // --- Pass 1: Compute sum of products for the plane ---
    float thread_sum = 0.0f;
    for (int idx = tid_in_block; idx < plane_size; idx += threads_in_block) {
        thread_sum += grad_Aq_plane[idx] * Aq_plane[idx];
    }
    s_reduction_pad[tid_in_block] = thread_sum;
    __syncthreads();

    // --- Pass 2: Reduce sums in shared memory ---
    for (int offset = threads_in_block / 2; offset > 0; offset >>= 1) {
        if (tid_in_block < offset) {
            s_reduction_pad[tid_in_block] += s_reduction_pad[tid_in_block + offset];
        }
        __syncthreads();
    }

    if (tid_in_block == 0) {
        sum_q_vec_out[i_current] = s_reduction_pad[0];
    }
}

// Stage 1, Kernel 2: Compute sum_r[j] = sum_{i,k} (grad_Ar[i,j,k] * Ar[i,j,k])
__global__ void compute_softmax_backward_sum_r_kernel(
    const float* __restrict__ grad_Ar_slice_in,
    const float* __restrict__ Ar_slice_in,
    float* __restrict__ sum_r_vec_out, // [J]
    int I_dim, int J_dim, int K_dim)
{
    extern __shared__ float s_reduction_pad[];
    int j_current = blockIdx.x;
    if (j_current >= J_dim) return;

    int plane_size = I_dim * K_dim;
    int tid_in_block = threadIdx.x;
    int threads_in_block = blockDim.x;

    float thread_sum = 0.0f;
    for (int idx = tid_in_block; idx < plane_size; idx += threads_in_block) {
        int i = idx / K_dim;
        int k = idx % K_dim;
        int64_t global_idx = (int64_t)i * J_dim * K_dim + (int64_t)j_current * K_dim + k;
        thread_sum += grad_Ar_slice_in[global_idx] * Ar_slice_in[global_idx];
    }
    s_reduction_pad[tid_in_block] = thread_sum;
    __syncthreads();

    for (int offset = threads_in_block / 2; offset > 0; offset >>= 1) {
        if (tid_in_block < offset) {
            s_reduction_pad[tid_in_block] += s_reduction_pad[tid_in_block + offset];
        }
        __syncthreads();
    }

    if (tid_in_block == 0) {
        sum_r_vec_out[j_current] = s_reduction_pad[0];
    }
}

// Stage 1, Kernel 3: Compute sum_s[k] = sum_{i,j} (grad_As[i,j,k] * As[i,j,k])
__global__ void compute_softmax_backward_sum_s_kernel(
    const float* __restrict__ grad_As_slice_in,
    const float* __restrict__ As_slice_in,
    float* __restrict__ sum_s_vec_out, // [K]
    int I_dim, int J_dim, int K_dim)
{
    extern __shared__ float s_reduction_pad[];
    int k_current = blockIdx.x;
    if (k_current >= K_dim) return;

    int plane_size = I_dim * J_dim;
    int tid_in_block = threadIdx.x;
    int threads_in_block = blockDim.x;

    float thread_sum = 0.0f;
    for (int idx = tid_in_block; idx < plane_size; idx += threads_in_block) {
        int i = idx / J_dim;
        int j = idx % J_dim;
        int64_t global_idx = (int64_t)i * J_dim * K_dim + (int64_t)j * K_dim + k_current;
        thread_sum += grad_As_slice_in[global_idx] * As_slice_in[global_idx];
    }
    s_reduction_pad[tid_in_block] = thread_sum;
    __syncthreads();

    for (int offset = threads_in_block / 2; offset > 0; offset >>= 1) {
        if (tid_in_block < offset) {
            s_reduction_pad[tid_in_block] += s_reduction_pad[tid_in_block + offset];
        }
        __syncthreads();
    }

    if (tid_in_block == 0) {
        sum_s_vec_out[k_current] = s_reduction_pad[0];
    }
}


// Stage 2: Final combination kernel
__global__ void apply_softmax_backward_optimized_kernel(
    const float* __restrict__ grad_Aq_slice_in,
    const float* __restrict__ grad_Ar_slice_in,
    const float* __restrict__ grad_As_slice_in,
    const float* __restrict__ Aq_slice_in,
    const float* __restrict__ Ar_slice_in,
    const float* __restrict__ As_slice_in,
    const float* __restrict__ sum_q_vec, // [I]
    const float* __restrict__ sum_r_vec, // [J]
    const float* __restrict__ sum_s_vec, // [K]
    float* __restrict__ grad_A_slice_out, // [I, J, K]
    int I_dim, int J_dim, int K_dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= I_dim || j >= J_dim || k >= K_dim) {
        return;
    }

    int64_t ijk_idx = (int64_t)i * J_dim * K_dim + (int64_t)j * K_dim + k;

    // Each component is calculated without loops using the pre-computed sums
    float grad_A_q_comp = (grad_Aq_slice_in[ijk_idx] - sum_q_vec[i]) * Aq_slice_in[ijk_idx];
    float grad_A_r_comp = (grad_Ar_slice_in[ijk_idx] - sum_r_vec[j]) * Ar_slice_in[ijk_idx];
    float grad_A_s_comp = (grad_As_slice_in[ijk_idx] - sum_s_vec[k]) * As_slice_in[ijk_idx];

    grad_A_slice_out[ijk_idx] = grad_A_q_comp + grad_A_r_comp + grad_A_s_comp;
}


torch::Tensor apply_softmax_backward_cuda_wrapper(
    const torch::Tensor& grad_Aq_slice_gpu,
    const torch::Tensor& grad_Ar_slice_gpu,
    const torch::Tensor& grad_As_slice_gpu,
    const torch::Tensor& Aq_slice_gpu,
    const torch::Tensor& Ar_slice_gpu,
    const torch::Tensor& As_slice_gpu
) {
    TORCH_CHECK(grad_Aq_slice_gpu.is_cuda(), "grad_Aq_slice_gpu must be CUDA");
    TORCH_CHECK(grad_Aq_slice_gpu.dim() == 3 && Aq_slice_gpu.dim() == 3, "Inputs must be 3D");
    
    const int I = grad_Aq_slice_gpu.size(0);
    const int J = grad_Aq_slice_gpu.size(1);
    const int K = grad_Aq_slice_gpu.size(2);
    auto options = grad_Aq_slice_gpu.options();

    // Ensure inputs are contiguous
    auto grad_Aq_cont = grad_Aq_slice_gpu.contiguous();
    auto grad_Ar_cont = grad_Ar_slice_gpu.contiguous();
    auto grad_As_cont = grad_As_slice_gpu.contiguous();
    auto Aq_cont = Aq_slice_gpu.contiguous();
    auto Ar_cont = Ar_slice_gpu.contiguous();
    auto As_cont = As_slice_gpu.contiguous();

    // --- Stage 1: Pre-compute sums ---
    torch::Tensor sum_q_vec = torch::zeros({I}, options);
    torch::Tensor sum_r_vec = torch::zeros({J}, options);
    torch::Tensor sum_s_vec = torch::zeros({K}, options);
    
    int threads_per_block = 256; // Common choice for reduction kernels

    // Launch sum_q kernel
    dim3 gridDim_q(I);
    dim3 blockDim(threads_per_block);
    size_t shmem_size = threads_per_block * sizeof(float);
    compute_softmax_backward_sum_q_kernel<<<gridDim_q, blockDim, shmem_size>>>(
        grad_Aq_cont.data_ptr<float>(), Aq_cont.data_ptr<float>(), sum_q_vec.data_ptr<float>(), I, J, K
    );

    // Launch sum_r kernel
    dim3 gridDim_r(J);
    compute_softmax_backward_sum_r_kernel<<<gridDim_r, blockDim, shmem_size>>>(
        grad_Ar_cont.data_ptr<float>(), Ar_cont.data_ptr<float>(), sum_r_vec.data_ptr<float>(), I, J, K
    );

    // Launch sum_s kernel
    dim3 gridDim_s(K);
    compute_softmax_backward_sum_s_kernel<<<gridDim_s, blockDim, shmem_size>>>(
        grad_As_cont.data_ptr<float>(), As_cont.data_ptr<float>(), sum_s_vec.data_ptr<float>(), I, J, K
    );

    // --- Stage 2: Final Combination ---
    torch::Tensor grad_A_slice_out_gpu = torch::zeros({I, J, K}, options);
    constexpr int BLOCK_DIM_I = 8;
    constexpr int BLOCK_DIM_J = 8;
    constexpr int BLOCK_DIM_K = 8; 
    dim3 blockDim_final(BLOCK_DIM_I, BLOCK_DIM_J, BLOCK_DIM_K);
    dim3 gridDim_final(
        (I + BLOCK_DIM_I - 1) / BLOCK_DIM_I,
        (J + BLOCK_DIM_J - 1) / BLOCK_DIM_J,
        (K + BLOCK_DIM_K - 1) / BLOCK_DIM_K
    );

    apply_softmax_backward_optimized_kernel<<<gridDim_final, blockDim_final>>>(
        grad_Aq_cont.data_ptr<float>(), grad_Ar_cont.data_ptr<float>(), grad_As_cont.data_ptr<float>(),
        Aq_cont.data_ptr<float>(), Ar_cont.data_ptr<float>(), As_cont.data_ptr<float>(),
        sum_q_vec.data_ptr<float>(), sum_r_vec.data_ptr<float>(), sum_s_vec.data_ptr<float>(),
        grad_A_slice_out_gpu.data_ptr<float>(),
        I, J, K
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in apply_softmax_backward_cuda_wrapper (optimized): %s\n", cudaGetErrorString(err));
    }

    return grad_A_slice_out_gpu;
}

// Kernel to compute gradient for Q, assuming grad_A is pre-computed
__global__ void grad_Q_kernel(
    const float* __restrict__ grad_A, // Shape [B, H, I, J, K]
    const float* __restrict__ R,      // Shape [B, H, J, D]
    const float* __restrict__ S,      // Shape [B, H, K, D]
    float*       __restrict__ grad_Q, // Shape [B, H, I, D] - Output
    const int B, const int H, const int I, const int J, const int K, const int D,
    const float scale
) {
    // --- Calculate indices for this thread ---
    // Map threads to output elements (b, h, i, d)
    // Using 1D grid/block for simplicity, can be optimized later
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = (int64_t)B * H * I * D;
    if (idx >= total_elements) {
        return;
    }

    // Decode indices (b, h, i, d) from linear index idx
    const int d = idx % D;
    const int64_t temp_idx_d = idx / D;
    const int i = temp_idx_d % I;
    const int64_t temp_idx_i = temp_idx_d / I;
    const int h = temp_idx_i % H;
    const int b = temp_idx_i / H;


    // --- Calculate strides ---
    const int64_t stride_A_B = (int64_t)H * I * J * K;
    const int64_t stride_A_H = (int64_t)I * J * K;
    const int64_t stride_A_I = (int64_t)J * K;
    const int64_t stride_A_J = (int64_t)K;
    const int64_t stride_A_K = 1;

    const int64_t stride_R_B = (int64_t)H * J * D;
    const int64_t stride_R_H = (int64_t)J * D;
    const int64_t stride_R_J = (int64_t)D;
    const int64_t stride_R_D = 1;

    const int64_t stride_S_B = (int64_t)H * K * D;
    const int64_t stride_S_H = (int64_t)K * D;
    const int64_t stride_S_K = (int64_t)D;
    const int64_t stride_S_D = 1;

    // Output stride calculation not needed as we write to grad_Q[idx]


    // --- Calculate base pointers for this slice (b, h) ---
    // Note: We need the base for the entire tensors here, not just slices,
    // because the loops below access elements across different j and k.
    const float* grad_A_base = grad_A + b * stride_A_B + h * stride_A_H;
    const float* R_base = R + b * stride_R_B + h * stride_R_H;
    const float* S_base = S + b * stride_S_B + h * stride_S_H;


    // --- Compute sum for grad_Q[b, h, i, d] ---
    // grad_Q[i,d] = scale * sum_{j,k} ( grad_A[i,j,k] * R[j,d] * S[k,d] )
    float sum_for_grad_q = 0.0f;
    for (int j_idx = 0; j_idx < J; ++j_idx) {
        for (int k_idx = 0; k_idx < K; ++k_idx) {
            // Calculate linear indices relative to slice base pointers (b,h)
            // grad_A[i, j, k] within the (b,h) slice
            int64_t idx_A = (int64_t)i * stride_A_I + (int64_t)j_idx * stride_A_J + (int64_t)k_idx * stride_A_K;
            // R[j, d] within the (b,h) slice
            int64_t idx_R = (int64_t)j_idx * stride_R_J + (int64_t)d * stride_R_D;
            // S[k, d] within the (b,h) slice
            int64_t idx_S = (int64_t)k_idx * stride_S_K + (int64_t)d * stride_S_D;

            // Accumulate directly as float
            sum_for_grad_q += grad_A_base[idx_A] * R_base[idx_R] * S_base[idx_S];
        }
    }

    grad_Q[idx] = scale * sum_for_grad_q;
}

// Kernel to compute gradient for R, following grad_Q pattern
__global__ void grad_R_kernel(
    const float* __restrict__ grad_A, // Shape [B, H, I, J, K]
    const float* __restrict__ Q,      // Shape [B, H, I, D]
    const float* __restrict__ S,      // Shape [B, H, K, D]
    float*       __restrict__ grad_R, // Shape [B, H, J, D] - Output
    const int B_dim, const int H_dim, const int I_dim, const int J_dim, const int K_dim, const int D_dim,
    const float scale
) {
    // --- Calculate indices for this thread ---
    // Map threads to output elements (b, h, j, d)
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = (int64_t)B_dim * H_dim * J_dim * D_dim; // Output shape B*H*J*D
    if (idx >= total_elements) {
        return;
    }

    // Decode indices (b, h, j, d) from linear index idx
    const int d = idx % D_dim;
    const int64_t temp_idx_d = idx / D_dim;
    const int j = temp_idx_d % J_dim;
    const int64_t temp_idx_j = temp_idx_d / J_dim;
    const int h = temp_idx_j % H_dim;
    const int b = temp_idx_j / H_dim;


    // --- Calculate strides ---
    const int64_t stride_A_B = (int64_t)H_dim * I_dim * J_dim * K_dim;
    const int64_t stride_A_H = (int64_t)I_dim * J_dim * K_dim;
    const int64_t stride_A_I = (int64_t)J_dim * K_dim;
    const int64_t stride_A_J = (int64_t)K_dim;
    const int64_t stride_A_K = 1;

    const int64_t stride_Q_B = (int64_t)H_dim * I_dim * D_dim;
    const int64_t stride_Q_H = (int64_t)I_dim * D_dim;
    const int64_t stride_Q_I = (int64_t)D_dim;
    const int64_t stride_Q_D = 1;

    const int64_t stride_S_B = (int64_t)H_dim * K_dim * D_dim;
    const int64_t stride_S_H = (int64_t)K_dim * D_dim;
    const int64_t stride_S_K = (int64_t)D_dim;
    const int64_t stride_S_D = 1;

    // Output stride calculation not needed as we write to grad_R[idx]


    // --- Calculate base pointers for this slice (b, h) ---
    // Note: We need the base for the entire tensors here, not just slices,
    // because the loops below access elements across different i and k.
    const float* grad_A_base = grad_A + b * stride_A_B + h * stride_A_H;
    const float* Q_base = Q + b * stride_Q_B + h * stride_Q_H;
    const float* S_base = S + b * stride_S_B + h * stride_S_H;


    // --- Compute sum for grad_R[b, h, j, d] ---
    // grad_R[b, h, j, d] = scale * sum_{i,k} ( grad_A[i,j,k] * Q[i,d] * S[k,d] )
    float sum_for_grad_r = 0.0f;
    for (int i_idx = 0; i_idx < I_dim; ++i_idx) {
        for (int k_idx = 0; k_idx < K_dim; ++k_idx) {
            // Calculate linear indices relative to slice base pointers (b,h)
            // grad_A[i, j, k] within the (b,h) slice
            int64_t idx_A = (int64_t)i_idx * stride_A_I + (int64_t)j * stride_A_J + (int64_t)k_idx * stride_A_K;
            // Q[i, d] within the (b,h) slice
            int64_t idx_Q = (int64_t)i_idx * stride_Q_I + (int64_t)d * stride_Q_D;
            // S[k, d] within the (b,h) slice
            int64_t idx_S = (int64_t)k_idx * stride_S_K + (int64_t)d * stride_S_D;

            // Accumulate directly as float
            sum_for_grad_r += grad_A_base[idx_A] * Q_base[idx_Q] * S_base[idx_S];
        }
    }

    grad_R[idx] = scale * sum_for_grad_r;
}

// Kernel to compute gradient for S, following grad_Q/R pattern
__global__ void grad_S_kernel(
    const float* __restrict__ grad_A, // Shape [B, H, I, J, K]
    const float* __restrict__ Q,      // Shape [B, H, I, D]
    const float* __restrict__ R,      // Shape [B, H, J, D]
    float*       __restrict__ grad_S, // Shape [B, H, K, D] - Output
    const int B_dim, const int H_dim, const int I_dim, const int J_dim, const int K_dim, const int D_dim,
    const float scale
) {
    // Map threads to output elements (b, h, k, d)
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = (int64_t)B_dim * H_dim * K_dim * D_dim; // Output shape B*H*K*D
    if (idx >= total_elements) {
        return;
    }

    // Decode indices (b, h, k, d) from linear index idx
    const int d = idx % D_dim;
    const int64_t temp_idx_d = idx / D_dim;
    const int k = temp_idx_d % K_dim; // This thread calculates for grad_S[k,d]
    const int64_t temp_idx_k = temp_idx_d / K_dim;
    const int h = temp_idx_k % H_dim;
    const int b = temp_idx_k / H_dim;

    // Strides for grad_A[B, H, I, J, K]
    const int64_t stride_A_B = (int64_t)H_dim * I_dim * J_dim * K_dim;
    const int64_t stride_A_H = (int64_t)I_dim * J_dim * K_dim;
    const int64_t stride_A_I = (int64_t)J_dim * K_dim;
    const int64_t stride_A_J = (int64_t)K_dim;
    const int64_t stride_A_K = 1;

    // Strides for Q[B, H, I, D]
    const int64_t stride_Q_B = (int64_t)H_dim * I_dim * D_dim;
    const int64_t stride_Q_H = (int64_t)I_dim * D_dim;
    const int64_t stride_Q_I = (int64_t)D_dim;
    const int64_t stride_Q_D = 1;

    // Strides for R[B, H, J, D]
    const int64_t stride_R_B = (int64_t)H_dim * J_dim * D_dim;
    const int64_t stride_R_H = (int64_t)J_dim * D_dim;
    const int64_t stride_R_J = (int64_t)D_dim;
    const int64_t stride_R_D = 1;

    // Calculate base pointers for this batch/head (b, h)
    const float* grad_A_base = grad_A + b * stride_A_B + h * stride_A_H;
    const float* Q_base = Q + b * stride_Q_B + h * stride_Q_H;
    const float* R_base = R + b * stride_R_B + h * stride_R_H;

    // Compute sum for grad_S[b, h, k, d]
    // grad_S[k,d] = scale * sum_{i,j} ( grad_A[i,j,k] * Q[i,d] * R[j,d] )
    float sum_for_grad_s = 0.0f;
    for (int i_loop = 0; i_loop < I_dim; ++i_loop) { // Sum over i
        for (int j_loop = 0; j_loop < J_dim; ++j_loop) { // Sum over j
            // Calculate linear indices relative to slice base pointers (b,h)
            int64_t idx_A = (int64_t)i_loop * stride_A_I + (int64_t)j_loop * stride_A_J + (int64_t)k * stride_A_K;
            int64_t idx_Q = (int64_t)i_loop * stride_Q_I + (int64_t)d * stride_Q_D;
            int64_t idx_R = (int64_t)j_loop * stride_R_J + (int64_t)d * stride_R_D;

            sum_for_grad_s += grad_A_base[idx_A] * Q_base[idx_Q] * R_base[idx_R];
        }
    }
    grad_S[idx] = scale * sum_for_grad_s;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
          torch::Tensor, torch::Tensor,
          torch::Tensor, torch::Tensor,
          torch::Tensor, torch::Tensor>
backward_cuda(
    torch::Tensor grad_output, 
    torch::Tensor Q,           
    torch::Tensor R,           
    torch::Tensor S,           
    torch::Tensor Vq_1,        
    torch::Tensor Vq_2,        
    torch::Tensor Vr_1,        
    torch::Tensor Vr_2,        
    torch::Tensor Vs_1,        
    torch::Tensor Vs_2,        
    double dropout_rate)
{
  grad_output = grad_output.contiguous();
  Q = Q.contiguous();  
  R = R.contiguous();  
  S = S.contiguous();
  Vq_1 = Vq_1.contiguous();
  Vq_2 = Vq_2.contiguous(); 
  Vr_1 = Vr_1.contiguous();
  Vr_2 = Vr_2.contiguous(); 
  Vs_1 = Vs_1.contiguous();
  Vs_2 = Vs_2.contiguous(); 

  // 1) allocate final output gradient tensors
  auto grad_Q   = torch::zeros_like(Q);
  auto grad_R   = torch::zeros_like(R); 
  auto grad_S   = torch::zeros_like(S); 
  auto grad_Vq_1 = torch::zeros_like(Vq_1);
  auto grad_Vq_2 = torch::zeros_like(Vq_2); 
  auto grad_Vr_1 = torch::zeros_like(Vr_1);
  auto grad_Vr_2 = torch::zeros_like(Vr_2);
  auto grad_Vs_1 = torch::zeros_like(Vs_1);
  auto grad_Vs_2 = torch::zeros_like(Vs_2);

  // 2) extract dims + scale
  const int B = Q.size(0);
  const int H = Q.size(1);
  const int I = Q.size(2); 
  const int J = R.size(2); 
  const int K = S.size(2); 
  const int D = Q.size(3);
  const int N_grad = grad_output.size(2); 

  const float scale = 1.0f / sqrtf((float)D);
  const int threads = 256; 

  // 3) Compute grad_A tensor on GPU by processing slice by slice - materializes on GPU 
  auto grad_A_batched_gpu = torch::zeros({B, H, I, J, K}, Q.options()); 

  for (int b = 0; b < B; ++b) {
      for (int h = 0; h < H; ++h) {
          // Get GPU slices for current (b,h)
          auto Q_slice_gpu = Q.select(0, b).select(0, h);
          auto R_slice_gpu = R.select(0, b).select(0, h);
          auto S_slice_gpu = S.select(0, b).select(0, h);
          auto grad_output_slice_gpu = grad_output.select(0, b).select(0, h);
          auto Vq_1_slice_gpu = Vq_1.select(0, b).select(0, h);
          auto Vq_2_slice_gpu = Vq_2.select(0, b).select(0, h);
          auto Vr_1_slice_gpu = Vr_1.select(0, b).select(0, h);
          auto Vr_2_slice_gpu = Vr_2.select(0, b).select(0, h);
          auto Vs_1_slice_gpu = Vs_1.select(0, b).select(0, h);
          auto Vs_2_slice_gpu = Vs_2.select(0, b).select(0, h);
        
          torch::Tensor A_slice_gpu = compute_A_slice_cuda_wrapper(
              Q_slice_gpu, R_slice_gpu, S_slice_gpu, scale
          );
          
          torch::Tensor Aq_slice_gpu = compute_Aq_slice_cuda_wrapper(A_slice_gpu);
          torch::Tensor Ar_slice_gpu = compute_Ar_slice_cuda_wrapper(A_slice_gpu);
          torch::Tensor As_slice_gpu = compute_As_slice_cuda_wrapper(A_slice_gpu);

                    // --- Launch OPTIMIZED gather kernels for V1 grads ---
          {
              auto gradVq1_slice = grad_Vq_1.select(0, b).select(0, h);
              const int64_t N_kernel = (int64_t)I * D;
              const dim3 blocks((N_kernel + threads - 1) / threads);
              gather_grad_Vq1_kernel_optimized<<<blocks, threads>>>(
                  grad_output_slice_gpu.data_ptr<float>(),
                  Vr_1_slice_gpu.data_ptr<float>(),
                  Vs_1_slice_gpu.data_ptr<float>(),
                  Ar_slice_gpu.data_ptr<float>(),
                  As_slice_gpu.data_ptr<float>(),
                  gradVq1_slice.data_ptr<float>(),
                  I, J, K, D, N_grad
              );
          }
          {
              auto gradVr1_slice = grad_Vr_1.select(0, b).select(0, h);
              const int64_t N_kernel = (int64_t)J * D;
              const dim3 blocks((N_kernel + threads - 1) / threads);
              gather_grad_Vr1_kernel_optimized<<<blocks, threads>>>(
                  grad_output_slice_gpu.data_ptr<float>(),
                  Vq_1_slice_gpu.data_ptr<float>(),
                  Vs_1_slice_gpu.data_ptr<float>(),
                  Aq_slice_gpu.data_ptr<float>(),
                  As_slice_gpu.data_ptr<float>(),
                  gradVr1_slice.data_ptr<float>(),
                  I, J, K, D, N_grad
              );
          }
          {
              auto gradVs1_slice = grad_Vs_1.select(0, b).select(0, h);
              const int64_t N_kernel = (int64_t)K * D;
              const dim3 blocks((N_kernel + threads - 1) / threads);
              gather_grad_Vs1_kernel_optimized<<<blocks, threads>>>(
                  grad_output_slice_gpu.data_ptr<float>(),
                  Vq_1_slice_gpu.data_ptr<float>(),
                  Vr_1_slice_gpu.data_ptr<float>(),
                  Aq_slice_gpu.data_ptr<float>(),
                  Ar_slice_gpu.data_ptr<float>(),
                  gradVs1_slice.data_ptr<float>(),
                  I, J, K, D, N_grad
              );
          }
          
          {
              auto gradVq2_slice = grad_Vq_2.select(0, b).select(0, h);
              const int64_t N_kernel = (int64_t)I * D;
              const dim3 blocks((N_kernel + threads - 1) / threads);
              scatter_grad_Vq2_kernel_optimized<<<blocks, threads>>>(
                  grad_output_slice_gpu.data_ptr<float>(),
                  Aq_slice_gpu.data_ptr<float>(), Ar_slice_gpu.data_ptr<float>(), As_slice_gpu.data_ptr<float>(),
                  Vr_2_slice_gpu.data_ptr<float>(), Vs_2_slice_gpu.data_ptr<float>(),
                  gradVq2_slice.data_ptr<float>(),
                  I, J, K, D, N_grad
              );
          }
          {
              auto gradVr2_slice = grad_Vr_2.select(0, b).select(0, h);
              const int64_t N_kernel = (int64_t)J * D;
              const dim3 blocks((N_kernel + threads - 1) / threads);
              scatter_grad_Vr2_kernel_optimized<<<blocks, threads>>>(
                  grad_output_slice_gpu.data_ptr<float>(),
                  Aq_slice_gpu.data_ptr<float>(), Ar_slice_gpu.data_ptr<float>(), As_slice_gpu.data_ptr<float>(),
                  Vq_2_slice_gpu.data_ptr<float>(), Vs_2_slice_gpu.data_ptr<float>(),
                  gradVr2_slice.data_ptr<float>(),
                  I, J, K, D, N_grad
              );
          }
          {
              auto gradVs2_slice = grad_Vs_2.select(0, b).select(0, h);
              const int64_t N_kernel = (int64_t)K * D;
              const dim3 blocks((N_kernel + threads - 1) / threads);
              scatter_grad_Vs2_kernel_optimized<<<blocks, threads>>>(
                  grad_output_slice_gpu.data_ptr<float>(),
                  Aq_slice_gpu.data_ptr<float>(), Ar_slice_gpu.data_ptr<float>(), As_slice_gpu.data_ptr<float>(),
                  Vq_2_slice_gpu.data_ptr<float>(), Vr_2_slice_gpu.data_ptr<float>(),
                  gradVs2_slice.data_ptr<float>(),
                  I, J, K, D, N_grad
              );
          }
          torch::Tensor grad_Aq_slice_gpu_tmp, grad_Ar_slice_gpu_tmp, grad_As_slice_gpu_tmp;
          std::tie(grad_Aq_slice_gpu_tmp, grad_Ar_slice_gpu_tmp, grad_As_slice_gpu_tmp) = 
              compute_interim_grads_cuda_wrapper(
                  grad_output_slice_gpu, 
                  Vq_1_slice_gpu, Vq_2_slice_gpu, 
                  Vr_1_slice_gpu, Vr_2_slice_gpu, 
                  Vs_1_slice_gpu, Vs_2_slice_gpu, 
                  Aq_slice_gpu, Ar_slice_gpu, As_slice_gpu);
          
          torch::Tensor final_grad_A_slice_gpu = apply_softmax_backward_cuda_wrapper(
              grad_Aq_slice_gpu_tmp, grad_Ar_slice_gpu_tmp, grad_As_slice_gpu_tmp,
              Aq_slice_gpu, Ar_slice_gpu, As_slice_gpu
          );
          // ----------------------------------------------------------------

          // Copy the computed slice into the full batched grad_A tensor on GPU
          grad_A_batched_gpu.select(0, b).select(0, h).copy_(final_grad_A_slice_gpu);
      }
  }
  // Error check after grad_A computation loop
  cudaError_t ga_err = cudaGetLastError();
  if (ga_err != cudaSuccess) {
    fprintf(stderr, "CUDA error after grad_A loop in backward_cuda: %s\n", cudaGetErrorString(ga_err));
  }

  // --- 5. Launch kernelS for full grad_Q/R/S using the full grad_A_batched_gpu ---
  {
      const int64_t N_kernel_Q = (int64_t)B * H * I * D; //output size for grad_Q/R/S
      const dim3 blocks_Q((N_kernel_Q + threads - 1) / threads);
      grad_Q_kernel<<<blocks_Q, threads>>>( 
          grad_A_batched_gpu.data_ptr<float>(), 
          R.data_ptr<float>(),                  
          S.data_ptr<float>(),                  
          grad_Q.data_ptr<float>(),             
          B, H, I, J, K, D, scale);      
  }  
  {
      const int64_t N_kernel_R = (int64_t)B * H * J * D; 
      const dim3 blocks_R((N_kernel_R + threads - 1) / threads);
      grad_R_kernel<<<blocks_R, threads>>>( 
          grad_A_batched_gpu.data_ptr<float>(), 
          Q.data_ptr<float>(),                 
          S.data_ptr<float>(),                 
          grad_R.data_ptr<float>(),            
          B, H, I, J, K, D, scale);
   }

  {
      const int64_t N_kernel_S = (int64_t)B * H * K * D; 
      const dim3 blocks_S((N_kernel_S + threads - 1) / threads);
      grad_S_kernel<<<blocks_S, threads>>>( 
          grad_A_batched_gpu.data_ptr<float>(), 
          Q.data_ptr<float>(),                 
          R.data_ptr<float>(),                 
          grad_S.data_ptr<float>(),            
          B, H, I, J, K, D, scale);
  }

  cudaDeviceSynchronize(); 

  return std::make_tuple(
      grad_Q, grad_R, grad_S,
      grad_Vq_1, grad_Vq_2, 
      grad_Vr_1, grad_Vr_2,
      grad_Vs_1, grad_Vs_2
  );
}



