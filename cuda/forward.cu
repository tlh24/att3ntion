#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>      
#include <cuda.h>
#include <cuda_runtime.h>
#include "../cpp/manual_att3ntion.h"

// -- Forward Pass --
#define TILE_J 16
#define TILE_K 16
#define TILE_I 16
#define TILE 16

#define TILE_I_SPECIAL 4
#define TILE_J_SPECIAL 4
#define TILE_K_SPECIAL 4

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

__global__ void
__launch_bounds__(256, 2) // make sure we can get 2 thread blocks per SM
Yq_scatter_smash(
	const float* __restrict__ Q,
	const float* __restrict__ R,
	const float* __restrict__ S,
	const float* __restrict__ Vr_2,
	const float* __restrict__ Vs_2,
	const float* __restrict__ m_j_in, const float* __restrict__ l_j_in,
	const float* __restrict__ m_k_in, const float* __restrict__ l_k_in,
	float* __restrict__ Y_q_,
	int B, int H, int I, int J, int K, int D, float scale
){
	// Q is [B, H, I, D] R is [B, H, J, D] S is [B, H, K, D]
	// launcher must assert i == j == k,
	// D % 32 == 0,
	// blockDim.x >= TILE,
	// blockdim.x <= TILE^3
	// TILE^3 % blockDim.x == 0,
	//			(for iter bounds checking for attn accum)
	// I,J,K >= 16
	//			(again for attn)
	// I,J,K % TILE == 0
	//
	const int b = blockIdx.z;
	const int h = blockIdx.y;
	const int i_start = blockIdx.x * TILE;
	// computation has no dependencies across the i index
	// but we need to parallelize it to reduce the number
	// of redundant dram R, S, Vr_2, Vs_2 loads
	// (which don't depend on i)
	// otherwise we'd reload them for each i!

	// thread indices change based on what step we're doing.
	// 1-D thread block:
	const int tid = threadIdx.x;
	const int tpb = blockDim.x; // threads per block

	// --- Pointers to Global Memory ---
	int64_t bh_offset = (int64_t)((b*H + h) * I * D);
	const int64_t mj_bh_offset = (int64_t)((b*H + h) * I);

	// shared memory
	extern __shared__ float smem[];
	float* q_tile = (float*)smem;
	float* r_tile = q_tile + TILE * D;
	float* s_tile = r_tile + TILE * D;
	float* vr_tile = s_tile + TILE * D;
	float* vs_tile = vr_tile + TILE * D;
	float* attn_tile = vs_tile + TILE * D;

	// m = max of row, l = 1 / sum of exps
	float* mj_tile = attn_tile + TILE * TILE * TILE;
	float* lj_tile = mj_tile + TILE;
	float* mk_tile = lj_tile + TILE;
	float* lk_tile = mk_tile + TILE;
	// each thread writes load_iters entries in yq.
	float yq_acc0, yq_acc1, yq_acc2, yq_acc3 ; // local, not shared!
	float yq_acc4, yq_acc5, yq_acc6, yq_acc7 ;
	yq_acc0 = 0.f; // can't nvcc figure this out??
	yq_acc1 = 0.f;
	yq_acc2 = 0.f;
	yq_acc3 = 0.f;
	yq_acc4 = 0.f;
	yq_acc5 = 0.f;
	yq_acc6 = 0.f;
	yq_acc7 = 0.f;

	// cooperative load Q, size [TILE, D]
	unsigned short i_load = tid / D; // i, j, or k: tpb=512, 0..15; tpb=256 0..7
	unsigned short d_load = tid % D; // if tpb = 256, '', load_iters = 2
	unsigned short load_iters = max(1, (TILE * D) / tpb);
	unsigned short load_step = tpb / D; // if tpb=256, load_step=8

	// Q is fixed for the life of the thread.
	// note these loads are cooperative & contiguous across the block.
	// if tpb > D*TILE, many warps will be noop here.
	for( unsigned short n = 0; n < TILE; n += load_step ){
		if( i_start + n + i_load < I ){
			q_tile[(n + i_load)*D + d_load] =
				Q[bh_offset + (i_start + n + i_load)*D + d_load];
		}
	}

	// iterate over j tiles.
	for( unsigned short jt = 0; jt < J; jt += TILE){
		// load r_tile
		for( unsigned short n = 0; n < TILE; n += load_step ){
			if( jt + n + i_load < J ){
				r_tile[(n + i_load)*D + d_load] =
					R[bh_offset + (jt + n + i_load)*D + d_load];
			}
		}
		// load vr_tile
		for( unsigned short n = 0; n < TILE; n += load_step ){
			if( jt + n + i_load < J ){
				vr_tile[(n + i_load)*D + d_load] =
					Vr_2[bh_offset + (jt + n + i_load)*D + d_load];
			}
		}
		// load mj_tile, lj_tile
		if( tid < TILE && jt + tid < J){
			// block/warp divergence but ok
			mj_tile[tid] = m_j_in[mj_bh_offset + jt + tid];
			lj_tile[tid] = 1.f / l_j_in[mj_bh_offset + jt + tid];
		}
		// iterate over the k tiles
		for( unsigned short kt = 0; kt < K; kt += TILE){
			// load s_tile
			for( unsigned short n = 0; n < TILE; n += load_step ){
				if( kt + n + i_load < K ){
					s_tile[(n + i_load)*D + d_load] =
						S[bh_offset + (kt + n + i_load)*D + d_load];
				}
			}
			// load mk_tile, lk_tile
			if( tid < TILE && kt + tid < K){
				mk_tile[tid] = m_k_in[mj_bh_offset + kt + tid];
				lk_tile[tid] = 1.f / l_k_in[mj_bh_offset + kt + tid];
			}
			__syncthreads();

			// == new attn block ==
			float acc[4][4][4]; // must be in registers!!
			#pragma unroll
			for(int i0 = 0; i0 < 4; i0++){
				#pragma unroll
				for(int i1 = 0; i1 < 4; i1++){
				#pragma unroll
					for(int i2 = 0; i2 < 4; i2++){
						acc[i0][i1][i2] = 0.f;
					}
				}
			}
			unsigned short da = tid / (TILE*TILE*TILE/64); // 0 .. 3
			// TODO iterate here (if needed)
			unsigned short ia = (tid / (TILE*TILE/16)) % (TILE/4);
			unsigned short ja = (tid / (TILE/4)) % (TILE/4);
			unsigned short ka = tid % (TILE/4);
			float qa[4], ra[4], sa[4];
			for(unsigned short db = 0; db < 8; db++){ // TODO calc n_iters
				#pragma unroll
				for(int u = 0; u < 4; u++){
					qa[u] = q_tile[(ia*4+u)*D + da*8 + db];
					ra[u] = r_tile[(ja*4+u)*D + da*8 + db];
					sa[u] = s_tile[(ka*4+u)*D + da*8 + db];
				}
				#pragma unroll
				for(int i0 = 0; i0 < 4; i0++){
					#pragma unroll
					for(int i1 = 0; i1 < 4; i1++){
						#pragma unroll
						for(int i2 = 0; i2 < 4; i2++){
							acc[i0][i1][i2] += qa[i0] * ra[i1] * sa[i2];
						}
					}
				}
			}
			// reduction time!
			// simple linear, not log - don't have enough smem
			for(unsigned char u = 1; u < 4; u++){
				if(da == u){
					#pragma unroll
					for(int i0 = 0; i0 < 4; i0++){
						#pragma unroll
						for(int i1 = 0; i1 < 4; i1++){
							#pragma unroll
							for(int i2 = 0; i2 < 4; i2++){
								attn_tile[(ia*4+i0)*TILE*TILE + (ja*4+i1)*TILE + (ka*4+i2)] = acc[i0][i1][i2];
							}
						}
					}
				}
				__syncthreads();
				if(da == 0){
					#pragma unroll
					for(int i0=0; i0 < 4; i0++){
						#pragma unroll
						for(int i1 = 0; i1 < 4; i1++){
							#pragma unroll
							for(int i2 = 0; i2 < 4; i2++){
								acc[i0][i1][i2] += attn_tile[(ia*4+i0)*TILE*TILE + (ja*4+i1)*TILE + (ka*4+i2)];
							}
						}
					}
				}
				__syncthreads();
			}
			if(da == 0){
				#pragma unroll
				for(int i0 = 0; i0 < 4; i0++){
					#pragma unroll
					for(int i1 = 0; i1 < 4; i1++){
						float mjt = mj_tile[ja*4+i1]; //constant over innerloop
						float ljt = lj_tile[ja*4+i1];
						#pragma unroll
						for(int i2 = 0; i2 < 4; i2++){
							float logit = acc[i0][i1][i2] * scale;
							float ar = expf(logit - mjt) * ljt;
							float as = expf(logit - mk_tile[ka*4+i2]) * lk_tile[ka*4+i2];
							attn_tile[(ia*4+i0)*TILE*TILE + (ja*4+i1)*TILE + (ka*4+i2)] = ar * as;
						}
					}
				}
			}
			__syncthreads();
			// // original attn calc
			// int attn_iters = (TILE*TILE*TILE) / tpb; // tpb = threads per block
			// for( int n = 0; n < attn_iters; n++ ){
			// 	int tid_n = tid + n*tpb;
			// 	int ia = tid_n / (TILE*TILE);
			// 	int ja = (tid_n / TILE) % TILE;
			// 	int ka = tid_n % TILE;
			// 	float f = 0.f;
			// 	for(int da = 0; da < D; da++){
			// 		// really should move q and r to registers
			// 		// this would require n being the fastest index.
			// 		// (each thread still computes attn_iters scores)
			// 		f += q_tile[ia*D + da] * r_tile[ja*D + da] * s_tile[ka*D + da];
			// 	}
			// 	float logit = f * scale;
			// 	float ar = expf(logit - mj_tile[ja]) * lj_tile[ja];
			// 	float as = expf(logit - mk_tile[ka]) * lk_tile[ka];
			// 	attn_tile[tid_n] = ar * as;
			// }
			// __syncthreads();
			// load vs_tile
			for( int n = 0; n < TILE; n += load_step ){
				if( kt + n + i_load < K ){
					vs_tile[(n + i_load)*D + d_load] =
					Vs_2[bh_offset + (kt + n + i_load)*D + d_load];
				}
			}
			__syncthreads();

			// iterate over yq
			for( unsigned short n=0; n < load_iters; n++){
				unsigned short tid_n = tid + n*tpb;
				float f = 0.f;
				if( tid_n < TILE*D ){
					unsigned short dy = tid_n % D; // = d_load
					unsigned short iy = tid_n / D;
					for( unsigned short jy = 0; jy < TILE; jy++){
						float vrt = vr_tile[jy*D + dy];
						for( unsigned short ky = 0; ky < TILE; ky++){
							// this too can be pushed to registers.
							f += attn_tile[iy*TILE*TILE + jy*TILE + ky]
									* vrt
									* vs_tile[ky*D + dy];
						}
					}
				}
				switch(n){
					case 0:
						yq_acc0 += f;
						break;
					case 1:
						yq_acc1 += f;
						break;
					case 2:
						yq_acc2 += f;
						break;
					case 3:
						yq_acc3 += f;
						break;
					case 4:
						yq_acc4 += f;
						break;
					case 5:
						yq_acc5 += f;
						break;
					case 6:
						yq_acc6 += f;
						break;
					case 7:
						yq_acc7 += f;
						break;
				}
			} // end yq_iters
		} // end k tiles
	} // end j tiles
	for( unsigned short n=0; n < TILE; n += load_step){
		if( i_start + n + i_load < I){
			float f = 0.f;
			switch(n/load_step){
				case 0:
					f = yq_acc0;
					break;
				case 1:
					f = yq_acc1;
					break;
				case 2:
					f = yq_acc2;
					break;
				case 3:
					f = yq_acc3;
					break;
				case 4:
					f = yq_acc4;
					break;
				case 5:
					f = yq_acc5;
					break;
				case 6:
					f = yq_acc6;
					break;
				case 7:
					f = yq_acc7;
					break;
			}
			Y_q_[bh_offset + (i_start + n + i_load)*D + d_load] = f;
		}
	}
}
void Yq_scatter_smash_launcher(
	const at::Tensor& Q, const at::Tensor& R, const at::Tensor& S,
	const at::Tensor& Vr_2, const at::Tensor& Vs_2,
	at::Tensor& Y_q_, float scale
){
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
	// TODO: lots of flops,
	// these should be computed *once* in the gather kernel.
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
		const int TPB = 256; // threads per block. 256, 512 or 1024

		TORCH_CHECK(D % 32 == 0, "D must be a multiple of 32");
		TORCH_CHECK(TPB % (TILE*TILE) == 0, "Threads per block must be a multiple of Tile^2");
		TORCH_CHECK(I == J, "Kernel only implemented for Q, R, S same size");
		TORCH_CHECK(K == J, "Kernel only implemented for Q, R, S same size");
		TORCH_CHECK((TILE*D) / TPB <= 8, "Maximum 8 loads per thread - D is too large");
		TORCH_CHECK(I >= TILE, "Need at least one tile along I,J,K");
		TORCH_CHECK(I % TILE == 0, "I,J,K need to be a multiple of TILE");


		dim3 grid((I + TILE - 1) / TILE, H, B); // x, y, z

		size_t smem_size = sizeof(float) * (
			TILE * D * 5 + // Q, R, S, Vr, Vs
			TILE*TILE*TILE + // attn
			TILE * 4 ); // mj, lj, mk, lk

		Yq_scatter_smash<<<grid, TPB, smem_size>>>(
			Q.data_ptr<float>(),
			R.data_ptr<float>(),
			S.data_ptr<float>(),
			Vr_2.data_ptr<float>(),
			Vs_2.data_ptr<float>(),
			m_j.data_ptr<float>(), l_j.data_ptr<float>(),
			m_k.data_ptr<float>(), l_k.data_ptr<float>(),
			Y_q_.data_ptr<float>(),
			B, H, I, J, K, D, scale );
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
	// Q is [B, H, I, D] R is [B, H, J, D] S is [b, h, k, d]
	// usually i == j == k
    // --- Grid Mapping---
    const int i_tile_idx_grid = blockIdx.x; // I split into tiles of TILE_I
    const int j_tile_idx_grid = blockIdx.y; // J split into tiles of TILE_J
    const int bh_idx = blockIdx.z;

    // --- Thread Mapping ---
    const int thread_d_idx = threadIdx.x; // curr thread's d_idx in (D, TILE_I)
    const int thread_i_idx = threadIdx.y; // curr thread's i_idx in (TILE_I)

    const int i_start = i_tile_idx_grid * TILE_I_SPECIAL; // start index of current I tile
    const int j_start = j_tile_idx_grid * TILE_J; // start index of current J tile

    const int i_idx = i_start + thread_i_idx; // global index of curr thread's I

    if (i_idx >= I || j_start >= J) return; // check bounds

    // --- Pointers to Global Memory ---
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
    float* r_tile = q_tile + TILE_I_SPECIAL * D;
    float* s_tile = r_tile + TILE_J * D;
    float* vr_tile = s_tile + TILE_K * D;
    float* vs_tile = vr_tile + TILE_J * D;

    // m = max of row, l = sum of exps
    float* mj_tile = vs_tile + TILE_K * D;
    float* lj_tile = mj_tile + TILE_J;
    float* mk_tile = lj_tile + TILE_J;
    float* lk_tile = mk_tile + TILE_K;

    float* o_tile = lk_tile + TILE_K;

    float* reduction_smem = o_tile + TILE_I_SPECIAL * D; 
    
    int flat_thread_id_2d = threadIdx.y * blockDim.x + threadIdx.x; 
    int threads_per_block = blockDim.x * blockDim.y; // tile I * D
    
    o_tile[thread_i_idx * D + thread_d_idx] = 0.0f;

    // --- Load Q tile to smem (strided/coalesced layout)---
    for (int load_idx = flat_thread_id_2d; load_idx < TILE_I_SPECIAL * D; load_idx += threads_per_block) {
        int row_in_tile = load_idx / D;
        int col_in_tile = load_idx % D;
        int i_global = i_start + row_in_tile;
        if (i_global < I) { // check bounds
            q_tile[row_in_tile * D + col_in_tile] = Q[q_bh_offset + i_global * D + col_in_tile];
        }
    }
    __syncthreads();

    // --- Load R and Vr2 tiles to smem (strided/coalesced layout) ---
    for (int load_idx = flat_thread_id_2d; load_idx < TILE_J * D; load_idx += threads_per_block) {
        int row_in_tile = load_idx / D;
        int col_in_tile = load_idx % D;
        int j_global = j_start + row_in_tile;
        if (j_global < J) {
            r_tile[row_in_tile * D + col_in_tile] = R[r_bh_offset + j_global * D + col_in_tile];
            vr_tile[row_in_tile * D + col_in_tile] = Vr_2[vr_bh_offset + j_global * D + col_in_tile];
        }
    }

    // load mj and lj tiles to smem (TILE_J)
    for (int j_load = flat_thread_id_2d; j_load < TILE_J; j_load += threads_per_block) {
        if (j_start + j_load < J) {
             mj_tile[j_load] = m_j_in[mj_bh_offset + j_start + j_load];
             lj_tile[j_load] = l_j_in[mj_bh_offset + j_start + j_load];
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
        for (int j_tile_idx = 0; j_tile_idx < TILE_J; ++j_tile_idx) { //j=0,1,2,...,TILE_J-1
            if (j_start + j_tile_idx >= J) continue;

            for (int k_tile_idx = 0; k_tile_idx < TILE_K; ++k_tile_idx) { //k=0,1,2,...,TILE_K-1
                if (k0 + k_tile_idx >= K) continue;
                
                // --- Step 2.1: Parallel Reduction for Dot Product ---
                const float* q_vec = q_tile + thread_i_idx * D; // 0 to start
                const float* r_vec = r_tile + j_tile_idx * D;
                const float* s_vec = s_tile + k_tile_idx * D;

                // Each thread computes its product and stores it in smem
                reduction_smem[thread_i_idx * D + thread_d_idx] = q_vec[thread_d_idx] * r_vec[thread_d_idx] * s_vec[thread_d_idx];
                __syncthreads(); 

                // Perform the reduction in smem.
                for (int offset = D / 2; offset > 0; offset >>= 1) {
                    if (thread_d_idx < offset) {
                        reduction_smem[thread_i_idx * D + thread_d_idx] += reduction_smem[thread_i_idx * D + thread_d_idx + offset];
                    }
                    __syncthreads();
                }
                // The final dot product is now in reduction_smem[thread_i_idx * D + 0].

                // --- Step 2.2: Redundant but Parallel Scalar Computation ---
                // All threads read the final dot product for curr I,J,K tile

                float dot = reduction_smem[thread_i_idx * D];
                float logit = dot * scale;

                float inv_lj = 1.0f / lj_tile[j_tile_idx]; 
                float inv_lk = 1.0f / lk_tile[k_tile_idx]; 
                float ar_val = expf(logit - mj_tile[j_tile_idx]) * inv_lj; //exp(dot - max) / sum of exps
                float as_val = expf(logit - mk_tile[k_tile_idx]) * inv_lk; 
                float combined_attn_val = ar_val * as_val; 
                
                // --- Step 2.3: Parallel Vector Update ---
                // NO __syncthreads() is needed here. The barrier stall is GONE.
                const float* vr_vec = vr_tile + j_tile_idx * D;
                const float* vs_vec = vs_tile + k_tile_idx * D;

                o_tile[thread_i_idx * D + thread_d_idx] += combined_attn_val * vr_vec[thread_d_idx] * vs_vec[thread_d_idx];
            }
        }
        __syncthreads(); // Sync after all j,k tiles are processed for this k0-block, before loading the next tile of K block.
    }
    
    // --- Write final result to global memory ---
    // Use atomicAdd to safely accumulate results
    float final_val = o_tile[thread_i_idx * D + thread_d_idx];
    if (i_idx < I && thread_d_idx < D) atomicAdd(&Y_q_[yq_bh_offset + i_idx * D + thread_d_idx], final_val);
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
        TORCH_CHECK(D * TILE_I_SPECIAL <= 1024, "D * TILE_I must be <= 1024 for the 2D block size.");
        // Change grid to be 3D: (I_tiles, J_tiles, B*H)
        dim3 grid(
            (I + TILE_I_SPECIAL - 1) / TILE_I_SPECIAL,
            (J + TILE_J - 1) / TILE_J,
            B * H
        );
        dim3 block(D, TILE_I_SPECIAL);
        size_t smem_size = sizeof(float) * (
            TILE_I_SPECIAL*D + TILE_J*D + TILE_K*D + 
            TILE_J*D + TILE_K*D +             
            TILE_J + TILE_J + TILE_K + TILE_K +
            TILE_I_SPECIAL*D // Add memory for the o_tile accumulator
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
    int B, int H, int I, int J, int K, int D, float scale)
{
    // --- Grid/Block Mapping ---
    const int j_tile_idx_grid = blockIdx.x;
    const int i_tile_idx_grid = blockIdx.y;
    const int bh_idx = blockIdx.z;

    const int d_idx = threadIdx.x;
    const int j_local_idx = threadIdx.y;

    const int j_base = j_tile_idx_grid * TILE_J_SPECIAL;
    const int j_idx = j_base + j_local_idx;

    const int i0 = i_tile_idx_grid * TILE_I;

    // Early exit for blocks outside the valid problem space
    if (j_idx >= J || i0 >= I) return;

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
    float* q_tile = r_tile + TILE_J_SPECIAL * D;
    float* s_tile = q_tile + TILE_I * D;
    float* vq_tile = s_tile + TILE_K * D;
    float* vs_tile = vq_tile + TILE_I * D;
    float* mi_tile = (float*)(vs_tile + TILE_K * D);
    float* li_tile = mi_tile + TILE_I;
    float* mk_tile = li_tile + TILE_I;
    float* lk_tile = mk_tile + TILE_K;
    float* o_tile = lk_tile + TILE_K;
    float* attn_scores_tile = o_tile + TILE_J_SPECIAL * D;

    int flat_thread_id_2d = threadIdx.y * blockDim.x + threadIdx.x;
    int threads_per_block = blockDim.x * blockDim.y;
    
    // Initialize the shared memory accumulator tile
    o_tile[j_local_idx * D + d_idx] = 0.0f;

    // --- Load R tile for this block ---
    for (int load_idx = flat_thread_id_2d; load_idx < TILE_J_SPECIAL * D; load_idx += threads_per_block) {
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
    float final_val = o_tile[j_local_idx * D + d_idx];
    if (d_idx < D) atomicAdd(&Y_r_[yr_bh_offset + j_idx * D + d_idx], final_val);
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
    
    Y_r_.zero_();

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
        TORCH_CHECK(D * TILE_J_SPECIAL <= 1024, "D * TILE_J_SPECIAL must be <= 1024 for the 2D block size.");
        dim3 grid(
            (J + TILE_J_SPECIAL - 1) / TILE_J_SPECIAL,
            (I + TILE_I - 1) / TILE_I,
            B * H
        );
        dim3 block(D, TILE_J_SPECIAL);
        size_t smem_size = sizeof(float) * (
            TILE_J_SPECIAL*D + TILE_I*D + TILE_K*D + 
            TILE_I*D + TILE_K*D +             
            TILE_I + TILE_I + TILE_K + TILE_K +
            TILE_J_SPECIAL*D
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
    const int i_tile_idx_grid = blockIdx.y;
    const int bh_idx = blockIdx.z;

    const int d_idx = threadIdx.x;
    const int k_local_idx = threadIdx.y;

    const int k_base = k_tile_idx_grid * TILE_K_SPECIAL;
    const int k_idx = k_base + k_local_idx;

    const int i0 = i_tile_idx_grid * TILE_I;

    // Early exit for blocks outside the valid problem space
    if (k_idx >= K || i0 >= I) return;

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
    float* q_tile = s_tile + TILE_K_SPECIAL * D;
    float* r_tile = q_tile + TILE_I * D;
    float* vq_tile = r_tile + TILE_J * D;
    float* vr_tile = vq_tile + TILE_I * D;
    float* mi_tile = (float*)(vr_tile + TILE_J * D);
    float* li_tile = mi_tile + TILE_I;
    float* mj_tile = li_tile + TILE_I;
    float* lj_tile = mj_tile + TILE_J;
    float* o_tile = lj_tile + TILE_J;
    float* attn_scores_tile = o_tile + TILE_K_SPECIAL * D;

    int flat_thread_id_2d = threadIdx.y * blockDim.x + threadIdx.x;
    int threads_per_block = blockDim.x * blockDim.y;
    
    o_tile[k_local_idx * D + d_idx] = 0.0f;

    // --- Load S tile for this block ---
    for (int load_idx = flat_thread_id_2d; load_idx < TILE_K_SPECIAL * D; load_idx += threads_per_block) {
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
    float final_val = o_tile[k_local_idx * D + d_idx];
    if (d_idx < D) atomicAdd(&Y_s_[ys_bh_offset + k_idx * D + d_idx], final_val);
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
    
    Y_s_.zero_();

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
        TORCH_CHECK(D * TILE_K_SPECIAL <= 1024, "D * TILE_K_SPECIAL must be <= 1024 for the 2D block size.");
        dim3 grid(
            (K + TILE_K_SPECIAL - 1) / TILE_K_SPECIAL,
            (I + TILE_I - 1) / TILE_I,
            B * H
        );
        dim3 block(D, TILE_K_SPECIAL);
        size_t smem_size = sizeof(float) * (
            TILE_K_SPECIAL*D + TILE_I*D + TILE_J*D + 
            TILE_I*D + TILE_J*D +             
            TILE_I + TILE_I + TILE_J + TILE_J +
            TILE_K_SPECIAL*D 
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

    // // GATHER
    // TORCH_CHECK(D % 4 == 0, "Head dimension D must be a multiple of 4 for the flash kernel.");
    // const int TpB = 128; // Threads per block
    // // Y_q Gather
    // {
    //     dim3 grid(I, H, B);
    //     dim3 block(TpB);
    //     size_t smem_size = sizeof(float) * (D + TILE_J*D + TILE_K*D + TILE_J*D + TILE_K*D + TILE_J*TILE_K + TpB + 2 + D);

    //     Yq_gather_flash_kernel<<<grid, block, smem_size>>>(
    //         reinterpret_cast<const float4*>(Q.data_ptr<float>()), 
    //         reinterpret_cast<const float4*>(R.data_ptr<float>()), 
    //         reinterpret_cast<const float4*>(S.data_ptr<float>()),
    //         reinterpret_cast<const float4*>(Vr_1.data_ptr<float>()), 
    //         reinterpret_cast<const float4*>(Vs_1.data_ptr<float>()),
    //         Y_q.data_ptr<float>(), 
    //         B, H, I, J, K, D, scale);
    // }
    // // Y_r Gather
    // {
    //     dim3 grid(J, H, B);
    //     dim3 block(TpB);
    //     size_t smem_size = sizeof(float) * (D + TILE_I*D + TILE_K*D + TILE_I*D + TILE_K*D + TILE_I*TILE_K + TpB + 2 + D);
    //     Yr_gather_flash_kernel<<<grid, block, smem_size>>>(
    //         reinterpret_cast<const float4*>(R.data_ptr<float>()), 
    //         reinterpret_cast<const float4*>(Q.data_ptr<float>()), 
    //         reinterpret_cast<const float4*>(S.data_ptr<float>()), 
    //         reinterpret_cast<const float4*>(Vq_1.data_ptr<float>()), 
    //         reinterpret_cast<const float4*>(Vs_1.data_ptr<float>()), 
    //         Y_r.data_ptr<float>(), 
    //         B, H, I, J, K, D, scale);
    // }
    // // Y_s Gather
    // {
    //     dim3 grid(K, H, B);
    //     dim3 block(TpB);
    //     size_t smem_size = sizeof(float) * (D + TILE_I*D + TILE_J*D + TILE_I*D + TILE_J*D + TILE_I*TILE_J + TpB + 2 + D);
    //     Ys_gather_flash_kernel<<<grid, block, smem_size>>>(
    //         reinterpret_cast<const float4*>(S.data_ptr<float>()), 
    //         reinterpret_cast<const float4*>(Q.data_ptr<float>()), 
    //         reinterpret_cast<const float4*>(R.data_ptr<float>()),
    //         reinterpret_cast<const float4*>(Vq_1.data_ptr<float>()), 
    //         reinterpret_cast<const float4*>(Vr_1.data_ptr<float>()), 
    //         Y_s.data_ptr<float>(), 
    //         B, H, I, J, K, D, scale);
    // }

    // SCATTER 
    Yq_scatter_smash_launcher(Q, R, S, Vr_2, Vs_2, Y_q_, scale);
    // Yr_scatter_flash_launcher(Q, R, S, Vq_2, Vs_2, Y_r_, scale);
    // Ys_scatter_flash_launcher(Q, R, S, Vq_2, Vr_2, Y_s_, scale);

    cudaDeviceSynchronize(); 
    return std::make_tuple(Y_q, Y_r, Y_s, Y_q_, Y_r_, Y_s_);
}




