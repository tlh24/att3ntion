/**
 * @file common.cuh
 * @brief Shared constants, utilities, and kernels for hypergraph attention.
 *
 * This header consolidates common definitions used by both forward and backward
 * CUDA kernels, eliminating duplication and ensuring consistency.
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

// =============================================================================
// Numerical Stability Constants
// =============================================================================

/** Maximum exponent value to prevent overflow in expf() */
constexpr float EXP_CLIP = 80.0f;

/** Epsilon to prevent division by zero in softmax denominators */
constexpr float DENOM_EPS = 1e-6f;

/** Large negative value for masked/padded attention scores */
constexpr float NEG_INF = -1e30f;

// =============================================================================
// Tile Size Configuration
// =============================================================================
// These control shared memory usage and parallelism. Tune based on GPU arch.

#ifndef TILE_I
#define TILE_I 16
#endif

#ifndef TILE_J
#define TILE_J 16
#endif

#ifndef TILE_K
#define TILE_K 16
#endif

// Smaller tiles for scatter kernels (higher register pressure)
#ifndef TILE_I_SCATTER
#define TILE_I_SCATTER 4
#endif

#ifndef TILE_J_SCATTER
#define TILE_J_SCATTER 4
#endif

#ifndef TILE_K_SCATTER
#define TILE_K_SCATTER 4
#endif

// Maximum embedding dimension that fits in registers
#ifndef MAX_D_REG
#define MAX_D_REG 64
#endif

// =============================================================================
// Utility Functions
// =============================================================================

/** Integer division rounded up */
__host__ __device__ __forceinline__ int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

/** Three-way element-wise dot product: sum(a[d] * b[d] * c[d]) */
__device__ __forceinline__ float dot3(
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ c,
    int D
) {
    float sum = 0.0f;
    #pragma unroll
    for (int d = 0; d < D; ++d) {
        sum += a[d] * b[d] * c[d];
    }
    return sum;
}

// =============================================================================
// Softmax Statistics Kernels
// =============================================================================
// These compute the running max (m) and sum-of-exponentials (l) for online
// softmax, used by both forward and backward passes.

/**
 * Computes softmax statistics for Aq = softmax_{j,k}(A)
 * Each block handles one i-index for a single (batch, head).
 */
static __global__ void softmax_stats_q(
    const float* __restrict__ Q,
    const float* __restrict__ R,
    const float* __restrict__ S,
    float* __restrict__ m_i_out,
    float* __restrict__ l_i_out,
    int B, int H, int I, int J, int K, int D, float scale
) {
    const int i_idx = blockIdx.x;
    const int bh_idx = blockIdx.y;

    if (i_idx >= I) return;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    // Batch-head offsets
    const int64_t q_bh_offset = (int64_t)bh_idx * I * D;
    const int64_t r_bh_offset = (int64_t)bh_idx * J * D;
    const int64_t s_bh_offset = (int64_t)bh_idx * K * D;

    const float* Q_slice = Q + q_bh_offset;
    const float* R_slice = R + r_bh_offset;
    const float* S_slice = S + s_bh_offset;

    // Shared memory layout
    extern __shared__ float smem[];
    float* q_vec_sh = smem;
    float* r_tile = q_vec_sh + D;
    float* s_tile = r_tile + TILE_J * D;
    float* p_tile = s_tile + TILE_K * D;
    float* red_buf = p_tile + TILE_J * TILE_K;

    // Load Q[i] into shared memory
    for (int d = tid; d < D; d += block_size) {
        q_vec_sh[d] = Q_slice[i_idx * D + d];
    }

    float m_block = NEG_INF;
    float l_block = 0.0f;

    __syncthreads();

    // Iterate over all (j, k) tiles
    for (int j0 = 0; j0 < J; j0 += TILE_J) {
        for (int k0 = 0; k0 < K; k0 += TILE_K) {
            // Cooperative tile loading
            for (int idx = tid; idx < TILE_J * D; idx += block_size) {
                int j_load = j0 + (idx / D);
                if (j_load < J) {
                    r_tile[idx] = R_slice[j_load * D + (idx % D)];
                }
            }
            for (int idx = tid; idx < TILE_K * D; idx += block_size) {
                int k_load = k0 + (idx / D);
                if (k_load < K) {
                    s_tile[idx] = S_slice[k_load * D + (idx % D)];
                }
            }
            __syncthreads();

            // Compute dot products
            for (int flat_idx = tid; flat_idx < TILE_J * TILE_K; flat_idx += block_size) {
                int jt = flat_idx / TILE_K;
                int kt = flat_idx % TILE_K;

                float dot = 0.0f;
                if (j0 + jt < J && k0 + kt < K) {
                    const float* r_vec = r_tile + jt * D;
                    const float* s_vec = s_tile + kt * D;
                    for (int d = 0; d < D; ++d) {
                        dot += q_vec_sh[d] * r_vec[d] * s_vec[d];
                    }
                    p_tile[flat_idx] = dot * scale;
                } else {
                    p_tile[flat_idx] = NEG_INF;
                }
            }
            __syncthreads();

            // Parallel reduction: find tile max
            float m_tile_thread = NEG_INF;
            for (int flat_idx = tid; flat_idx < TILE_J * TILE_K; flat_idx += block_size) {
                m_tile_thread = fmaxf(m_tile_thread, p_tile[flat_idx]);
            }
            red_buf[tid] = m_tile_thread;
            __syncthreads();
            for (int s = block_size / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    red_buf[tid] = fmaxf(red_buf[tid], red_buf[tid + s]);
                }
                __syncthreads();
            }
            float m_tile = red_buf[0];
            __syncthreads();

            // Parallel reduction: sum of exponentials
            float l_tile_thread = 0.0f;
            for (int flat_idx = tid; flat_idx < TILE_J * TILE_K; flat_idx += block_size) {
                if (p_tile[flat_idx] > NEG_INF + 1.0f) {
                    l_tile_thread += expf(fminf(p_tile[flat_idx] - m_tile, EXP_CLIP));
                }
            }
            red_buf[tid] = l_tile_thread;
            __syncthreads();
            for (int s = block_size / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    red_buf[tid] += red_buf[tid + s];
                }
                __syncthreads();
            }
            float l_tile = red_buf[0];

            // Online update
            float m_new = fmaxf(m_block, m_tile);
            l_block = expf(fminf(m_block - m_new, EXP_CLIP)) * l_block +
                      expf(fminf(m_tile - m_new, EXP_CLIP)) * l_tile;
            m_block = m_new;
            __syncthreads();
        }
    }

    // Write final stats
    if (tid == 0) {
        m_i_out[bh_idx * I + i_idx] = m_block;
        l_i_out[bh_idx * I + i_idx] = l_block;
    }
}

/**
 * Computes softmax statistics for Ar = softmax_{i,k}(A)
 * Each block handles one j-index for a single (batch, head).
 */
static __global__ void softmax_stats_r(
    const float* __restrict__ Q,
    const float* __restrict__ R,
    const float* __restrict__ S,
    float* __restrict__ m_j_out,
    float* __restrict__ l_j_out,
    int B, int H, int I, int J, int K, int D, float scale
) {
    const int j_idx = blockIdx.x;
    const int bh_idx = blockIdx.y;

    if (j_idx >= J) return;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    const int64_t q_bh_offset = (int64_t)bh_idx * I * D;
    const int64_t r_bh_offset = (int64_t)bh_idx * J * D;
    const int64_t s_bh_offset = (int64_t)bh_idx * K * D;

    const float* Q_slice = Q + q_bh_offset;
    const float* R_slice = R + r_bh_offset;
    const float* S_slice = S + s_bh_offset;

    extern __shared__ float smem[];
    float* r_vec_sh = smem;
    float* q_tile = r_vec_sh + D;
    float* s_tile = q_tile + TILE_I * D;
    float* p_tile = s_tile + TILE_K * D;
    float* red_buf = p_tile + TILE_I * TILE_K;

    // Load R[j] into shared memory
    for (int d = tid; d < D; d += block_size) {
        r_vec_sh[d] = R_slice[j_idx * D + d];
    }

    float m_block = NEG_INF;
    float l_block = 0.0f;

    __syncthreads();

    // Iterate over all (i, k) tiles
    for (int i0 = 0; i0 < I; i0 += TILE_I) {
        for (int k0 = 0; k0 < K; k0 += TILE_K) {
            for (int idx = tid; idx < TILE_I * D; idx += block_size) {
                int i_load = i0 + (idx / D);
                if (i_load < I) {
                    q_tile[idx] = Q_slice[i_load * D + (idx % D)];
                }
            }
            for (int idx = tid; idx < TILE_K * D; idx += block_size) {
                int k_load = k0 + (idx / D);
                if (k_load < K) {
                    s_tile[idx] = S_slice[k_load * D + (idx % D)];
                }
            }
            __syncthreads();

            for (int flat_idx = tid; flat_idx < TILE_I * TILE_K; flat_idx += block_size) {
                int it = flat_idx / TILE_K;
                int kt = flat_idx % TILE_K;

                float dot = 0.0f;
                if (i0 + it < I && k0 + kt < K) {
                    const float* q_vec = q_tile + it * D;
                    const float* s_vec = s_tile + kt * D;
                    for (int d = 0; d < D; ++d) {
                        dot += q_vec[d] * r_vec_sh[d] * s_vec[d];
                    }
                    p_tile[flat_idx] = dot * scale;
                } else {
                    p_tile[flat_idx] = NEG_INF;
                }
            }
            __syncthreads();

            float m_tile_thread = NEG_INF;
            for (int flat_idx = tid; flat_idx < TILE_I * TILE_K; flat_idx += block_size) {
                m_tile_thread = fmaxf(m_tile_thread, p_tile[flat_idx]);
            }
            red_buf[tid] = m_tile_thread;
            __syncthreads();
            for (int s = block_size / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    red_buf[tid] = fmaxf(red_buf[tid], red_buf[tid + s]);
                }
                __syncthreads();
            }
            float m_tile = red_buf[0];
            __syncthreads();

            float l_tile_thread = 0.0f;
            for (int flat_idx = tid; flat_idx < TILE_I * TILE_K; flat_idx += block_size) {
                if (p_tile[flat_idx] > NEG_INF + 1.0f) {
                    l_tile_thread += expf(fminf(p_tile[flat_idx] - m_tile, EXP_CLIP));
                }
            }
            red_buf[tid] = l_tile_thread;
            __syncthreads();
            for (int s = block_size / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    red_buf[tid] += red_buf[tid + s];
                }
                __syncthreads();
            }
            float l_tile = red_buf[0];

            float m_new = fmaxf(m_block, m_tile);
            l_block = expf(fminf(m_block - m_new, EXP_CLIP)) * l_block +
                      expf(fminf(m_tile - m_new, EXP_CLIP)) * l_tile;
            m_block = m_new;
            __syncthreads();
        }
    }

    if (tid == 0) {
        m_j_out[bh_idx * J + j_idx] = m_block;
        l_j_out[bh_idx * J + j_idx] = l_block;
    }
}

/**
 * Computes softmax statistics for As = softmax_{i,j}(A)
 * Each block handles one k-index for a single (batch, head).
 */
static __global__ void softmax_stats_s(
    const float* __restrict__ Q,
    const float* __restrict__ R,
    const float* __restrict__ S,
    float* __restrict__ m_k_out,
    float* __restrict__ l_k_out,
    int B, int H, int I, int J, int K, int D, float scale
) {
    const int k_idx = blockIdx.x;
    const int bh_idx = blockIdx.y;

    if (k_idx >= K) return;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    const int64_t q_bh_offset = (int64_t)bh_idx * I * D;
    const int64_t r_bh_offset = (int64_t)bh_idx * J * D;
    const int64_t s_bh_offset = (int64_t)bh_idx * K * D;

    const float* Q_slice = Q + q_bh_offset;
    const float* R_slice = R + r_bh_offset;
    const float* S_slice = S + s_bh_offset;

    extern __shared__ float smem[];
    float* s_vec_sh = smem;
    float* q_tile = s_vec_sh + D;
    float* r_tile = q_tile + TILE_I * D;
    float* p_tile = r_tile + TILE_J * D;
    float* red_buf = p_tile + TILE_I * TILE_J;

    // Load S[k] into shared memory
    for (int d = tid; d < D; d += block_size) {
        s_vec_sh[d] = S_slice[k_idx * D + d];
    }

    float m_block = NEG_INF;
    float l_block = 0.0f;

    __syncthreads();

    // Iterate over all (i, j) tiles
    for (int i0 = 0; i0 < I; i0 += TILE_I) {
        for (int j0 = 0; j0 < J; j0 += TILE_J) {
            for (int idx = tid; idx < TILE_I * D; idx += block_size) {
                int i_load = i0 + (idx / D);
                if (i_load < I) {
                    q_tile[idx] = Q_slice[i_load * D + (idx % D)];
                }
            }
            for (int idx = tid; idx < TILE_J * D; idx += block_size) {
                int j_load = j0 + (idx / D);
                if (j_load < J) {
                    r_tile[idx] = R_slice[j_load * D + (idx % D)];
                }
            }
            __syncthreads();

            for (int flat_idx = tid; flat_idx < TILE_I * TILE_J; flat_idx += block_size) {
                int it = flat_idx / TILE_J;
                int jt = flat_idx % TILE_J;

                float dot = 0.0f;
                if (i0 + it < I && j0 + jt < J) {
                    const float* q_vec = q_tile + it * D;
                    const float* r_vec = r_tile + jt * D;
                    for (int d = 0; d < D; ++d) {
                        dot += q_vec[d] * r_vec[d] * s_vec_sh[d];
                    }
                    p_tile[flat_idx] = dot * scale;
                } else {
                    p_tile[flat_idx] = NEG_INF;
                }
            }
            __syncthreads();

            float m_tile_thread = NEG_INF;
            for (int i = tid; i < TILE_I * TILE_J; i += block_size) {
                m_tile_thread = fmaxf(m_tile_thread, p_tile[i]);
            }
            red_buf[tid] = m_tile_thread;
            __syncthreads();
            for (int s = block_size / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    red_buf[tid] = fmaxf(red_buf[tid], red_buf[tid + s]);
                }
                __syncthreads();
            }
            float m_tile = red_buf[0];
            __syncthreads();

            float l_tile_thread = 0.0f;
            for (int i = tid; i < TILE_I * TILE_J; i += block_size) {
                if (p_tile[i] > NEG_INF + 1.0f) {
                    l_tile_thread += expf(fminf(p_tile[i] - m_tile, EXP_CLIP));
                }
            }
            red_buf[tid] = l_tile_thread;
            __syncthreads();
            for (int s = block_size / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    red_buf[tid] += red_buf[tid + s];
                }
                __syncthreads();
            }
            float l_tile = red_buf[0];

            float m_new = fmaxf(m_block, m_tile);
            l_block = expf(fminf(m_block - m_new, EXP_CLIP)) * l_block +
                      expf(fminf(m_tile - m_new, EXP_CLIP)) * l_tile;
            m_block = m_new;
            __syncthreads();
        }
    }

    if (tid == 0) {
        m_k_out[bh_idx * K + k_idx] = m_block;
        l_k_out[bh_idx * K + k_idx] = l_block;
    }
}

// =============================================================================
// Launcher Helpers for Softmax Stats
// =============================================================================

inline size_t softmax_stats_smem_Aq(int D) {
    return sizeof(float) * (D + TILE_J * D + TILE_K * D + TILE_J * TILE_K + 256);
}

inline size_t softmax_stats_smem_Ar(int D) {
    return sizeof(float) * (D + TILE_I * D + TILE_K * D + TILE_I * TILE_K + 256);
}

inline size_t softmax_stats_smem_As(int D) {
    return sizeof(float) * (D + TILE_I * D + TILE_J * D + TILE_I * TILE_J + 256);
}
