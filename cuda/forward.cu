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
constexpr int MIN_SPLIT_CHUNKS = 4;  // Only split loops across blocks when >= this many chunks
constexpr int MAX_SPLIT_CHUNKS = 16; // Cap split workspace growth and reducer scratch.

// Multi-i warp-parallel gather: 4 output vectors per block, warp-shuffle softmax
template<int D_CONST>
__global__
void Yq_gather(
    const bf16* __restrict__ Q_bf,
    const bf16* __restrict__ R_bf,
    const bf16* __restrict__ S_bf,
    const bf16* __restrict__ V1_bf,
    const bf16* __restrict__ V2_bf,
    float*       __restrict__ Y,
    float*       __restrict__ m_i_out,
    float*       __restrict__ l_i_out,
    int B, int H, int I, int J, int K, float scale,
    int num_j_chunks)
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
    constexpr int DP = D_CONST + SMEM_PAD;

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
        for (int k0 = 0; k0 < K; k0 += TILE_K) {
            
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
                    
                    float dot = 0.0f;
                    if (my_i_valid && j0 + jt < J && k0 + kt < K) {
                        float d_accum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                        #pragma unroll 4
                        for (int d = 0; d < D_CONST; d += 4) {
                            d_accum[0] += bf2f(my_q[d+0]) * bf2f(r_tile[jt*DP + d+0]) * bf2f(s_tile[kt*DP + d+0]);
                            d_accum[1] += bf2f(my_q[d+1]) * bf2f(r_tile[jt*DP + d+1]) * bf2f(s_tile[kt*DP + d+1]);
                            d_accum[2] += bf2f(my_q[d+2]) * bf2f(r_tile[jt*DP + d+2]) * bf2f(s_tile[kt*DP + d+2]);
                            d_accum[3] += bf2f(my_q[d+3]) * bf2f(r_tile[jt*DP + d+3]) * bf2f(s_tile[kt*DP + d+3]);
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
                    // Factored output: T[kt] = sum_jt P[jt,kt]*V1[jt,d], then O[d] = sum_kt T[kt]*V2[kt,d]
                    // Eliminates V2 reads from inner loop (256 → 16 V2 smem reads per d-value)
                    float T_k[TILE_K];
                    #pragma unroll
                    for (int kt = 0; kt < TILE_K; kt++) T_k[kt] = 0.0f;

                    for (int jt = 0; jt < TILE_J; jt++) {
                        if (j0 + jt >= J) continue;
                        float v1_val = bf2f(v1_tile[jt * DP + d]);
                        #pragma unroll
                        for (int kt = 0; kt < TILE_K; kt++) {
                            T_k[kt] += my_p[jt * TILE_K + kt] * v1_val;
                        }
                    }

                    #pragma unroll
                    for (int kt = 0; kt < TILE_K; kt++) {
                        if (k0 + kt < K) {
                            new_o_d += T_k[kt] * bf2f(v2_tile[kt * DP + d]);
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
    const bf16* __restrict__ R_query_bf,
    const bf16* __restrict__ Q_bf,
    const bf16* __restrict__ S_bf,
    const bf16* __restrict__ V1_bf,
    const bf16* __restrict__ V2_bf,
    float*       __restrict__ Y,
    float*       __restrict__ m_j_out,
    float*       __restrict__ l_j_out,
    int B, int H, int I, int J, int K, float scale,
    int num_i_chunks)
{
    const int i_chunk = blockIdx.x % num_i_chunks;
    const int j_base = (blockIdx.x / num_i_chunks) * N_I_GATHER;
    const int h = blockIdx.y;
    const int b = blockIdx.z;

    // Compute i-range for this chunk
    const int total_i_tiles = (I + TILE_I - 1) / TILE_I;
    const int i_tpc = (total_i_tiles + num_i_chunks - 1) / num_i_chunks;
    const int i_start = i_chunk * i_tpc * TILE_I;
    const int i_end = min(i_start + i_tpc * TILE_I, I);

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int block_size = blockDim.x;
    constexpr int DP = D_CONST + SMEM_PAD;

    extern __shared__ char smem_raw[];
    bf16* q_vecs  = reinterpret_cast<bf16*>(smem_raw);
    bf16* i_tile  = q_vecs + N_I_GATHER * D_CONST;
    bf16* k_tile  = i_tile + TILE_I * DP;
    bf16* v1_tile = k_tile + TILE_K * DP;
    bf16* v2_tile = v1_tile + TILE_I * DP;
    float* p_tiles = reinterpret_cast<float*>(v2_tile + TILE_K * DP);
    float* m_l_sh  = p_tiles + N_I_GATHER * TILE_I * TILE_K;
    float* o_sh    = m_l_sh + N_I_GATHER * 2;
    
    const int my_j = j_base + warp_id;
    const bool my_j_valid = (my_j < J);
    bf16* my_q = q_vecs + warp_id * D_CONST;
    float* my_p = p_tiles + warp_id * TILE_I * TILE_K;
    float* my_o = o_sh + warp_id * D_CONST;
    float* my_ml = m_l_sh + warp_id * 2;

    for (int n = 0; n < N_I_GATHER; n++) {
        int j_global = j_base + n;
        if (j_global < J) {
            const int64_t q_off = (((int64_t)b * H + h) * J + j_global) * D_CONST;
            for (int d = tid; d < D_CONST; d += block_size) {
                q_vecs[n * D_CONST + d] = R_query_bf[q_off + d];
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

    for (int i0 = i_start; i0 < i_end; i0 += TILE_I) {
        for (int k0 = 0; k0 < K; k0 += TILE_K) {
            
            for (int idx = tid; idx < TILE_I * D_CONST; idx += block_size) {
                int it = idx / D_CONST;
                int d = idx % D_CONST;
                int i_global = i0 + it;
                if (i_global < I) {
                    const int64_t i_off = (((int64_t)b * H + h) * I + i_global) * D_CONST + d;
                    i_tile[it * DP + d] = Q_bf[i_off];
                    v1_tile[it * DP + d] = V1_bf[i_off];
                }
            }
            
            for (int idx = tid; idx < TILE_K * D_CONST; idx += block_size) {
                int kt = idx / D_CONST;
                int d = idx % D_CONST;
                int k_global = k0 + kt;
                if (k_global < K) {
                    const int64_t k_off = (((int64_t)b * H + h) * K + k_global) * D_CONST + d;
                    k_tile[kt * DP + d] = S_bf[k_off];
                    v2_tile[kt * DP + d] = V2_bf[k_off];
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
                            d_accum[0] += bf2f(my_q[d+0]) * bf2f(i_tile[it*DP + d+0]) * bf2f(k_tile[kt*DP + d+0]);
                            d_accum[1] += bf2f(my_q[d+1]) * bf2f(i_tile[it*DP + d+1]) * bf2f(k_tile[kt*DP + d+1]);
                            d_accum[2] += bf2f(my_q[d+2]) * bf2f(i_tile[it*DP + d+2]) * bf2f(k_tile[kt*DP + d+2]);
                            d_accum[3] += bf2f(my_q[d+3]) * bf2f(i_tile[it*DP + d+3]) * bf2f(k_tile[kt*DP + d+3]);
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
                    // Factored: T[kt] = sum_it P[it,kt]*V1[it,d], then O[d] = sum_kt T[kt]*V2[kt,d]
                    float T_k[TILE_K];
                    #pragma unroll
                    for (int kt = 0; kt < TILE_K; kt++) T_k[kt] = 0.0f;

                    for (int it = 0; it < TILE_I; it++) {
                        if (i0 + it >= I) continue;
                        float v1_val = bf2f(v1_tile[it * DP + d]);
                        #pragma unroll
                        for (int kt = 0; kt < TILE_K; kt++) {
                            T_k[kt] += my_p[it * TILE_K + kt] * v1_val;
                        }
                    }

                    #pragma unroll
                    for (int kt = 0; kt < TILE_K; kt++) {
                        if (k0 + kt < K) {
                            new_o_d += T_k[kt] * bf2f(v2_tile[kt * DP + d]);
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
            const int64_t out_off = ((((int64_t)b * H + h) * J + j_global) * num_i_chunks + i_chunk) * D_CONST;
            for (int d = tid; d < D_CONST; d += block_size) {
                Y[out_off + d] = o_sh[n * D_CONST + d];
            }
        }
    }

    if (lane_id == 0 && m_j_out != nullptr && l_j_out != nullptr && my_j_valid) {
        int64_t stats_idx = (((int64_t)b * H + h) * J + my_j) * num_i_chunks + i_chunk;
        m_j_out[stats_idx] = my_ml[0];
        l_j_out[stats_idx] = my_ml[1];
    }
}


// Multi-k warp-parallel gather: 4 output vectors per block, warp-shuffle softmax
template<int D_CONST>
__global__
void Ys_gather(
    const bf16* __restrict__ S_query_bf,
    const bf16* __restrict__ Q_bf,
    const bf16* __restrict__ R_bf,
    const bf16* __restrict__ V1_bf,
    const bf16* __restrict__ V2_bf,
    float*       __restrict__ Y,
    float*       __restrict__ m_k_out,
    float*       __restrict__ l_k_out,
    int B, int H, int I, int J, int K, float scale,
    int num_i_chunks)
{
    const int i_chunk = blockIdx.x % num_i_chunks;
    const int k_base = (blockIdx.x / num_i_chunks) * N_I_GATHER;
    const int h = blockIdx.y;
    const int b = blockIdx.z;

    // Compute i-range for this chunk
    const int total_i_tiles = (I + TILE_I - 1) / TILE_I;
    const int i_tpc = (total_i_tiles + num_i_chunks - 1) / num_i_chunks;
    const int i_start = i_chunk * i_tpc * TILE_I;
    const int i_end = min(i_start + i_tpc * TILE_I, I);

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int block_size = blockDim.x;
    constexpr int DP = D_CONST + SMEM_PAD;

    extern __shared__ char smem_raw[];
    bf16* q_vecs  = reinterpret_cast<bf16*>(smem_raw);
    bf16* i_tile  = q_vecs + N_I_GATHER * D_CONST;
    bf16* j_tile  = i_tile + TILE_I * DP;
    bf16* v1_tile = j_tile + TILE_J * DP;
    bf16* v2_tile = v1_tile + TILE_I * DP;
    float* p_tiles = reinterpret_cast<float*>(v2_tile + TILE_J * DP);
    float* m_l_sh  = p_tiles + N_I_GATHER * TILE_I * TILE_J;
    float* o_sh    = m_l_sh + N_I_GATHER * 2;
    
    const int my_k = k_base + warp_id;
    const bool my_k_valid = (my_k < K);
    bf16* my_q = q_vecs + warp_id * D_CONST;
    float* my_p = p_tiles + warp_id * TILE_I * TILE_J;
    float* my_o = o_sh + warp_id * D_CONST;
    float* my_ml = m_l_sh + warp_id * 2;

    for (int n = 0; n < N_I_GATHER; n++) {
        int k_global = k_base + n;
        if (k_global < K) {
            const int64_t q_off = (((int64_t)b * H + h) * K + k_global) * D_CONST;
            for (int d = tid; d < D_CONST; d += block_size) {
                q_vecs[n * D_CONST + d] = S_query_bf[q_off + d];
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

    for (int i0 = i_start; i0 < i_end; i0 += TILE_I) {
        for (int j0 = 0; j0 < J; j0 += TILE_J) {
            
            for (int idx = tid; idx < TILE_I * D_CONST; idx += block_size) {
                int it = idx / D_CONST;
                int d = idx % D_CONST;
                int i_global = i0 + it;
                if (i_global < I) {
                    const int64_t i_off = (((int64_t)b * H + h) * I + i_global) * D_CONST + d;
                    i_tile[it * DP + d] = Q_bf[i_off];
                    v1_tile[it * DP + d] = V1_bf[i_off];
                }
            }
            
            for (int idx = tid; idx < TILE_J * D_CONST; idx += block_size) {
                int jt = idx / D_CONST;
                int d = idx % D_CONST;
                int j_global = j0 + jt;
                if (j_global < J) {
                    const int64_t j_off = (((int64_t)b * H + h) * J + j_global) * D_CONST + d;
                    j_tile[jt * DP + d] = R_bf[j_off];
                    v2_tile[jt * DP + d] = V2_bf[j_off];
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
                            d_accum[0] += bf2f(my_q[d+0]) * bf2f(i_tile[it*DP + d+0]) * bf2f(j_tile[jt*DP + d+0]);
                            d_accum[1] += bf2f(my_q[d+1]) * bf2f(i_tile[it*DP + d+1]) * bf2f(j_tile[jt*DP + d+1]);
                            d_accum[2] += bf2f(my_q[d+2]) * bf2f(i_tile[it*DP + d+2]) * bf2f(j_tile[jt*DP + d+2]);
                            d_accum[3] += bf2f(my_q[d+3]) * bf2f(i_tile[it*DP + d+3]) * bf2f(j_tile[jt*DP + d+3]);
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
                    // Factored: T[jt] = sum_it P[it,jt]*V1[it,d], then O[d] = sum_jt T[jt]*V2[jt,d]
                    float T_j[TILE_J];
                    #pragma unroll
                    for (int jt = 0; jt < TILE_J; jt++) T_j[jt] = 0.0f;

                    for (int it = 0; it < TILE_I; it++) {
                        if (i0 + it >= I) continue;
                        float v1_val = bf2f(v1_tile[it * DP + d]);
                        #pragma unroll
                        for (int jt = 0; jt < TILE_J; jt++) {
                            T_j[jt] += my_p[it * TILE_J + jt] * v1_val;
                        }
                    }

                    #pragma unroll
                    for (int jt = 0; jt < TILE_J; jt++) {
                        if (j0 + jt < J) {
                            new_o_d += T_j[jt] * bf2f(v2_tile[jt * DP + d]);
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
            const int64_t out_off = ((((int64_t)b * H + h) * K + k_global) * num_i_chunks + i_chunk) * D_CONST;
            for (int d = tid; d < D_CONST; d += block_size) {
                Y[out_off + d] = o_sh[n * D_CONST + d];
            }
        }
    }

    if (lane_id == 0 && m_k_out != nullptr && l_k_out != nullptr && my_k_valid) {
        int64_t stats_idx = (((int64_t)b * H + h) * K + my_k) * num_i_chunks + i_chunk;
        m_k_out[stats_idx] = my_ml[0];
        l_k_out[stats_idx] = my_ml[1];
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
    for (int c = 0; c < num_chunks; c++) {
        alpha_l[c] = expf(alpha_l[c] - m_global) * l_partial[ml_off + c];
        l_global += alpha_l[c];
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
    const bf16* __restrict__ Q,
    const bf16* __restrict__ R,
    const bf16* __restrict__ S,
    const bf16* __restrict__ Vr_2,
    const bf16* __restrict__ Vs_2,
    const float* __restrict__ m_j_in,
    const float* __restrict__ l_j_in,
    const float* __restrict__ m_k_in,
    const float* __restrict__ l_k_in,
    float* __restrict__ Y_q_,
    int B, int H, int I, int J, int K, float scale,
    int num_j_chunks
) {
    // --- Grid/Block Mapping (split-J) ---
    const int b = blockIdx.z;
    const int h = blockIdx.y;
    const int j_chunk = blockIdx.x % num_j_chunks;
    const int i_start = (blockIdx.x / num_j_chunks) * TILE_I;

    // Compute J-range for this chunk
    const int total_j_tiles = (J + TILE_J - 1) / TILE_J;
    const int j_tpc = (total_j_tiles + num_j_chunks - 1) / num_j_chunks;
    const int j_loop_start = j_chunk * j_tpc * TILE_J;
    const int j_loop_end = min(j_loop_start + j_tpc * TILE_J, J);

    const int tid = threadIdx.x;
    const int tpb = blockDim.x;

    // --- Memory Offsets ---
    const int64_t q_bh_offset = (int64_t)(b * H + h) * I * D_CONST;
    const int64_t r_bh_offset = (int64_t)(b * H + h) * J * D_CONST;
    const int64_t s_bh_offset = (int64_t)(b * H + h) * K * D_CONST;
    const int64_t mj_bh_offset = (int64_t)(b * H + h) * J;
    const int64_t mk_bh_offset = (int64_t)(b * H + h) * K;

    // --- Shared Memory Layout ---
    extern __shared__ char smem_raw[];
    bf16* q_tile = reinterpret_cast<bf16*>(smem_raw);
    bf16* r_tile = q_tile + TILE_I * D_CONST;
    bf16* s_tile = r_tile + TILE_J * D_CONST;
    bf16* vr_tile = s_tile + TILE_K * D_CONST;
    bf16* vs_tile = vr_tile + TILE_J * D_CONST;
    float* attn_tile = reinterpret_cast<float*>(vs_tile + TILE_K * D_CONST);
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

    // --- Main Loop: Iterate Over J Tiles (split range) ---
    for (int jt = j_loop_start; jt < j_loop_end; jt += TILE_J) {
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
                    qa[u] = bf2f(q_tile[(ia * 4 + u) * D_CONST + d_idx]);
                    ra[u] = bf2f(r_tile[(ja * 4 + u) * D_CONST + d_idx]);
                    sa[u] = bf2f(s_tile[(ka * 4 + u) * D_CONST + d_idx]);
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
                        float vrt = bf2f(vr_tile[jy * D_CONST + dy]);
                        for (int ky = 0; ky < TILE_K; ky++) {
                            f += attn_tile[iy * TILE_J * TILE_K + jy * TILE_K + ky] * vrt * bf2f(vs_tile[ky * D_CONST + dy]);
                        }
                    }
                    yq_acc[n] += f;
                }
            }
            __syncthreads();
        }
    }

    // --- Write Output to Global Memory (chunked layout) ---
    const int64_t bh_I = q_bh_offset / D_CONST;  // = (b*H + h) * I
    for (int n = 0; n < load_iters; n++) {
        int tid_n = tid + n * tpb;
        if (tid_n < TILE_I * D_CONST) {
            int iy = tid_n / D_CONST;
            int dy = tid_n % D_CONST;
            int i_global = i_start + iy;
            if (i_global < I) {
                int64_t out_off = (bh_I + i_global) * (int64_t)num_j_chunks * D_CONST
                                + (int64_t)j_chunk * D_CONST + dy;
                Y_q_[out_off] = yq_acc[n];
            }
        }
    }
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
    const bf16* __restrict__ Q,
    const bf16* __restrict__ R,
    const bf16* __restrict__ S,
    const bf16* __restrict__ Vq_2,
    const bf16* __restrict__ Vs_2,
    const float* __restrict__ m_i_in,
    const float* __restrict__ l_i_in,
    const float* __restrict__ m_k_in,
    const float* __restrict__ l_k_in,
    float* __restrict__ Y_r_,
    int B, int H, int I, int J, int K, float scale,
    int num_i_chunks
) {
    // --- Grid/Block Mapping (split-I) ---
    const int b = blockIdx.z;
    const int h = blockIdx.y;
    const int i_chunk = blockIdx.x % num_i_chunks;
    const int j_start = (blockIdx.x / num_i_chunks) * TILE_J;

    // Compute I-range for this chunk
    const int total_i_tiles = (I + TILE_I - 1) / TILE_I;
    const int i_tpc = (total_i_tiles + num_i_chunks - 1) / num_i_chunks;
    const int i_loop_start = i_chunk * i_tpc * TILE_I;
    const int i_loop_end = min(i_loop_start + i_tpc * TILE_I, I);

    const int tid = threadIdx.x;
    const int tpb = blockDim.x;

    // --- Memory Offsets ---
    const int64_t r_bh_offset = (int64_t)(b * H + h) * J * D_CONST;
    const int64_t q_bh_offset = (int64_t)(b * H + h) * I * D_CONST;
    const int64_t s_bh_offset = (int64_t)(b * H + h) * K * D_CONST;
    const int64_t mi_bh_offset = (int64_t)(b * H + h) * I;
    const int64_t mk_bh_offset = (int64_t)(b * H + h) * K;

    // --- Shared Memory Layout ---
    extern __shared__ char smem_raw[];
    bf16* r_tile = reinterpret_cast<bf16*>(smem_raw);
    bf16* q_tile = r_tile + TILE_J * D_CONST;
    bf16* s_tile = q_tile + TILE_I * D_CONST;
    bf16* vq_tile = s_tile + TILE_K * D_CONST;
    bf16* vs_tile = vq_tile + TILE_I * D_CONST;
    float* attn_tile = reinterpret_cast<float*>(vs_tile + TILE_K * D_CONST);
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

    // --- Main Loop: Iterate Over I Tiles (split range) ---
    for (int it = i_loop_start; it < i_loop_end; it += TILE_I) {
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
                    qa[u] = bf2f(q_tile[(ia * 4 + u) * D_CONST + d_idx]);
                    ra[u] = bf2f(r_tile[(ja * 4 + u) * D_CONST + d_idx]);
                    sa[u] = bf2f(s_tile[(ka * 4 + u) * D_CONST + d_idx]);
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
                        float vqt = bf2f(vq_tile[iy * D_CONST + dy]);
                        for (int ky = 0; ky < TILE_K; ky++) {
                            f += attn_tile[iy * TILE_J * TILE_K + jy * TILE_K + ky] * vqt * bf2f(vs_tile[ky * D_CONST + dy]);
                        }
                    }
                    yr_acc[n] += f;
                }
            }
            __syncthreads();
        }
    }

    // --- Write Output to Global Memory (chunked layout) ---
    const int64_t bh_J = r_bh_offset / D_CONST;  // = (b*H + h) * J
    for (int n = 0; n < load_iters; n++) {
        int tid_n = tid + n * tpb;
        if (tid_n < TILE_J * D_CONST) {
            int jy = tid_n / D_CONST;
            int dy = tid_n % D_CONST;
            int j_global = j_start + jy;
            if (j_global < J) {
                int64_t out_off = (bh_J + j_global) * (int64_t)num_i_chunks * D_CONST
                                + (int64_t)i_chunk * D_CONST + dy;
                Y_r_[out_off] = yr_acc[n];
            }
        }
    }
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
    const bf16* __restrict__ Q,
    const bf16* __restrict__ R,
    const bf16* __restrict__ S,
    const bf16* __restrict__ Vq_2,
    const bf16* __restrict__ Vr_2,
    const float* __restrict__ m_i_in,
    const float* __restrict__ l_i_in,
    const float* __restrict__ m_j_in,
    const float* __restrict__ l_j_in,
    float* __restrict__ Y_s_,
    int B, int H, int I, int J, int K, float scale,
    int num_i_chunks
) {
    // --- Grid/Block Mapping (split-I) ---
    const int b = blockIdx.z;
    const int h = blockIdx.y;
    const int i_chunk = blockIdx.x % num_i_chunks;
    const int k_start = (blockIdx.x / num_i_chunks) * TILE_K;

    // Compute I-range for this chunk
    const int total_i_tiles = (I + TILE_I - 1) / TILE_I;
    const int i_tpc = (total_i_tiles + num_i_chunks - 1) / num_i_chunks;
    const int i_loop_start = i_chunk * i_tpc * TILE_I;
    const int i_loop_end = min(i_loop_start + i_tpc * TILE_I, I);

    const int tid = threadIdx.x;
    const int tpb = blockDim.x;

    // --- Memory Offsets ---
    const int64_t s_bh_offset = (int64_t)(b * H + h) * K * D_CONST;
    const int64_t q_bh_offset = (int64_t)(b * H + h) * I * D_CONST;
    const int64_t r_bh_offset = (int64_t)(b * H + h) * J * D_CONST;
    const int64_t mi_bh_offset = (int64_t)(b * H + h) * I;
    const int64_t mj_bh_offset = (int64_t)(b * H + h) * J;

    // --- Shared Memory Layout ---
    extern __shared__ char smem_raw[];
    bf16* s_tile = reinterpret_cast<bf16*>(smem_raw);
    bf16* q_tile = s_tile + TILE_K * D_CONST;
    bf16* r_tile = q_tile + TILE_I * D_CONST;
    bf16* vq_tile = r_tile + TILE_J * D_CONST;
    bf16* vr_tile = vq_tile + TILE_I * D_CONST;
    float* attn_tile = reinterpret_cast<float*>(vr_tile + TILE_J * D_CONST);
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

    // --- Main Loop: Iterate Over I Tiles (split range) ---
    for (int it = i_loop_start; it < i_loop_end; it += TILE_I) {
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
                    qa[u] = bf2f(q_tile[(ia * 4 + u) * D_CONST + d_idx]);
                    ra[u] = bf2f(r_tile[(ja * 4 + u) * D_CONST + d_idx]);
                    sa[u] = bf2f(s_tile[(ka * 4 + u) * D_CONST + d_idx]);
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
                        float vqt = bf2f(vq_tile[iy * D_CONST + dy]);
                        for (int jy = 0; jy < TILE_J; jy++) {
                            f += attn_tile[iy * TILE_J * TILE_K + jy * TILE_K + ky] * vqt * bf2f(vr_tile[jy * D_CONST + dy]);
                        }
                    }
                    ys_acc[n] += f;
                }
            }
            __syncthreads();
        }
    }

    // --- Write Output to Global Memory (chunked layout) ---
    const int64_t bh_K = s_bh_offset / D_CONST;  // = (b*H + h) * K
    for (int n = 0; n < load_iters; n++) {
        int tid_n = tid + n * tpb;
        if (tid_n < TILE_K * D_CONST) {
            int ky = tid_n / D_CONST;
            int dy = tid_n % D_CONST;
            int k_global = k_start + ky;
            if (k_global < K) {
                int64_t out_off = (bh_K + k_global) * (int64_t)num_i_chunks * D_CONST
                                + (int64_t)i_chunk * D_CONST + dy;
                Y_s_[out_off] = ys_acc[n];
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

    // Scatter shared memory size (same for all 3 scatter kernels)
    const size_t scatter_smem_size =
        sizeof(bf16) * (
            TILE_I * D_TMPL +              // q/r/s_tile (output dim)
            TILE_J * D_TMPL +              // r/q/q_tile
            TILE_K * D_TMPL +              // s/s/r_tile
            TILE_J * D_TMPL +              // vr/vq/vq_tile  (reused for second value dim)
            TILE_K * D_TMPL                // vs/vs/vr_tile
        ) +
        sizeof(float) * (
            TILE_I * TILE_J * TILE_K +     // attn_tile
            TILE_I + TILE_I +              // m1_tile, l1_tile (first stats dim)
            TILE_K + TILE_K                // m2_tile, l2_tile (second stats dim)
        );

    // GATHER: Y_q on stream 0 (split over J)
    {
        const int num_i_tiles = (I + N_I_GATHER - 1) / N_I_GATHER;
        dim3 grid_yq(num_i_tiles * num_j_chunks_q, H, B);
        
        constexpr int DP = D_TMPL + SMEM_PAD;
        size_t smem_size =
            sizeof(bf16) * (
                N_I_GATHER * D_TMPL +         // q_vecs
                TILE_J * DP +                 // r_tile (padded)
                TILE_K * DP +                 // s_tile (padded)
                TILE_J * DP +                 // v1_tile (padded)
                TILE_K * DP                   // v2_tile (padded)
            ) +
            sizeof(float) * (
                N_I_GATHER * TILE_J * TILE_K +// p_tiles
                N_I_GATHER * 2 +              // m_l_sh
                N_I_GATHER * D_TMPL           // o_sh
            );

        Yq_gather<D_TMPL><<<grid_yq, block, smem_size, streams[0]>>>(
            reinterpret_cast<const bf16*>(Q.data_ptr<at::BFloat16>()),
            reinterpret_cast<const bf16*>(R.data_ptr<at::BFloat16>()),
            reinterpret_cast<const bf16*>(S.data_ptr<at::BFloat16>()),
            reinterpret_cast<const bf16*>(Vr_1.data_ptr<at::BFloat16>()),
            reinterpret_cast<const bf16*>(Vs_1.data_ptr<at::BFloat16>()),
            O_part_q.data_ptr<float>(),
            m_part_q.data_ptr<float>(),
            l_part_q.data_ptr<float>(),
            B, H, I, J, K, scale,
            num_j_chunks_q
        );

        reduce_gather_partials<D_TMPL><<<dim3(I, H, B), D_TMPL, 0, streams[0]>>>(
            O_part_q.data_ptr<float>(),
            m_part_q.data_ptr<float>(),
            l_part_q.data_ptr<float>(),
            reinterpret_cast<bf16*>(Y_q.data_ptr<at::BFloat16>()),
            m_i.data_ptr<float>(),
            l_i.data_ptr<float>(),
            I, num_j_chunks_q
        );
    }
    AT_CUDA_CHECK(cudaEventRecord(gather_done[0], streams[0]));
    
    // GATHER: Y_r on stream 1 (split over I)
    {
        const int num_j_tiles = (J + N_I_GATHER - 1) / N_I_GATHER;
        dim3 grid(num_j_tiles * num_i_chunks_r, H, B);
        constexpr int DP = D_TMPL + SMEM_PAD;
        size_t smem_size =
            sizeof(bf16) * (
                N_I_GATHER * D_TMPL +
                TILE_I * DP + TILE_K * DP +
                TILE_I * DP + TILE_K * DP
            ) +
            sizeof(float) * (
                N_I_GATHER * TILE_I * TILE_K + N_I_GATHER * 2 + N_I_GATHER * D_TMPL
            );

        Yr_gather<D_TMPL><<<grid, block, smem_size, streams[1]>>>(
            reinterpret_cast<const bf16*>(R.data_ptr<at::BFloat16>()),
            reinterpret_cast<const bf16*>(Q.data_ptr<at::BFloat16>()),
            reinterpret_cast<const bf16*>(S.data_ptr<at::BFloat16>()),
            reinterpret_cast<const bf16*>(Vq_1.data_ptr<at::BFloat16>()),
            reinterpret_cast<const bf16*>(Vs_1.data_ptr<at::BFloat16>()),
            O_part_r.data_ptr<float>(),
            m_part_r.data_ptr<float>(),
            l_part_r.data_ptr<float>(),
            B, H, I, J, K, scale,
            num_i_chunks_r
        );

        reduce_gather_partials<D_TMPL><<<dim3(J, H, B), D_TMPL, 0, streams[1]>>>(
            O_part_r.data_ptr<float>(),
            m_part_r.data_ptr<float>(),
            l_part_r.data_ptr<float>(),
            reinterpret_cast<bf16*>(Y_r.data_ptr<at::BFloat16>()),
            m_j.data_ptr<float>(),
            l_j.data_ptr<float>(),
            J, num_i_chunks_r
        );
    }
    AT_CUDA_CHECK(cudaEventRecord(gather_done[1], streams[1]));

    // GATHER: Y_s on stream 2 (split over I)
    {
        const int num_k_tiles = (K + N_I_GATHER - 1) / N_I_GATHER;
        dim3 grid(num_k_tiles * num_i_chunks_s, H, B);
        constexpr int DP = D_TMPL + SMEM_PAD;
        size_t smem_size =
            sizeof(bf16) * (
                N_I_GATHER * D_TMPL +
                TILE_I * DP + TILE_J * DP +
                TILE_I * DP + TILE_J * DP
            ) +
            sizeof(float) * (
                N_I_GATHER * TILE_I * TILE_J + N_I_GATHER * 2 + N_I_GATHER * D_TMPL
            );

        Ys_gather<D_TMPL><<<grid, block, smem_size, streams[2]>>>(
            reinterpret_cast<const bf16*>(S.data_ptr<at::BFloat16>()),
            reinterpret_cast<const bf16*>(Q.data_ptr<at::BFloat16>()),
            reinterpret_cast<const bf16*>(R.data_ptr<at::BFloat16>()),
            reinterpret_cast<const bf16*>(Vq_1.data_ptr<at::BFloat16>()),
            reinterpret_cast<const bf16*>(Vr_1.data_ptr<at::BFloat16>()),
            O_part_s.data_ptr<float>(),
            m_part_s.data_ptr<float>(),
            l_part_s.data_ptr<float>(),
            B, H, I, J, K, scale,
            num_i_chunks_s
        );

        reduce_gather_partials<D_TMPL><<<dim3(K, H, B), D_TMPL, 0, streams[2]>>>(
            O_part_s.data_ptr<float>(),
            m_part_s.data_ptr<float>(),
            l_part_s.data_ptr<float>(),
            reinterpret_cast<bf16*>(Y_s.data_ptr<at::BFloat16>()),
            m_k.data_ptr<float>(),
            l_k.data_ptr<float>(),
            K, num_i_chunks_s
        );
    }
    AT_CUDA_CHECK(cudaEventRecord(gather_done[2], streams[2]));

    // Barrier: all scatter kernels wait for all gather kernels to complete
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            AT_CUDA_CHECK(cudaStreamWaitEvent(streams[i], gather_done[j], 0));
        }
    }

    // SCATTER kernels (split outer loops for occupancy, additive reduce).
    TORCH_CHECK(I == J && J == K, "Scatter requires I == J == K");
    TORCH_CHECK(I % TILE_I == 0, "Scatter requires I to be a multiple of TILE_I (16)");

    const dim3 scatter_block(256);

    // SCATTER: Y_q_ on stream 0 (output=I, split over J)
    {
        const int num_i_tiles = (I + TILE_I - 1) / TILE_I;
        dim3 grid(num_i_tiles * scat_j_chunks_q, H, B);
        Yq_scatter<D_TMPL><<<grid, scatter_block, scatter_smem_size, streams[0]>>>(
            reinterpret_cast<const bf16*>(Q.data_ptr<at::BFloat16>()),
            reinterpret_cast<const bf16*>(R.data_ptr<at::BFloat16>()),
            reinterpret_cast<const bf16*>(S.data_ptr<at::BFloat16>()),
            reinterpret_cast<const bf16*>(Vr_2.data_ptr<at::BFloat16>()),
            reinterpret_cast<const bf16*>(Vs_2.data_ptr<at::BFloat16>()),
            m_j.data_ptr<float>(), l_j.data_ptr<float>(),
            m_k.data_ptr<float>(), l_k.data_ptr<float>(),
            Yq_scat_part.data_ptr<float>(),
            B, H, I, J, K, scale,
            scat_j_chunks_q
        );
        reduce_scatter_partials<D_TMPL><<<dim3(I, H, B), D_TMPL, 0, streams[0]>>>(
            Yq_scat_part.data_ptr<float>(),
            reinterpret_cast<bf16*>(Y_q_.data_ptr<at::BFloat16>()),
            I, scat_j_chunks_q
        );
    }

    // SCATTER: Y_r_ on stream 1 (output=J, split over I)
    {
        const int num_j_tiles = (J + TILE_J - 1) / TILE_J;
        dim3 grid(num_j_tiles * scat_i_chunks_r, H, B);
        Yr_scatter<D_TMPL><<<grid, scatter_block, scatter_smem_size, streams[1]>>>(
            reinterpret_cast<const bf16*>(Q.data_ptr<at::BFloat16>()),
            reinterpret_cast<const bf16*>(R.data_ptr<at::BFloat16>()),
            reinterpret_cast<const bf16*>(S.data_ptr<at::BFloat16>()),
            reinterpret_cast<const bf16*>(Vq_2.data_ptr<at::BFloat16>()),
            reinterpret_cast<const bf16*>(Vs_2.data_ptr<at::BFloat16>()),
            m_i.data_ptr<float>(), l_i.data_ptr<float>(),
            m_k.data_ptr<float>(), l_k.data_ptr<float>(),
            Yr_scat_part.data_ptr<float>(),
            B, H, I, J, K, scale,
            scat_i_chunks_r
        );
        reduce_scatter_partials<D_TMPL><<<dim3(J, H, B), D_TMPL, 0, streams[1]>>>(
            Yr_scat_part.data_ptr<float>(),
            reinterpret_cast<bf16*>(Y_r_.data_ptr<at::BFloat16>()),
            J, scat_i_chunks_r
        );
    }

    // SCATTER: Y_s_ on stream 2 (output=K, split over I)
    {
        const int num_k_tiles = (K + TILE_K - 1) / TILE_K;
        dim3 grid(num_k_tiles * scat_i_chunks_s, H, B);
        Ys_scatter<D_TMPL><<<grid, scatter_block, scatter_smem_size, streams[2]>>>(
            reinterpret_cast<const bf16*>(Q.data_ptr<at::BFloat16>()),
            reinterpret_cast<const bf16*>(R.data_ptr<at::BFloat16>()),
            reinterpret_cast<const bf16*>(S.data_ptr<at::BFloat16>()),
            reinterpret_cast<const bf16*>(Vq_2.data_ptr<at::BFloat16>()),
            reinterpret_cast<const bf16*>(Vr_2.data_ptr<at::BFloat16>()),
            m_i.data_ptr<float>(), l_i.data_ptr<float>(),
            m_j.data_ptr<float>(), l_j.data_ptr<float>(),
            Ys_scat_part.data_ptr<float>(),
            B, H, I, J, K, scale,
            scat_i_chunks_s
        );
        reduce_scatter_partials<D_TMPL><<<dim3(K, H, B), D_TMPL, 0, streams[2]>>>(
            Ys_scat_part.data_ptr<float>(),
            reinterpret_cast<bf16*>(Y_s_.data_ptr<at::BFloat16>()),
            K, scat_i_chunks_s
        );
    }

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




