/**
 * @file backward.cu
 * @brief Backward pass CUDA kernels for hypergraph attention.
 *
 * Computes gradients for Q, R, S, and all V tensors using online softmax
 * statistics for numerical stability. Includes Jacobian correction terms
 * for proper gradient flow through the softmax.
 *
 * Copyright (c) 2026 Springtail AI. MIT License.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <tuple>

#include "common.cuh"
#include "../cpp/cuda_bindings.h"

// Backward-specific tile aliases for gradient kernels
#ifndef T_I
#define T_I TILE_I
#endif
#ifndef T_J
#define T_J TILE_J
#endif
#ifndef T_K
#define T_K TILE_K
#endif

// =============================================================================
// NOTE: Softmax stats (m_i, l_i, m_j, l_j, m_k, l_k) are computed during the
// forward pass and passed to backward. The backward pass does NOT recompute
// these stats - this avoids redundant work.
// =============================================================================

// =============================================================================
// Gradient Kernels for V Tensors (Gather Path)
// =============================================================================

template<int D_CONST>
__global__ void Vq_gather_grad(
    const float* __restrict__ Q,        // [B,H,N,D]
    const float* __restrict__ R,        // [B,H,N,D]
    const float* __restrict__ S,        // [B,H,N,D]
    const float* __restrict__ Vs,       // [B,H,N,D]   (Vs_1 slice)
    const float* __restrict__ Vr,       // [B,H,N,D]   (Vr_1 slice)
    const float* __restrict__ gradY,    // [B,H,N,D]   (grad_output slice)   <-- single source
    const float* __restrict__ m_j,      // [B,H,N]
    const float* __restrict__ l_j,      // [B,H,N]
    const float* __restrict__ m_k,      // [B,H,N]
    const float* __restrict__ l_k,      // [B,H,N]
    float*       __restrict__ gradVq,   // [B,H,N,D]   (output)
    int  N,
    float scale )
{
    /* ---- decode indices ------------------------------------------------ */
    int bh  = blockIdx.z;          // flattened (batch, head)
    int i0  = blockIdx.x * T_I + threadIdx.x;
    int k0  = blockIdx.y * T_K + threadIdx.y;
    
    // Track if this thread should compute (DON'T return early - need all threads for shared mem loading)
    const bool active = (i0 < N && k0 < N);

    const int64_t stride_BH = (int64_t)N * D_CONST;
    const float* QBH   = Q       + bh * stride_BH;
    const float* RBH   = R       + bh * stride_BH;
    const float* SBH   = S       + bh * stride_BH;
    const float* VsBH  = Vs      + bh * stride_BH;
    const float* VrBH  = Vr      + bh * stride_BH;
    const float* gYBH  = gradY   + bh * stride_BH;
    const float* mJBH  = m_j     + (int64_t)bh * N;
    const float* lJBH  = l_j     + (int64_t)bh * N;
    const float* mKBH  = m_k     + (int64_t)bh * N;
    const float* lKBH  = l_k     + (int64_t)bh * N;
          float* gVqBH = gradVq  + bh * stride_BH;

    /* ---- load Q[i,:]  S[k,:]  Vs[k,:] into registers ------------------- */
    float q_vec[D_CONST];
    float s_vec[D_CONST], vs_vec[D_CONST];
    float grad_acc[D_CONST] = {0.0f};
    
    // Clamp indices for safe memory access (inactive threads load from valid location)
    const int i0_safe = min(i0, N-1);
    const int k0_safe = min(k0, N-1);
    
    #pragma unroll
    for (int d=0; d<D_CONST; ++d){
        q_vec[d]  = QBH[i0_safe*D_CONST + d];
        s_vec[d]  = SBH[k0_safe*D_CONST + d];
        vs_vec[d] = VsBH[k0_safe*D_CONST + d];
    }


    /* ---- loop over J in chunks ---------------------------------------- */
    for (int jBase=0; jBase<N; jBase+=T_J){
        __shared__ float sh_R [T_J][D_CONST];
        __shared__ float sh_Vr[T_J][D_CONST];
        __shared__ float sh_gY[T_J][D_CONST];
        __shared__ float sh_mj[T_J];
        __shared__ float sh_lj[T_J];

        // Cooperative loading: ALL threads participate to cover all D dimensions
        int lj = threadIdx.y;                       // 0 … T_K-1 (≤T_J)
        if (lj < T_J && (jBase+lj) < N){
            int jGlob = jBase + lj;
            #pragma unroll
            for (int d=threadIdx.x; d<D_CONST; d+=T_I){
                sh_R [lj][d] = RBH [jGlob*D_CONST + d];
                sh_Vr[lj][d] = VrBH[jGlob*D_CONST + d];
                sh_gY[lj][d] = gYBH[jGlob*D_CONST + d];   // grad_Yr = grad_output
            }
            if (threadIdx.x == 0){
                sh_mj[lj] = mJBH[jGlob];
                sh_lj[lj] = lJBH[jGlob];
            }
        }
        __syncthreads();

        /* ---- iterate inside loaded j-chunk (only active threads compute) */
        if (active) {
            for (int jOff=0; jOff<T_J && (jBase+jOff)<N; ++jOff){
                float logits=0.f;
                #pragma unroll
                for (int d=0; d<D_CONST; ++d)
                    logits += q_vec[d]*sh_R[jOff][d]*s_vec[d];
                logits *= scale;

                float wj = __expf(fminf(logits - sh_mj[jOff], EXP_CLIP)) / fmaxf(sh_lj[jOff], DENOM_EPS);
                float wk = __expf(fminf(logits - mKBH[k0], EXP_CLIP))    / fmaxf(lKBH[k0], DENOM_EPS);

                #pragma unroll
                for (int d=0; d<D_CONST; ++d){
                    grad_acc[d] += wj * sh_gY[jOff][d] * vs_vec[d]        /* Yr path */
                                  + wk * gYBH[k0*D_CONST + d]  * sh_Vr[jOff][d];/* Ys path */
                }
            }
        }
        __syncthreads();
    }

    /* ---- atomic add to global grad (only active threads) --------------- */
    if (active) {
        #pragma unroll
        for (int d=0; d<D_CONST; ++d)
            atomicAdd(&gVqBH[i0*D_CONST + d], grad_acc[d]);
    }
}

template<int D_CONST>
__global__ void Vr_gather_grad(
    const float* __restrict__ Q,        // [B,H,N,D]
    const float* __restrict__ R,        // [B,H,N,D]
    const float* __restrict__ S,        // [B,H,N,D]
    const float* __restrict__ Vq,       // [B,H,N,D]
    const float* __restrict__ Vs,       // [B,H,N,D]
    const float* __restrict__ gradY,    // [B,H,N,D]
    const float* __restrict__ m_i,      // [B,H,N]
    const float* __restrict__ l_i,      // [B,H,N]
    const float* __restrict__ m_k,      // [B,H,N]
    const float* __restrict__ l_k,      // [B,H,N]
    float*       __restrict__ gradVr,   // [B,H,N,D]
    int  N,
    float scale )
{
    int bh  = blockIdx.z;
    int j0  = blockIdx.x * T_I + threadIdx.x;
    int k0  = blockIdx.y * T_K + threadIdx.y;
    
    // Track if this thread should compute (DON'T return early - need all threads for shared mem loading)
    const bool active = (j0 < N && k0 < N);

    const int64_t stride_BH = (int64_t)N * D_CONST;
    const float* QBH   = Q      + bh * stride_BH;
    const float* RBH   = R      + bh * stride_BH;
    const float* SBH   = S      + bh * stride_BH;
    const float* VqBH  = Vq     + bh * stride_BH;
    const float* VsBH  = Vs     + bh * stride_BH;
    const float* gYBH  = gradY  + bh * stride_BH;
    const float* mIBH  = m_i    + (int64_t)bh * N;
    const float* lIBH  = l_i    + (int64_t)bh * N;
    const float* mKBH  = m_k    + (int64_t)bh * N;
    const float* lKBH  = l_k    + (int64_t)bh * N;
          float* gVrBH = gradVr + bh * stride_BH;

    // Clamp indices for safe memory access (inactive threads load from valid location)
    const int j0_safe = min(j0, N-1);
    const int k0_safe = min(k0, N-1);

    float r_vec[D_CONST];
    float s_vec[D_CONST];
    float vs_vec[D_CONST];
    float gy_k_vec[D_CONST];

    #pragma unroll
    for (int d=0; d<D_CONST; ++d){
        r_vec[d]    = RBH[j0_safe*D_CONST + d];
        s_vec[d]    = SBH[k0_safe*D_CONST + d];
        vs_vec[d]   = VsBH[k0_safe*D_CONST + d];
        gy_k_vec[d] = gYBH[k0_safe*D_CONST + d];
    }

    float grad_acc[D_CONST] = {0.0f};
    float m_k_val = mKBH[k0_safe];
    float l_k_val = lKBH[k0_safe];


    for (int iBase=0; iBase<N; iBase+=T_J){
        __shared__ float sh_Q [T_J][D_CONST];
        __shared__ float sh_Vq[T_J][D_CONST];
        __shared__ float sh_gY[T_J][D_CONST];
        __shared__ float sh_mi[T_J];
        __shared__ float sh_li[T_J];

        // Cooperative loading: ALL threads participate to cover all D dimensions
        int li = threadIdx.y;
        if (li < T_J && (iBase+li) < N){
            int iGlob = iBase + li;
            #pragma unroll
            for (int d=threadIdx.x; d<D_CONST; d+=T_I){
                sh_Q [li][d] = QBH [iGlob*D_CONST + d];
                sh_Vq[li][d] = VqBH[iGlob*D_CONST + d];
                sh_gY[li][d] = gYBH[iGlob*D_CONST + d];
            }
            if (threadIdx.x == 0){
                sh_mi[li] = mIBH[iGlob];
                sh_li[li] = lIBH[iGlob];
            }
        }
        __syncthreads();

        // Only active threads compute
        if (active) {
            for (int iOff=0; iOff<T_J && (iBase+iOff)<N; ++iOff){
                float logits=0.f;
                #pragma unroll
                for (int d=0; d<D_CONST; ++d)
                    logits += sh_Q[iOff][d] * r_vec[d] * s_vec[d];
                logits *= scale;

                float wi = __expf(fminf(logits - sh_mi[iOff], EXP_CLIP)) / fmaxf(sh_li[iOff], DENOM_EPS);
                float wk = __expf(fminf(logits - m_k_val, EXP_CLIP))     / fmaxf(l_k_val, DENOM_EPS);

                #pragma unroll
                for (int d=0; d<D_CONST; ++d){
                    grad_acc[d] += wi * sh_gY[iOff][d] * vs_vec[d]
                                  + wk * gy_k_vec[d]   * sh_Vq[iOff][d];
                }
            }
        }
        __syncthreads();
    }

    /* ---- atomic add to global grad (only active threads) --------------- */
    if (active) {
        #pragma unroll
        for (int d=0; d<D_CONST; ++d)
            atomicAdd(&gVrBH[j0*D_CONST + d], grad_acc[d]);
    }
}

template<int D_CONST>
__global__ void Vs_gather_grad(
    const float* __restrict__ Q,        // [B,H,N,D]
    const float* __restrict__ R,        // [B,H,N,D]
    const float* __restrict__ S,        // [B,H,N,D]
    const float* __restrict__ Vq,       // [B,H,N,D]
    const float* __restrict__ Vr,       // [B,H,N,D]
    const float* __restrict__ gradY,    // [B,H,N,D]
    const float* __restrict__ m_i,      // [B,H,N]
    const float* __restrict__ l_i,      // [B,H,N]
    const float* __restrict__ m_j,      // [B,H,N]
    const float* __restrict__ l_j,      // [B,H,N]
    float*       __restrict__ gradVs,   // [B,H,N,D]
    int  N,
    float scale )
{
    int bh  = blockIdx.z;
    int i0  = blockIdx.x * T_I + threadIdx.x;
    int k0  = blockIdx.y * T_K + threadIdx.y;
    
    // Track if this thread should compute (DON'T return early - need all threads for shared mem loading)
    const bool active = (i0 < N && k0 < N);

    const int64_t stride_BH = (int64_t)N * D_CONST;
    const float* QBH   = Q      + bh * stride_BH;
    const float* RBH   = R      + bh * stride_BH;
    const float* SBH   = S      + bh * stride_BH;
    const float* VqBH  = Vq     + bh * stride_BH;
    const float* VrBH  = Vr     + bh * stride_BH;
    const float* gYBH  = gradY  + bh * stride_BH;
    const float* mIBH  = m_i    + (int64_t)bh * N;
    const float* lIBH  = l_i    + (int64_t)bh * N;
    const float* mJBH  = m_j    + (int64_t)bh * N;
    const float* lJBH  = l_j    + (int64_t)bh * N;
          float* gVsBH = gradVs + bh * stride_BH;

    // Clamp indices for safe memory access (inactive threads load from valid location)
    const int i0_safe = min(i0, N-1);
    const int k0_safe = min(k0, N-1);

    float q_vec[D_CONST];
    float s_vec[D_CONST];
    float vq_vec[D_CONST];
    float gy_i_vec[D_CONST];

    #pragma unroll
    for (int d=0; d<D_CONST; ++d){
        q_vec[d]    = QBH[i0_safe*D_CONST + d];
        s_vec[d]    = SBH[k0_safe*D_CONST + d];
        vq_vec[d]   = VqBH[i0_safe*D_CONST + d];
        gy_i_vec[d] = gYBH[i0_safe*D_CONST + d];
    }

    float grad_acc[D_CONST] = {0.0f};
    float m_i_val = mIBH[i0_safe];
    float l_i_val = lIBH[i0_safe];


    for (int jBase=0; jBase<N; jBase+=T_J){
        __shared__ float sh_R [T_J][D_CONST];
        __shared__ float sh_Vr[T_J][D_CONST];
        __shared__ float sh_gY[T_J][D_CONST];
        __shared__ float sh_mj[T_J];
        __shared__ float sh_lj[T_J];

        // Cooperative loading: ALL threads participate to cover all D dimensions
        int lj = threadIdx.y;
        if (lj < T_J && (jBase+lj) < N){
            int jGlob = jBase + lj;
            #pragma unroll
            for (int d=threadIdx.x; d<D_CONST; d+=T_I){
                sh_R [lj][d] = RBH [jGlob*D_CONST + d];
                sh_Vr[lj][d] = VrBH[jGlob*D_CONST + d];
                sh_gY[lj][d] = gYBH[jGlob*D_CONST + d];
            }
            if (threadIdx.x == 0){
                sh_mj[lj] = mJBH[jGlob];
                sh_lj[lj] = lJBH[jGlob];
            }
        }
        __syncthreads();

        // Only active threads compute
        if (active) {
            for (int jOff=0; jOff<T_J && (jBase+jOff)<N; ++jOff){
                float logits=0.f;
                #pragma unroll
                for (int d=0; d<D_CONST; ++d)
                    logits += q_vec[d] * sh_R[jOff][d] * s_vec[d];
                logits *= scale;

                float wi = __expf(fminf(logits - m_i_val, EXP_CLIP)) / fmaxf(l_i_val, DENOM_EPS);
                float wj = __expf(fminf(logits - sh_mj[jOff], EXP_CLIP)) / fmaxf(sh_lj[jOff], DENOM_EPS);

                #pragma unroll
                for (int d=0; d<D_CONST; ++d){
                    grad_acc[d] += wi * gy_i_vec[d]   * sh_Vr[jOff][d]
                                  + wj * sh_gY[jOff][d] * vq_vec[d];
                }
            }
        }
        __syncthreads();
    }

    /* ---- atomic add to global grad (only active threads) --------------- */
    if (active) {
        #pragma unroll
        for (int d=0; d<D_CONST; ++d)
            atomicAdd(&gVsBH[k0*D_CONST + d], grad_acc[d]);
    }
}

// ===================== scatter-grad Vq2, Vr2, Vs2 ======================


template<int D_CONST>
__global__ void Vq_scatter_grad(
    const float* __restrict__ Q,      // [B,H,N,D]
    const float* __restrict__ R,      // [B,H,N,D]
    const float* __restrict__ S,      // [B,H,N,D]
    const float* __restrict__ Vr2,    // [B,H,N,D]
    const float* __restrict__ Vs2,    // [B,H,N,D]
    const float* __restrict__ gradY,  // [B,H,N,D]  (same tensor provides dYr & dYs)
    const float* __restrict__ m_i,    // [B,H,N]
    const float* __restrict__ l_i,    // [B,H,N]
    const float* __restrict__ m_j,    // [B,H,N]
    const float* __restrict__ l_j,    // [B,H,N]
    const float* __restrict__ m_k,    // [B,H,N]
    const float* __restrict__ l_k,    // [B,H,N]
    float*       __restrict__ gradVq, // [B,H,N,D]
    int N, float scale)
{
    /* threadblock organisation = (i,k) tile  --------------------------- */
    const int i0 = blockIdx.x * T_I + threadIdx.x;   // 0..N-1 (column)
    const int k0 = blockIdx.y * T_K + threadIdx.y;   // 0..N-1 (row)
    const int bh = blockIdx.z;                       // flattened (B,H)
    
    // ALL threads must participate in cooperative loading - use active flag instead of early return
    const bool active = (i0 < N && k0 < N);
    // Clamped indices for safe memory access during cooperative loading
    const int i0_safe = min(i0, N - 1);
    const int k0_safe = min(k0, N - 1);

    // per-BH base pointers and strides
    const int64_t stride_BH = (int64_t)N * D_CONST;
    const float* QBH   = Q    + (int64_t)bh * stride_BH;
    const float* RBH   = R    + (int64_t)bh * stride_BH;
    const float* SBH   = S    + (int64_t)bh * stride_BH;
    const float* Vr2BH = Vr2  + (int64_t)bh * stride_BH;
    const float* Vs2BH = Vs2  + (int64_t)bh * stride_BH;
    const float* gYBH  = gradY+ (int64_t)bh * stride_BH;
    const float* m_iBH = m_i  + (int64_t)bh * N;
    const float* l_iBH = l_i  + (int64_t)bh * N;
    const float* m_jBH = m_j  + (int64_t)bh * N;
    const float* l_jBH = l_j  + (int64_t)bh * N;
    const float* m_kBH = m_k  + (int64_t)bh * N;
    const float* l_kBH = l_k  + (int64_t)bh * N;
          float* gVqBH = gradVq + (int64_t)bh * stride_BH;

    // ---- registers for Q[i0], S[k0], Vs2[k0] --------------------------
    float q_vec[D_CONST];
    float s_vec[D_CONST], vs2_vec[D_CONST];
    #pragma unroll
    for (int d=0; d<D_CONST; ++d){
        q_vec[d]  = QBH[i0_safe*D_CONST + d];
        s_vec[d]  = SBH[k0_safe*D_CONST + d];
        vs2_vec[d]= Vs2BH[k0_safe*D_CONST + d];
    }
    float grad_acc[D_CONST] = {0.0f};

    // ---- shared memory tiles for (j) ----------------------------------
    extern __shared__ float shmem[];
    float* sh_R   = shmem;                        // T_J * D_CONST
    float* sh_Vr2 = sh_R   + T_J * D_CONST;       // T_J * D_CONST
    float* sh_gYr = sh_Vr2 + T_J * D_CONST;       // T_J * D_CONST   (dYr[j,:])
    float* sh_mj  = (float*)(sh_gYr + T_J * D_CONST);  // T_J scalars
    float* sh_lj  = sh_mj  + T_J;

    /* iterate over J tiles --------------------------------------------- */
    for (int jBase=0; jBase < N; jBase+=T_J){
        // cooperative load by all (i,k) threads inside TB
        const int ld_idx = threadIdx.y;  // reuse y-dimension for co-load rows
        if (ld_idx < T_J && (jBase+ld_idx) < N){
            const int jGlob = jBase + ld_idx;
            for (int d=threadIdx.x; d<D_CONST; d+=T_I){
                sh_R  [ld_idx*D_CONST + d] = RBH  [jGlob*D_CONST + d];
                sh_Vr2[ld_idx*D_CONST + d] = Vr2BH[jGlob*D_CONST + d];
                sh_gYr[ld_idx*D_CONST + d] = gYBH [jGlob*D_CONST + d];
            }
            if (threadIdx.x==0){
                sh_mj[ld_idx] = m_jBH[jGlob];
                sh_lj[ld_idx] = l_jBH[jGlob];
            }
        }
        __syncthreads();

        // ---- loop inside J-tile (only active threads compute) --------
        if (active) {
            for (int jOff=0; jOff<T_J && (jBase+jOff)<N; ++jOff){
                // dot(Q[i],R[j],S[k])
                float dot = 0.f;
                #pragma unroll
                for (int d=0; d<D_CONST; ++d)
                    dot += q_vec[d] * sh_R[jOff*D_CONST + d] * s_vec[d];
                float logits = dot * scale;

                // Compute combined attention weights in LOG-SPACE to avoid overflow
                // Aq = exp(logits - m_i) / l_i,  As = exp(logits - m_k) / l_k
                // Aq * As = exp(2*logits - m_i - m_k) / (l_i * l_k)
                float log_Aq = logits - m_iBH[i0_safe];
                float log_Ar = logits - sh_mj[jOff];
                float log_As = logits - m_kBH[k0_safe];

                float l_i_val = fmaxf(l_iBH[i0_safe], DENOM_EPS);
                float l_j_val = fmaxf(sh_lj[jOff], DENOM_EPS);
                float l_k_val = fmaxf(l_kBH[k0_safe], DENOM_EPS);

                // term 1: Aq*As = exp(log_Aq + log_As) / (l_i * l_k)
                float w1 = __expf(fminf(log_Aq + log_As, EXP_CLIP)) / (l_i_val * l_k_val);
                // term 2: Aq*Ar = exp(log_Aq + log_Ar) / (l_i * l_j)
                float w2 = __expf(fminf(log_Aq + log_Ar, EXP_CLIP)) / (l_i_val * l_j_val);

                // load vectors
                const float* dYr_vec = &sh_gYr[jOff*D_CONST];
                const float* Vr2_vec = &sh_Vr2[jOff*D_CONST];
                const float* dYs_vec = &gYBH[k0*D_CONST]; // contiguous in global, fine

                #pragma unroll
                for (int d=0; d<D_CONST; ++d){
                    grad_acc[d] += w1 * dYr_vec[d] * vs2_vec[d] +
                                   w2 * dYs_vec[d] * Vr2_vec[d];
                }
            }
        }
        __syncthreads();
    }

    // ---- atomic add results (only active threads) --------------------
    if (active) {
        #pragma unroll
        for (int d=0; d<D_CONST; ++d)
            atomicAdd(&gVqBH[i0*D_CONST + d], grad_acc[d]);
    }
}

template<int D_CONST>
__global__ void Vr_scatter_grad(
    const float* __restrict__ Q,      // [B,H,N,D]
    const float* __restrict__ R,      // [B,H,N,D]
    const float* __restrict__ S,      // [B,H,N,D]
    const float* __restrict__ Vq2,    // [B,H,N,D]
    const float* __restrict__ Vs2,    // [B,H,N,D]
    const float* __restrict__ gradY,  // [B,H,N,D]
    const float* __restrict__ m_i,    // [B,H,N]
    const float* __restrict__ l_i,    // [B,H,N]
    const float* __restrict__ m_j,    // [B,H,N]
    const float* __restrict__ l_j,    // [B,H,N]
    const float* __restrict__ m_k,    // [B,H,N]
    const float* __restrict__ l_k,    // [B,H,N]
    float*       __restrict__ gradVr, // [B,H,N,D]
    int N, float scale)
{
    const int j0 = blockIdx.x * T_I + threadIdx.x;
    const int k0 = blockIdx.y * T_K + threadIdx.y;
    const int bh = blockIdx.z;
    
    // ALL threads must participate in cooperative loading - use active flag instead of early return
    const bool active = (j0 < N && k0 < N);
    // Clamped indices for safe memory access during cooperative loading
    const int j0_safe = min(j0, N - 1);
    const int k0_safe = min(k0, N - 1);

    const int64_t stride_BH = (int64_t)N * D_CONST;
    const float* QBH   = Q     + (int64_t)bh * stride_BH;
    const float* RBH   = R     + (int64_t)bh * stride_BH;
    const float* SBH   = S     + (int64_t)bh * stride_BH;
    const float* Vq2BH = Vq2   + (int64_t)bh * stride_BH;
    const float* Vs2BH = Vs2   + (int64_t)bh * stride_BH;
    const float* gYBH  = gradY + (int64_t)bh * stride_BH;
    const float* m_iBH = m_i   + (int64_t)bh * N;
    const float* l_iBH = l_i   + (int64_t)bh * N;
    const float* m_jBH = m_j   + (int64_t)bh * N;
    const float* l_jBH = l_j   + (int64_t)bh * N;
    const float* m_kBH = m_k   + (int64_t)bh * N;
    const float* l_kBH = l_k   + (int64_t)bh * N;
          float* gVrBH = gradVr+ (int64_t)bh * stride_BH;

    float r_vec[D_CONST];
    float s_vec[D_CONST];
    float vs2_vec[D_CONST];
    float gy_k_vec[D_CONST];
    #pragma unroll
    for (int d=0; d<D_CONST; ++d){
        r_vec[d]    = RBH[j0_safe*D_CONST + d];
        s_vec[d]    = SBH[k0_safe*D_CONST + d];
        vs2_vec[d]  = Vs2BH[k0_safe*D_CONST + d];
        gy_k_vec[d] = gYBH[k0_safe*D_CONST + d];
    }

    float grad_acc[D_CONST] = {0.0f};

    extern __shared__ float shmem[];
    float* sh_Q   = shmem;                         // T_J * D_CONST
    float* sh_Vq2 = sh_Q   + T_J * D_CONST;        // T_J * D_CONST
    float* sh_gYq = sh_Vq2 + T_J * D_CONST;        // T_J * D_CONST
    float* sh_mi  = (float*)(sh_gYq + T_J * D_CONST); // T_J
    float* sh_li  = sh_mi + T_J;                   // T_J

    for (int iBase=0; iBase < N; iBase+=T_J){
        const int li = threadIdx.y;
        if (li < T_J && (iBase + li) < N){
            const int iGlob = iBase + li;
            for (int d=threadIdx.x; d<D_CONST; d+=T_I){
                sh_Q  [li*D_CONST + d] = QBH [iGlob*D_CONST + d];
                sh_Vq2[li*D_CONST + d] = Vq2BH[iGlob*D_CONST + d];
                sh_gYq[li*D_CONST + d] = gYBH [iGlob*D_CONST + d];
            }
            if (threadIdx.x == 0){
                sh_mi[li] = m_iBH[iGlob];
                sh_li[li] = l_iBH[iGlob];
            }
        }
        __syncthreads();

        // Only active threads compute
        if (active) {
            for (int iOff=0; iOff<T_J && (iBase+iOff)<N; ++iOff){
                float dot = 0.f;
                #pragma unroll
                for (int d=0; d<D_CONST; ++d)
                    dot += sh_Q[iOff*D_CONST + d] * r_vec[d] * s_vec[d];
                float logits = dot * scale;

                // Compute combined attention weights in LOG-SPACE to avoid overflow
                float log_Aq = logits - sh_mi[iOff];
                float log_Ar = logits - m_jBH[j0_safe];
                float log_As = logits - m_kBH[k0_safe];

                float l_i_val = fmaxf(sh_li[iOff], DENOM_EPS);
                float l_j_val = fmaxf(l_jBH[j0_safe], DENOM_EPS);
                float l_k_val = fmaxf(l_kBH[k0_safe], DENOM_EPS);

                // term 1: Ar*As = exp(log_Ar + log_As) / (l_j * l_k)
                float w1 = __expf(fminf(log_Ar + log_As, EXP_CLIP)) / (l_j_val * l_k_val);
                // term 2: Aq*Ar = exp(log_Aq + log_Ar) / (l_i * l_j)
                float w2 = __expf(fminf(log_Aq + log_Ar, EXP_CLIP)) / (l_i_val * l_j_val);

                const float* dy_q_vec = &sh_gYq[iOff*D_CONST];
                const float* vq2_vec  = &sh_Vq2[iOff*D_CONST];

                #pragma unroll
                for (int d=0; d<D_CONST; ++d){
                    grad_acc[d] += w1 * dy_q_vec[d] * vs2_vec[d]
                                  + w2 * gy_k_vec[d] * vq2_vec[d];
                }
            }
        }
        __syncthreads();
    }

    // Only active threads write results
    if (active) {
        #pragma unroll
        for (int d=0; d<D_CONST; ++d)
            atomicAdd(&gVrBH[j0*D_CONST + d], grad_acc[d]);
    }
}

template<int D_CONST>
__global__ void Vs_scatter_grad(
    const float* __restrict__ Q,      // [B,H,N,D]
    const float* __restrict__ R,      // [B,H,N,D]
    const float* __restrict__ S,      // [B,H,N,D]
    const float* __restrict__ Vq2,    // [B,H,N,D]
    const float* __restrict__ Vr2,    // [B,H,N,D]
    const float* __restrict__ gradY,  // [B,H,N,D]
    const float* __restrict__ m_i,    // [B,H,N]
    const float* __restrict__ l_i,    // [B,H,N]
    const float* __restrict__ m_j,    // [B,H,N]
    const float* __restrict__ l_j,    // [B,H,N]
    const float* __restrict__ m_k,    // [B,H,N]
    const float* __restrict__ l_k,    // [B,H,N]
    float*       __restrict__ gradVs, // [B,H,N,D]
    int N, float scale)
{
    const int i0 = blockIdx.x * T_I + threadIdx.x;
    const int k0 = blockIdx.y * T_K + threadIdx.y;
    const int bh = blockIdx.z;
    
    // ALL threads must participate in cooperative loading - use active flag instead of early return
    const bool active = (i0 < N && k0 < N);
    // Clamped indices for safe memory access during cooperative loading
    const int i0_safe = min(i0, N - 1);
    const int k0_safe = min(k0, N - 1);

    const int64_t stride_BH = (int64_t)N * D_CONST;
    const float* QBH   = Q     + (int64_t)bh * stride_BH;
    const float* RBH   = R     + (int64_t)bh * stride_BH;
    const float* SBH   = S     + (int64_t)bh * stride_BH;
    const float* Vq2BH = Vq2   + (int64_t)bh * stride_BH;
    const float* Vr2BH = Vr2   + (int64_t)bh * stride_BH;
    const float* gYBH  = gradY + (int64_t)bh * stride_BH;
    const float* m_iBH = m_i   + (int64_t)bh * N;
    const float* l_iBH = l_i   + (int64_t)bh * N;
    const float* m_jBH = m_j   + (int64_t)bh * N;
    const float* l_jBH = l_j   + (int64_t)bh * N;
    const float* m_kBH = m_k   + (int64_t)bh * N;
    const float* l_kBH = l_k   + (int64_t)bh * N;
          float* gVsBH = gradVs+ (int64_t)bh * stride_BH;

    float q_vec[D_CONST];
    float s_vec[D_CONST];
    float vq2_vec[D_CONST];
    float gy_i_vec[D_CONST];
    #pragma unroll
    for (int d=0; d<D_CONST; ++d){
        q_vec[d]    = QBH[i0_safe*D_CONST + d];
        s_vec[d]    = SBH[k0_safe*D_CONST + d];
        vq2_vec[d]  = Vq2BH[i0_safe*D_CONST + d];
        gy_i_vec[d] = gYBH [i0_safe*D_CONST + d];
    }

    float grad_acc[D_CONST] = {0.0f};

    extern __shared__ float shmem[];
    float* sh_R   = shmem;                         // T_J * D_CONST
    float* sh_Vr2 = sh_R   + T_J * D_CONST;        // T_J * D_CONST
    float* sh_gYr = sh_Vr2 + T_J * D_CONST;        // T_J * D_CONST
    float* sh_mj  = (float*)(sh_gYr + T_J * D_CONST);
    float* sh_lj  = sh_mj + T_J;

    for (int jBase=0; jBase < N; jBase+=T_J){
        int lj = threadIdx.y;
        if (lj < T_J && (jBase + lj) < N){
            int jGlob = jBase + lj;
            #pragma unroll
            for (int d=threadIdx.x; d<D_CONST; d+=T_I){
                sh_R  [lj*D_CONST + d] = RBH [jGlob*D_CONST + d];
                sh_Vr2[lj*D_CONST + d] = Vr2BH[jGlob*D_CONST + d];
                sh_gYr[lj*D_CONST + d] = gYBH[jGlob*D_CONST + d];
            }
            if (threadIdx.x == 0){
                sh_mj[lj] = m_jBH[jGlob];
                sh_lj[lj] = l_jBH[jGlob];
            }
        }
        __syncthreads();

        // Only active threads compute
        if (active) {
            for (int jOff=0; jOff<T_J && (jBase+jOff)<N; ++jOff){
                float dot = 0.f;
                #pragma unroll
                for (int d=0; d<D_CONST; ++d)
                    dot += q_vec[d] * sh_R[jOff*D_CONST + d] * s_vec[d];
                float logits = dot * scale;

                // Compute combined attention weights in LOG-SPACE to avoid overflow
                float log_Aq = logits - m_iBH[i0_safe];
                float log_Ar = logits - sh_mj[jOff];
                float log_As = logits - m_kBH[k0_safe];

                float l_i_val = fmaxf(l_iBH[i0_safe], DENOM_EPS);
                float l_j_val = fmaxf(sh_lj[jOff], DENOM_EPS);
                float l_k_val = fmaxf(l_kBH[k0_safe], DENOM_EPS);

                // term 1: Ar*As = exp(log_Ar + log_As) / (l_j * l_k)
                float w1 = __expf(fminf(log_Ar + log_As, EXP_CLIP)) / (l_j_val * l_k_val);
                // term 2: Aq*As = exp(log_Aq + log_As) / (l_i * l_k)
                float w2 = __expf(fminf(log_Aq + log_As, EXP_CLIP)) / (l_i_val * l_k_val);

                const float* vr2_vec = &sh_Vr2[jOff*D_CONST];
                const float* dy_r_vec = &sh_gYr[jOff*D_CONST];

                #pragma unroll
                for (int d=0; d<D_CONST; ++d){
                    grad_acc[d] += w1 * gy_i_vec[d] * vr2_vec[d]
                                  + w2 * dy_r_vec[d] * vq2_vec[d];
                }
            }
        }
        __syncthreads();
    }

    // Only active threads write results
    if (active) {
        #pragma unroll
        for (int d=0; d<D_CONST; ++d)
            atomicAdd(&gVsBH[k0*D_CONST + d], grad_acc[d]);
    }
}

// =============================================================================
// NOTE: The old 3D jacobian_corrections kernel has been replaced by two-pass
// 2D-tiled correction passes using QS_grad_kernel<true> and R_grad_kernel<true>.
// This eliminates the 3D thread grid (8192 blocks) in favor of 2D grids
// (128 blocks each), reducing atomic contention by 32x.
// =============================================================================



// =============================================================================
// Gradient Kernels for Q, R, S (with integrated Jacobian corrections)
// =============================================================================
//
// Architecture: Two-pass 2D-tiled approach
//   Phase 1: QS_grad_kernel<true>  → computes sum_q, sum_s (correction sums)
//            R_grad_kernel<true>   → computes sum_r (correction sum)
//   Phase 2: QS_grad_kernel<false> → computes gradQ, gradS using corrections
//            R_grad_kernel<false>  → computes gradR using corrections
//
// This replaces the old 3D jacobian_corrections kernel (8192 blocks, 512
// threads, 256 atomics/element) with 2D-tiled correction passes (128 blocks,
// 256 threads, 8 atomics/element) — a 32x reduction in atomic contention.
//
// Both kernels use a compile-time template bool CORRECTION_ONLY to share
// 95% of the code between correction and gradient modes.
//
// =============================================================================

/**
 * QS_grad_kernel - Computes gradQ and gradS with Jacobian corrections.
 * 
 * CORRECTION_ONLY=true:  Computes correction sums (sum_q, sum_r, sum_s)
 * CORRECTION_ONLY=false: Computes gradQ and gradS using precomputed corrections
 */
template<bool CORRECTION_ONLY, int BLOCK_I, int BLOCK_J, int BLOCK_K, int D_CONST, int REG_CAP = D_CONST>
__global__ void __launch_bounds__(256, 1) QS_grad_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ R,
    const float* __restrict__ S,
    const float* __restrict__ Vq1, const float* __restrict__ Vq2,
    const float* __restrict__ Vr1, const float* __restrict__ Vr2,
    const float* __restrict__ Vs1, const float* __restrict__ Vs2,
    const float* __restrict__ gradY,
    const float* __restrict__ m_i, const float* __restrict__ l_i,
    const float* __restrict__ m_j, const float* __restrict__ l_j,
    const float* __restrict__ m_k, const float* __restrict__ l_k,
    float* __restrict__ sum_q,
    float* __restrict__ sum_r,
    float* __restrict__ sum_s,
    float* __restrict__ gradQ,
    float* __restrict__ gradS,
    int  N, float scale)
{
    const int i0 = blockIdx.x * BLOCK_I + threadIdx.x;
    const int k0 = blockIdx.y * BLOCK_K + threadIdx.y;
    const int bh = blockIdx.z;
    const bool valid = (i0 < N && k0 < N);

    // Per (B,H) base pointers
    const int64_t stride_BH = (int64_t)N * D_CONST;
    const float* Qbh   = Q   + bh * stride_BH;
    const float* Rbh   = R   + bh * stride_BH;
    const float* Sbh   = S   + bh * stride_BH;
    const float* Vq1bh = Vq1 + bh * stride_BH;
    const float* Vq2bh = Vq2 + bh * stride_BH;
    const float* Vr1bh = Vr1 + bh * stride_BH;
    const float* Vr2bh = Vr2 + bh * stride_BH;
    const float* Vs1bh = Vs1 + bh * stride_BH;
    const float* Vs2bh = Vs2 + bh * stride_BH;
    const float* gYbh  = gradY + bh * stride_BH;
    const float* miBH  = m_i + bh * N;
    const float* liBH  = l_i + bh * N;
    const float* mjBH  = m_j + bh * N;
    const float* ljBH  = l_j + bh * N;
    const float* mkBH  = m_k + bh * N;
    const float* lkBH  = l_k + bh * N;
    float* sum_qBH = sum_q + bh * N;
    float* sum_sBH = sum_s + bh * N;

    constexpr int D_PAD = D_CONST + 1;  // bank-conflict-free stride
    extern __shared__ float shmem[];
    float* sh_Qi   = shmem;
    float* sh_Vq1i = sh_Qi   + BLOCK_I * D_PAD;
    float* sh_Vq2i = sh_Vq1i + BLOCK_I * D_PAD;
    float* sh_dYi  = sh_Vq2i + BLOCK_I * D_PAD;
    float* sh_Sk   = sh_dYi  + BLOCK_I * D_PAD;
    float* sh_Vs1k = sh_Sk   + BLOCK_K * D_PAD;
    float* sh_Vs2k = sh_Vs1k + BLOCK_K * D_PAD;
    float* sh_dYk  = sh_Vs2k + BLOCK_K * D_PAD;
    float* sh_R    = sh_dYk  + BLOCK_K * D_PAD;
    float* sh_Vr1  = sh_R    + BLOCK_J * D_CONST;
    float* sh_Vr2  = sh_Vr1  + BLOCK_J * D_CONST;
    float* sh_gYj  = sh_Vr2  + BLOCK_J * D_CONST;
    float* sh_mj   = sh_gYj  + BLOCK_J * D_CONST;
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
                sh_Qi  [ii * D_PAD + dd] = Qbh  [iGlob * D_CONST + dd];
                sh_Vq1i[ii * D_PAD + dd] = Vq1bh[iGlob * D_CONST + dd];
                sh_Vq2i[ii * D_PAD + dd] = Vq2bh[iGlob * D_CONST + dd];
                sh_dYi [ii * D_PAD + dd] = gYbh [iGlob * D_CONST + dd];
            } else {
                sh_Qi  [ii * D_PAD + dd] = 0.0f;
                sh_Vq1i[ii * D_PAD + dd] = 0.0f;
                sh_Vq2i[ii * D_PAD + dd] = 0.0f;
                sh_dYi [ii * D_PAD + dd] = 0.0f;
            }
        }
        for (int idx = tid; idx < BLOCK_K * D_CONST; idx += nThreads) {
            const int kk = idx / D_CONST;
            const int dd = idx % D_CONST;
            const int kGlob = blockIdx.y * BLOCK_K + kk;
            if (kGlob < N) {
                sh_Sk  [kk * D_PAD + dd] = Sbh  [kGlob * D_CONST + dd];
                sh_Vs1k[kk * D_PAD + dd] = Vs1bh[kGlob * D_CONST + dd];
                sh_Vs2k[kk * D_PAD + dd] = Vs2bh[kGlob * D_CONST + dd];
                sh_dYk [kk * D_PAD + dd] = gYbh [kGlob * D_CONST + dd];
            } else {
                sh_Sk  [kk * D_PAD + dd] = 0.0f;
                sh_Vs1k[kk * D_PAD + dd] = 0.0f;
                sh_Vs2k[kk * D_PAD + dd] = 0.0f;
                sh_dYk [kk * D_PAD + dd] = 0.0f;
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
    // Algebraic factoring: accumulate rj_weighted[d] = Σⱼ grad_A_j * R[j,d]
    // Then gradQ[d] = rj_weighted[d] * S[k,d], gradS[d] = rj_weighted[d] * Q[i,d]
    // This replaces two D-sized accumulators with one, saving 64 registers and
    // reducing the hot inner loop from 3 shmem loads/d to 1 shmem load/d.
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

        for (int ld = threadIdx.y; ld < BLOCK_J && (jBase + ld) < N; ld += BLOCK_K) {
            const int jGlob = jBase + ld;
            for (int d = threadIdx.x; d < D_CONST; d += BLOCK_I) {
                sh_R[ld*D_CONST + d]   = Rbh[jGlob*D_CONST + d];
                sh_Vr1[ld*D_CONST + d] = Vr1bh[jGlob*D_CONST + d];
                sh_Vr2[ld*D_CONST + d] = Vr2bh[jGlob*D_CONST + d];
                sh_gYj[ld*D_CONST + d] = gYbh[jGlob*D_CONST + d];
            }
            if (threadIdx.x == 0) {
                sh_mj[ld]   = mjBH[jGlob];
                sh_lj[ld]   = ljBH[jGlob];
                if constexpr (!CORRECTION_ONLY) {
                    sh_sumr[ld] = (sum_r + (int64_t)bh * N)[jGlob];
                }
            }
        }
        __syncthreads();

        // ============================================================
        // D-tiled dot products with j sub-tiling.
        // Precomputes i/k pairwise products per D_TILE, then sweeps j.
        // i/k values are loaded once per D_TILE instead of once per
        // (j, d) pair, eliminating 50% of shmem loads.
        //
        // Shmem loads per j-tile: 6,144 (was 12,288).
        // Register cost: 7 × J_SUB = 28 per-j accumulators (was 7 scalar).
        // ============================================================
        constexpr int J_SUB  = 4;  // j sub-tile size
        constexpr int D_TILE = 4;  // d tile size

        for (int jSub = 0; jSub < BLOCK_J && (jBase + jSub) < N; jSub += J_SUB) {
            // Per-j accumulators for this sub-tile
            float dot_j[J_SUB], d1_j[J_SUB], d2_j[J_SUB], d3_j[J_SUB];
            float d4_j[J_SUB], d5_j[J_SUB], d6_j[J_SUB];
            #pragma unroll
            for (int jj = 0; jj < J_SUB; jj++) {
                dot_j[jj] = 0.f; d1_j[jj] = 0.f; d2_j[jj] = 0.f; d3_j[jj] = 0.f;
                d4_j[jj] = 0.f; d5_j[jj] = 0.f; d6_j[jj] = 0.f;
            }

            // D-tiled precomputation: d-outer, j-inner
            for (int d_base = 0; d_base < D_CONST; d_base += D_TILE) {
                // Precompute 7 i/k pairwise products (8 shmem loads × D_TILE)
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
                    const float dyk  = sh_dYk [sh_k_off + d];

                    p_dot[dd] = qi * sk;
                    p_d1[dd]  = dyi * vs1k;
                    p_d2[dd]  = vq1i * vs1k;
                    p_d3[dd]  = vq1i * dyk;
                    p_d4[dd]  = dyi * vs2k;
                    p_d5[dd]  = vq2i * vs2k;
                    p_d6[dd]  = vq2i * dyk;
                }

                // Accumulate over j sub-tile. j-arrays have stride D_CONST
                // (no padding) and d_base is a multiple of D_TILE=4, so each
                // 4-float slice is 16-byte aligned → one LDS.128 per array.
                #pragma unroll
                for (int jj = 0; jj < J_SUB; jj++) {
                    const int jOff = jSub + jj;
                    if (jBase + jOff >= N) break;
                    const int rowOff = jOff * D_CONST + d_base;
                    const float4 rj4  = *reinterpret_cast<const float4*>(&sh_R  [rowOff]);
                    const float4 vr14 = *reinterpret_cast<const float4*>(&sh_Vr1[rowOff]);
                    const float4 vr24 = *reinterpret_cast<const float4*>(&sh_Vr2[rowOff]);
                    const float4 gyj4 = *reinterpret_cast<const float4*>(&sh_gYj[rowOff]);
                    const float rj[4]  = { rj4.x,  rj4.y,  rj4.z,  rj4.w  };
                    const float vr1[4] = { vr14.x, vr14.y, vr14.z, vr14.w };
                    const float vr2[4] = { vr24.x, vr24.y, vr24.z, vr24.w };
                    const float gyj[4] = { gyj4.x, gyj4.y, gyj4.z, gyj4.w };
                    #pragma unroll
                    for (int dd = 0; dd < D_TILE; dd++) {
                        dot_j[jj] += p_dot[dd] * rj[dd];
                        d1_j[jj]  += p_d1[dd]  * vr1[dd];
                        d2_j[jj]  += p_d2[dd]  * gyj[dd];
                        d3_j[jj]  += p_d3[dd]  * vr1[dd];
                        d4_j[jj]  += p_d4[dd]  * vr2[dd];
                        d5_j[jj]  += p_d5[dd]  * gyj[dd];
                        d6_j[jj]  += p_d6[dd]  * vr2[dd];
                    }
                }
            }

            // Process accumulated dot products for this sub-tile
            #pragma unroll
            for (int jj = 0; jj < J_SUB; jj++) {
                const int jOff = jSub + jj;
                if (jBase + jOff >= N) break;

                const float logits = dot_j[jj] * scale;
                const float Aq = __expf(fminf(logits - mi, EXP_CLIP)) / fmaxf(li, DENOM_EPS);
                const float Ar = __expf(fminf(logits - sh_mj[jOff], EXP_CLIP)) / fmaxf(sh_lj[jOff], DENOM_EPS);
                const float As = __expf(fminf(logits - mk, EXP_CLIP)) / fmaxf(lk, DENOM_EPS);

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
                    // Factored accumulation: float4 load per 4 d's (LDS.128).
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
        // Algebraic factoring epilogue:
        //   gradQ[i,d] = scale * rj_weighted[d] * S[k,d]
        //   gradS[k,d] = scale * rj_weighted[d] * Q[i,d]
        // S[k,d] and Q[i,d] are still in shared memory (loaded before j-loop).
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
 * R_grad_kernel - Computes gradR with Jacobian corrections.
 *
 * CORRECTION_ONLY=true:  Computes correction sum sum_r[j]
 * CORRECTION_ONLY=false: Computes gradR using precomputed corrections
 */
template<bool CORRECTION_ONLY, int BLOCK_J, int BLOCK_I, int BLOCK_K, int D_CONST, int REG_CAP = D_CONST>
__global__ void __launch_bounds__(256, 1) R_grad_kernel(
    const float* __restrict__ Q, const float* __restrict__ R, const float* __restrict__ S,
    const float* __restrict__ Vq1, const float* __restrict__ Vq2,
    const float* __restrict__ Vr1, const float* __restrict__ Vr2,
    const float* __restrict__ Vs1, const float* __restrict__ Vs2,
    const float* __restrict__ gradY,
    const float* __restrict__ m_i, const float* __restrict__ l_i,
    const float* __restrict__ m_j, const float* __restrict__ l_j,
    const float* __restrict__ m_k, const float* __restrict__ l_k,
    float* __restrict__ sum_q, float* __restrict__ sum_r, float* __restrict__ sum_s,
    float* __restrict__ gradR,
    int N, float scale)
{
    const int j0 = blockIdx.x * BLOCK_J + threadIdx.x;
    const int k0 = blockIdx.y * BLOCK_K + threadIdx.y;
    const int bh = blockIdx.z;
    const bool valid = (j0 < N && k0 < N);

    // Per (B,H) base pointers
    const int64_t stride_BH = (int64_t)N * D_CONST;
    const float* Qbh   = Q   + bh * stride_BH;
    const float* Rbh   = R   + bh * stride_BH;
    const float* Sbh   = S   + bh * stride_BH;
    const float* Vq1bh = Vq1 + bh * stride_BH;
    const float* Vq2bh = Vq2 + bh * stride_BH;
    const float* Vr1bh = Vr1 + bh * stride_BH;
    const float* Vr2bh = Vr2 + bh * stride_BH;
    const float* Vs1bh = Vs1 + bh * stride_BH;
    const float* Vs2bh = Vs2 + bh * stride_BH;
    const float* gYbh  = gradY + bh * stride_BH;
    const float* miBH  = m_i + bh * N;
    const float* liBH  = l_i + bh * N;
    const float* mjBH  = m_j + bh * N;
    const float* ljBH  = l_j + bh * N;
    const float* mkBH  = m_k + bh * N;
    const float* lkBH  = l_k + bh * N;
    float* sum_rBH = sum_r + bh * N;

    constexpr int D_PAD = D_CONST + 1;  // bank-conflict-free stride
    extern __shared__ float shmem[];

    // Persistent j/k data (padded stride)
    float* sh_Rj   = shmem;
    float* sh_Vr1j = sh_Rj   + BLOCK_J * D_PAD;
    float* sh_Vr2j = sh_Vr1j + BLOCK_J * D_PAD;
    float* sh_dYj  = sh_Vr2j + BLOCK_J * D_PAD;
    float* sh_Sk   = sh_dYj  + BLOCK_J * D_PAD;
    float* sh_Vs1k = sh_Sk   + BLOCK_K * D_PAD;
    float* sh_Vs2k = sh_Vs1k + BLOCK_K * D_PAD;
    float* sh_dYk  = sh_Vs2k + BLOCK_K * D_PAD;
    // I-tile data (streamed, padded to remove cooperative-store bank conflicts)
    float* sh_Q    = sh_dYk  + BLOCK_K * D_PAD;
    float* sh_Vq1  = sh_Q    + BLOCK_I * D_PAD;
    float* sh_Vq2  = sh_Vq1  + BLOCK_I * D_PAD;
    float* sh_dYi  = sh_Vq2  + BLOCK_I * D_PAD;
    float* sh_mi   = sh_dYi  + BLOCK_I * D_PAD;
    float* sh_li   = sh_mi   + BLOCK_I;
    float* sh_sumq = sh_li   + BLOCK_I;

    // Cooperative load of j-indexed and k-indexed data into shared memory
    {
        const int tid = threadIdx.x + threadIdx.y * BLOCK_J;
        const int nThreads = BLOCK_J * BLOCK_K;
        for (int idx = tid; idx < BLOCK_J * D_CONST; idx += nThreads) {
            const int jj = idx / D_CONST;
            const int dd = idx % D_CONST;
            const int jGlob = blockIdx.x * BLOCK_J + jj;
            if (jGlob < N) {
                sh_Rj  [jj * D_PAD + dd] = Rbh  [jGlob * D_CONST + dd];
                sh_Vr1j[jj * D_PAD + dd] = Vr1bh[jGlob * D_CONST + dd];
                sh_Vr2j[jj * D_PAD + dd] = Vr2bh[jGlob * D_CONST + dd];
                sh_dYj [jj * D_PAD + dd] = gYbh [jGlob * D_CONST + dd];
            } else {
                sh_Rj  [jj * D_PAD + dd] = 0.0f;
                sh_Vr1j[jj * D_PAD + dd] = 0.0f;
                sh_Vr2j[jj * D_PAD + dd] = 0.0f;
                sh_dYj [jj * D_PAD + dd] = 0.0f;
            }
        }
        for (int idx = tid; idx < BLOCK_K * D_CONST; idx += nThreads) {
            const int kk = idx / D_CONST;
            const int dd = idx % D_CONST;
            const int kGlob = blockIdx.y * BLOCK_K + kk;
            if (kGlob < N) {
                sh_Sk  [kk * D_PAD + dd] = Sbh  [kGlob * D_CONST + dd];
                sh_Vs1k[kk * D_PAD + dd] = Vs1bh[kGlob * D_CONST + dd];
                sh_Vs2k[kk * D_PAD + dd] = Vs2bh[kGlob * D_CONST + dd];
                sh_dYk [kk * D_PAD + dd] = gYbh [kGlob * D_CONST + dd];
            } else {
                sh_Sk  [kk * D_PAD + dd] = 0.0f;
                sh_Vs1k[kk * D_PAD + dd] = 0.0f;
                sh_Vs2k[kk * D_PAD + dd] = 0.0f;
                sh_dYk [kk * D_PAD + dd] = 0.0f;
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

    // Stream i-tiles through shared memory
    for (int iBase = 0; iBase < N; iBase += BLOCK_I) {
        // Cooperative load of i-tile
        for (int ld = threadIdx.y; ld < BLOCK_I && (iBase + ld) < N; ld += BLOCK_K) {
            const int iGlob = iBase + ld;
            for (int d = threadIdx.x; d < D_CONST; d += BLOCK_J) {
                sh_Q[ld*D_PAD + d]   = Qbh[iGlob*D_CONST + d];
                sh_Vq1[ld*D_PAD + d] = Vq1bh[iGlob*D_CONST + d];
                sh_Vq2[ld*D_PAD + d] = Vq2bh[iGlob*D_CONST + d];
                sh_dYi[ld*D_PAD + d] = gYbh[iGlob*D_CONST + d];
            }
            if (threadIdx.x == 0) {
                sh_mi[ld]   = miBH[iGlob];
                sh_li[ld]   = liBH[iGlob];
                if constexpr (!CORRECTION_ONLY) {
                    sh_sumq[ld] = (sum_q + (int64_t)bh * N)[iGlob];
                }
            }
        }
        __syncthreads();

        // D-tiled dot products with i sub-tiling.
        // Mirrors the QS kernel strategy: precompute thread-invariant j/k products
        // once per D_TILE and reuse them across several i rows.
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
                    const float dyk  = sh_dYk [sh_k_off + d];

                    p_dot[dd] = rj * sk;
                    p_d1[dd]  = vr1j * vs1k;
                    p_d2[dd]  = dyj * vs1k;
                    p_d3[dd]  = dyk * vr1j;
                    p_d4[dd]  = vr2j * vs2k;
                    p_d5[dd]  = dyj * vs2k;
                    p_d6[dd]  = dyk * vr2j;
                }

                #pragma unroll
                for (int ii = 0; ii < I_SUB; ++ii) {
                    const int iOff = iSub + ii;
                    if (iBase + iOff >= N) break;
                    const int iRow = iOff * D_PAD + d_base;
                    #pragma unroll
                    for (int dd = 0; dd < D_TILE; ++dd) {
                        const float qi  = sh_Q  [iRow + dd];
                        const float vq1 = sh_Vq1[iRow + dd];
                        const float vq2 = sh_Vq2[iRow + dd];
                        const float dyi = sh_dYi[iRow + dd];
                        dot_i[ii] += qi  * p_dot[dd];
                        if constexpr (!CORRECTION_ONLY) d1_i[ii] += dyi * p_d1[dd];
                        d2_i[ii] += vq1 * p_d2[dd];
                        if constexpr (!CORRECTION_ONLY) d3_i[ii] += vq1 * p_d3[dd];
                        d4_i[ii] += dyi * p_d4[dd];
                        if constexpr (!CORRECTION_ONLY) d5_i[ii] += vq2 * p_d5[dd];
                        d6_i[ii] += vq2 * p_d6[dd];
                    }
                }
            }

            #pragma unroll
            for (int ii = 0; ii < I_SUB; ++ii) {
                const int iOff = iSub + ii;
                if (iBase + iOff >= N) break;
                const float mi = sh_mi[iOff];
                const float li = sh_li[iOff];

                const float logits = dot_i[ii] * scale;
                const float Aq = __expf(fminf(logits - mi, EXP_CLIP)) / fmaxf(li, DENOM_EPS);
                const float Ar = __expf(fminf(logits - mj, EXP_CLIP)) / fmaxf(lj, DENOM_EPS);
                const float As = __expf(fminf(logits - mk, EXP_CLIP)) / fmaxf(lk, DENOM_EPS);
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
                    const int iRow = iOff * D_PAD;
                    for (int d = 0; d < D_CONST; ++d)
                        grad_acc[d] += grad_A * sh_Q[iRow + d] * sh_Sk[sh_k_off + d];
                }
            }
        }
        __syncthreads();
    }

    // ======== EPILOGUE ========
    if constexpr (CORRECTION_ONLY) {
        // Block reduction for sum_r[j]: reduce reg_sum_r across k-dim (threadIdx.y)
        float* reduce_buf = shmem;  // reuse shared memory (i-tile data done)

        reduce_buf[threadIdx.x * BLOCK_K + threadIdx.y] = valid ? reg_sum_r : 0.0f;
        __syncthreads();
        for (int s = BLOCK_K / 2; s > 0; s >>= 1) {
            if (threadIdx.y < s)
                reduce_buf[threadIdx.x * BLOCK_K + threadIdx.y] +=
                    reduce_buf[threadIdx.x * BLOCK_K + threadIdx.y + s];
            __syncthreads();
        }
        if (threadIdx.y == 0 && j0 < N)
            atomicAdd(&sum_rBH[j0], reduce_buf[threadIdx.x * BLOCK_K]);
    } else {
        // Write result (atomic due to k-dimension overlap)
        float* gRbh = gradR + bh * stride_BH;
        if (valid) {
            for (int d = 0; d < D_CONST; ++d)
                atomicAdd(&gRbh[j0*D_CONST + d], scale * grad_acc[d]);
        }
    }
}



// =============================================================================
// D-dispatch: routes to D_CONST=32 or D_CONST=64 template instantiation
// =============================================================================
#define DISPATCH_D(D_VAL, ...) \
  [&] { \
    if ((D_VAL) == 16)      { constexpr int D_TMPL = 16; __VA_ARGS__; } \
    else if ((D_VAL) == 32) { constexpr int D_TMPL = 32; __VA_ARGS__; } \
    else if ((D_VAL) == 64) { constexpr int D_TMPL = 64; __VA_ARGS__; } \
    else { TORCH_CHECK(false, "backward: unsupported D=", (D_VAL), ". Supported: 16, 32, 64"); } \
  }()

// =============================================================================
// Internal implementation that uses pre-computed softmax stats
// =============================================================================
static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
backward_impl(torch::Tensor grad_output,
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
              double dropout_rate) {
                
  // ============================================================================
  // 1. EXTRACT DIMENSIONS AND CONSTANTS
  // ============================================================================
  const int B = Q.size(0);
  const int H = Q.size(1);
  const int N = Q.size(2); //i think N and I/J/K are aliases, deal with later
  const int I = Q.size(2);
  const int J = R.size(2);
  const int K = S.size(2);
  const int D = Q.size(3);
  const int N_grad = grad_output.size(2);
  const float scale = 1.0f / sqrtf(static_cast<float>(D));

  // ============================================================================
  // 2. ALLOCATE GRADIENT TENSORS
  // ============================================================================
  auto grad_Q = torch::zeros_like(Q);
  auto grad_R = torch::zeros_like(R);
  auto grad_S = torch::zeros_like(S);
  auto grad_Vq_1 = torch::zeros_like(Vq_1);
  auto grad_Vq_2 = torch::zeros_like(Vq_2);
  auto grad_Vr_1 = torch::zeros_like(Vr_1);
  auto grad_Vr_2 = torch::zeros_like(Vr_2);
  auto grad_Vs_1 = torch::zeros_like(Vs_1);
  auto grad_Vs_2 = torch::zeros_like(Vs_2);

  auto optionsBH = Q.options();
  auto sum_q = torch::zeros({B, H, N}, optionsBH);
  auto sum_r = torch::zeros({B, H, N}, optionsBH);
  auto sum_s = torch::zeros({B, H, N}, optionsBH);

  // ============================================================================
  // 3. COMPUTE grad_{Vq,Vr,Vs}_1 (GATHER-GRAD KERNELS)
  // ============================================================================
  DISPATCH_D(D, {
    constexpr int TI = T_I;
    constexpr int TK = T_K;
    dim3 block_dim(TI, TK);
    dim3 grid_dim((N + TI - 1) / TI, (N + TK - 1) / TK, B * H);

    // --- grad_Vq_1 ---
    {
      // Gather kernels use static shared memory sized by D_TMPL
      Vq_gather_grad<D_TMPL><<<grid_dim, block_dim, 0, at::cuda::getCurrentCUDAStream()>>>(
          Q.data_ptr<float>(),
          R.data_ptr<float>(),
          S.data_ptr<float>(),
          Vs_1.data_ptr<float>(),
          Vr_1.data_ptr<float>(),
          grad_output.data_ptr<float>(),
          m_j.data_ptr<float>(),
          l_j.data_ptr<float>(),
          m_k.data_ptr<float>(),
          l_k.data_ptr<float>(),
          grad_Vq_1.data_ptr<float>(),
          N, scale);
    }

    // --- grad_Vr_1 ---
    {
      Vr_gather_grad<D_TMPL><<<grid_dim, block_dim, 0, at::cuda::getCurrentCUDAStream()>>>(
          Q.data_ptr<float>(),
          R.data_ptr<float>(),
          S.data_ptr<float>(),
          Vq_1.data_ptr<float>(),
          Vs_1.data_ptr<float>(),
          grad_output.data_ptr<float>(),
          m_i.data_ptr<float>(),
          l_i.data_ptr<float>(),
          m_k.data_ptr<float>(),
          l_k.data_ptr<float>(),
          grad_Vr_1.data_ptr<float>(),
          N, scale);
    }

    // --- grad_Vs_1 ---
    {
      Vs_gather_grad<D_TMPL><<<grid_dim, block_dim, 0, at::cuda::getCurrentCUDAStream()>>>(
          Q.data_ptr<float>(),
          R.data_ptr<float>(),
          S.data_ptr<float>(),
          Vq_1.data_ptr<float>(),
          Vr_1.data_ptr<float>(),
          grad_output.data_ptr<float>(),
          m_i.data_ptr<float>(),
          l_i.data_ptr<float>(),
          m_j.data_ptr<float>(),
          l_j.data_ptr<float>(),
          grad_Vs_1.data_ptr<float>(),
          N, scale);
    }
  });

  // ============================================================================
  // 4. COMPUTE grad_{Vq,Vr,Vs}_2 (SCATTER-GRAD KERNELS)
  // ============================================================================
  DISPATCH_D(D, {
    constexpr int TI = T_I;
    constexpr int TK = T_K;
    dim3 block_dim(TI, TK);
    dim3 grid_dim((N + TI - 1) / TI, (N + TK - 1) / TK, B * H);
    size_t shmem_bytes =
        T_J * D_TMPL * 3 * sizeof(float) + T_J * 2 * sizeof(float);

    // --- grad_Vq_2 ---
    Vq_scatter_grad<D_TMPL><<<grid_dim, block_dim, shmem_bytes, at::cuda::getCurrentCUDAStream()>>>(
        Q.data_ptr<float>(),
        R.data_ptr<float>(),
        S.data_ptr<float>(),
        Vr_2.data_ptr<float>(),
        Vs_2.data_ptr<float>(),
        grad_output.data_ptr<float>(),
        m_i.data_ptr<float>(),
        l_i.data_ptr<float>(),
        m_j.data_ptr<float>(),
        l_j.data_ptr<float>(),
        m_k.data_ptr<float>(),
        l_k.data_ptr<float>(),
        grad_Vq_2.data_ptr<float>(),
        N, scale);
    AT_CUDA_CHECK(cudaGetLastError());

    // --- grad_Vr_2 ---
    Vr_scatter_grad<D_TMPL><<<grid_dim, block_dim, shmem_bytes, at::cuda::getCurrentCUDAStream()>>>(
        Q.data_ptr<float>(),
        R.data_ptr<float>(),
        S.data_ptr<float>(),
        Vq_2.data_ptr<float>(),
        Vs_2.data_ptr<float>(),
        grad_output.data_ptr<float>(),
        m_i.data_ptr<float>(),
        l_i.data_ptr<float>(),
        m_j.data_ptr<float>(),
        l_j.data_ptr<float>(),
        m_k.data_ptr<float>(),
        l_k.data_ptr<float>(),
        grad_Vr_2.data_ptr<float>(),
        N, scale);

    // --- grad_Vs_2 ---
    Vs_scatter_grad<D_TMPL><<<grid_dim, block_dim, shmem_bytes, at::cuda::getCurrentCUDAStream()>>>(
        Q.data_ptr<float>(),
        R.data_ptr<float>(),
        S.data_ptr<float>(),
        Vq_2.data_ptr<float>(),
        Vr_2.data_ptr<float>(),
        grad_output.data_ptr<float>(),
        m_i.data_ptr<float>(),
        l_i.data_ptr<float>(),
        m_j.data_ptr<float>(),
        l_j.data_ptr<float>(),
        m_k.data_ptr<float>(),
        l_k.data_ptr<float>(),
        grad_Vs_2.data_ptr<float>(),
        N, scale);
  });


  // ===========================================================================
  // 5. JACOBIAN CORRECTIONS + 6. GRAD Q/S/R
  //    All dispatched through D template
  // ============================================================================
  DISPATCH_D(D, {
    // Phase 1: Correction sums
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
          4 * corrI * D_PAD_c * sizeof(float) +
          4 * corrK * D_PAD_c * sizeof(float) +
          4 * corrJ * D_TMPL * sizeof(float) +
          3 * corrJ * sizeof(float);

      cudaFuncSetAttribute(
          QS_grad_kernel<true, corrI, corrJ, corrK, D_TMPL>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          shmem_corr_qs);

      QS_grad_kernel<true, corrI, corrJ, corrK, D_TMPL>
          <<<grid_qs, block_qs, shmem_corr_qs, at::cuda::getCurrentCUDAStream()>>>(
              Q.data_ptr<float>(),
              R.data_ptr<float>(),
              S.data_ptr<float>(),
              Vq_1.data_ptr<float>(),
              Vq_2.data_ptr<float>(),
              Vr_1.data_ptr<float>(),
              Vr_2.data_ptr<float>(),
              Vs_1.data_ptr<float>(),
              Vs_2.data_ptr<float>(),
              grad_output.data_ptr<float>(),
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
              N, scale);

      AT_CUDA_CHECK(cudaGetLastError());
    }

    // Phase 2: grad_Q + grad_S (fused)
    {
      constexpr int tileI = TILE_I;
      constexpr int tileK = TILE_K;
      constexpr int tileJ = 16;

      dim3 block_dim(tileI, tileK);
      dim3 grid_dim((N + tileI - 1) / tileI,
                    (N + tileK - 1) / tileK,
                    B * H);

      constexpr int D_PAD_g = D_TMPL + 1;
      const size_t shmem_bytes =
          4 * tileI * D_PAD_g * sizeof(float) +
          4 * tileK * D_PAD_g * sizeof(float) +
          4 * tileJ * D_TMPL * sizeof(float) +
          3 * tileJ * sizeof(float);

      cudaFuncSetAttribute(
          QS_grad_kernel<false, tileI, tileJ, tileK, D_TMPL>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          shmem_bytes);

      QS_grad_kernel<false, tileI, tileJ, tileK, D_TMPL>
          <<<grid_dim, block_dim, shmem_bytes, at::cuda::getCurrentCUDAStream()>>>(
              Q.data_ptr<float>(),
              R.data_ptr<float>(),
              S.data_ptr<float>(),
              Vq_1.data_ptr<float>(),
              Vq_2.data_ptr<float>(),
              Vr_1.data_ptr<float>(),
              Vr_2.data_ptr<float>(),
              Vs_1.data_ptr<float>(),
              Vs_2.data_ptr<float>(),
              grad_output.data_ptr<float>(),
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
              N, scale);

      AT_CUDA_CHECK(cudaGetLastError());
    }

    // Phase 3: grad_R
    {
      constexpr int tileJ = TILE_J;
      constexpr int tileK = TILE_K;
      constexpr int tileI = 16;

      dim3 block_dim(tileJ, tileK);
      dim3 grid_dim((N + tileJ - 1) / tileJ,
                    (N + tileK - 1) / tileK,
                    B * H);

      constexpr int D_PAD_r = D_TMPL + 1;
      const size_t shmem_bytes =
          4 * tileJ * D_PAD_r * sizeof(float) +
          4 * tileK * D_PAD_r * sizeof(float) +
          4 * tileI * D_PAD_r * sizeof(float) +
          3 * tileI * sizeof(float);

      cudaFuncSetAttribute(
          R_grad_kernel<false, tileJ, tileI, tileK, D_TMPL>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          shmem_bytes);

      R_grad_kernel<false, tileJ, tileI, tileK, D_TMPL>
          <<<grid_dim, block_dim, shmem_bytes, at::cuda::getCurrentCUDAStream()>>>(
              Q.data_ptr<float>(),
              R.data_ptr<float>(),
              S.data_ptr<float>(),
              Vq_1.data_ptr<float>(),
              Vq_2.data_ptr<float>(),
              Vr_1.data_ptr<float>(),
              Vr_2.data_ptr<float>(),
              Vs_1.data_ptr<float>(),
              Vs_2.data_ptr<float>(),
              grad_output.data_ptr<float>(),
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
              N, scale);

      AT_CUDA_CHECK(cudaGetLastError());
    }
  });

  cudaDeviceSynchronize();

  return std::make_tuple(
      grad_Q,
      grad_R,
      grad_S,
      grad_Vq_1,
      grad_Vq_2,
      grad_Vr_1,
      grad_Vr_2,
      grad_Vs_1,
      grad_Vs_2);
}

// =============================================================================
// Public API: backward_cuda (uses pre-computed softmax stats from forward pass)
// =============================================================================
// NOTE: The forward pass computes and returns softmax stats (m_i, l_i, m_j, l_j, m_k, l_k).

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
backward_cuda(torch::Tensor grad_output,
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
              double dropout_rate) {
                
  // Ensure all tensors are contiguous
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
  m_i = m_i.contiguous();
  l_i = l_i.contiguous();
  m_j = m_j.contiguous();
  l_j = l_j.contiguous();
  m_k = m_k.contiguous();
  l_k = l_k.contiguous();

  // Call the internal implementation directly with provided stats
  return backward_impl(grad_output, Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2,
                       m_i, l_i, m_j, l_j, m_k, l_k, dropout_rate);
}

// #lalala