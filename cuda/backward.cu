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
    int  N, int D,
    float scale )
{
    /* ---- decode indices ------------------------------------------------ */
    int bh  = blockIdx.z;          // flattened (batch, head)
    int i0  = blockIdx.x * T_I + threadIdx.x;
    int k0  = blockIdx.y * T_K + threadIdx.y;
    
    // Track if this thread should compute (DON'T return early - need all threads for shared mem loading)
    const bool active = (i0 < N && k0 < N);

    const int64_t stride_BH = (int64_t)N * D;
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
    float q_vec[MAX_D_REG];
    float s_vec[MAX_D_REG], vs_vec[MAX_D_REG];
    float grad_acc[MAX_D_REG] = {0.0f};
    
    // Clamp indices for safe memory access (inactive threads load from valid location)
    const int i0_safe = min(i0, N-1);
    const int k0_safe = min(k0, N-1);
    
    #pragma unroll
    for (int d=0; d<D; ++d){
        q_vec[d]  = QBH[i0_safe*D + d];
        s_vec[d]  = SBH[k0_safe*D + d];
        vs_vec[d] = VsBH[k0_safe*D + d];
    }


    /* ---- loop over J in chunks ---------------------------------------- */
    for (int jBase=0; jBase<N; jBase+=T_J){
        __shared__ float sh_R [T_J][MAX_D_REG];
        __shared__ float sh_Vr[T_J][MAX_D_REG];
        __shared__ float sh_gY[T_J][MAX_D_REG];
        __shared__ float sh_mj[T_J];
        __shared__ float sh_lj[T_J];

        // Cooperative loading: ALL threads participate to cover all D dimensions
        int lj = threadIdx.y;                       // 0 … T_K-1 (≤T_J)
        if (lj < T_J && (jBase+lj) < N){
            int jGlob = jBase + lj;
            #pragma unroll
            for (int d=threadIdx.x; d<D; d+=T_I){
                sh_R [lj][d] = RBH [jGlob*D + d];
                sh_Vr[lj][d] = VrBH[jGlob*D + d];
                sh_gY[lj][d] = gYBH[jGlob*D + d];   // grad_Yr = grad_output
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
                for (int d=0; d<D; ++d)
                    logits += q_vec[d]*sh_R[jOff][d]*s_vec[d];
                logits *= scale;

                float wj = __expf(fminf(logits - sh_mj[jOff], EXP_CLIP)) / fmaxf(sh_lj[jOff], DENOM_EPS);
                float wk = __expf(fminf(logits - mKBH[k0], EXP_CLIP))    / fmaxf(lKBH[k0], DENOM_EPS);

                #pragma unroll
                for (int d=0; d<D; ++d){
                    grad_acc[d] += wj * sh_gY[jOff][d] * vs_vec[d]        /* Yr path */
                                  + wk * gYBH[k0*D + d]  * sh_Vr[jOff][d];/* Ys path */
                }
            }
        }
        __syncthreads();
    }

    /* ---- atomic add to global grad (only active threads) --------------- */
    if (active) {
        #pragma unroll
        for (int d=0; d<D; ++d)
            atomicAdd(&gVqBH[i0*D + d], grad_acc[d]);
    }
}

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
    int  N, int D,
    float scale )
{
    int bh  = blockIdx.z;
    int j0  = blockIdx.x * T_I + threadIdx.x;
    int k0  = blockIdx.y * T_K + threadIdx.y;
    
    // Track if this thread should compute (DON'T return early - need all threads for shared mem loading)
    const bool active = (j0 < N && k0 < N);

    const int64_t stride_BH = (int64_t)N * D;
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

    float r_vec[MAX_D_REG];
    float s_vec[MAX_D_REG];
    float vs_vec[MAX_D_REG];
    float gy_k_vec[MAX_D_REG];

    #pragma unroll
    for (int d=0; d<D; ++d){
        r_vec[d]    = RBH[j0_safe*D + d];
        s_vec[d]    = SBH[k0_safe*D + d];
        vs_vec[d]   = VsBH[k0_safe*D + d];
        gy_k_vec[d] = gYBH[k0_safe*D + d];
    }

    float grad_acc[MAX_D_REG] = {0.0f};
    float m_k_val = mKBH[k0_safe];
    float l_k_val = lKBH[k0_safe];


    for (int iBase=0; iBase<N; iBase+=T_J){
        __shared__ float sh_Q [T_J][MAX_D_REG];
        __shared__ float sh_Vq[T_J][MAX_D_REG];
        __shared__ float sh_gY[T_J][MAX_D_REG];
        __shared__ float sh_mi[T_J];
        __shared__ float sh_li[T_J];

        // Cooperative loading: ALL threads participate to cover all D dimensions
        int li = threadIdx.y;
        if (li < T_J && (iBase+li) < N){
            int iGlob = iBase + li;
            #pragma unroll
            for (int d=threadIdx.x; d<D; d+=T_I){
                sh_Q [li][d] = QBH [iGlob*D + d];
                sh_Vq[li][d] = VqBH[iGlob*D + d];
                sh_gY[li][d] = gYBH[iGlob*D + d];
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
                for (int d=0; d<D; ++d)
                    logits += sh_Q[iOff][d] * r_vec[d] * s_vec[d];
                logits *= scale;

                float wi = __expf(fminf(logits - sh_mi[iOff], EXP_CLIP)) / fmaxf(sh_li[iOff], DENOM_EPS);
                float wk = __expf(fminf(logits - m_k_val, EXP_CLIP))     / fmaxf(l_k_val, DENOM_EPS);

                #pragma unroll
                for (int d=0; d<D; ++d){
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
        for (int d=0; d<D; ++d)
            atomicAdd(&gVrBH[j0*D + d], grad_acc[d]);
    }
}

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
    int  N, int D,
    float scale )
{
    int bh  = blockIdx.z;
    int i0  = blockIdx.x * T_I + threadIdx.x;
    int k0  = blockIdx.y * T_K + threadIdx.y;
    
    // Track if this thread should compute (DON'T return early - need all threads for shared mem loading)
    const bool active = (i0 < N && k0 < N);

    const int64_t stride_BH = (int64_t)N * D;
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

    float q_vec[MAX_D_REG];
    float s_vec[MAX_D_REG];
    float vq_vec[MAX_D_REG];
    float gy_i_vec[MAX_D_REG];

    #pragma unroll
    for (int d=0; d<D; ++d){
        q_vec[d]    = QBH[i0_safe*D + d];
        s_vec[d]    = SBH[k0_safe*D + d];
        vq_vec[d]   = VqBH[i0_safe*D + d];
        gy_i_vec[d] = gYBH[i0_safe*D + d];
    }

    float grad_acc[MAX_D_REG] = {0.0f};
    float m_i_val = mIBH[i0_safe];
    float l_i_val = lIBH[i0_safe];


    for (int jBase=0; jBase<N; jBase+=T_J){
        __shared__ float sh_R [T_J][MAX_D_REG];
        __shared__ float sh_Vr[T_J][MAX_D_REG];
        __shared__ float sh_gY[T_J][MAX_D_REG];
        __shared__ float sh_mj[T_J];
        __shared__ float sh_lj[T_J];

        // Cooperative loading: ALL threads participate to cover all D dimensions
        int lj = threadIdx.y;
        if (lj < T_J && (jBase+lj) < N){
            int jGlob = jBase + lj;
            #pragma unroll
            for (int d=threadIdx.x; d<D; d+=T_I){
                sh_R [lj][d] = RBH [jGlob*D + d];
                sh_Vr[lj][d] = VrBH[jGlob*D + d];
                sh_gY[lj][d] = gYBH[jGlob*D + d];
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
                for (int d=0; d<D; ++d)
                    logits += q_vec[d] * sh_R[jOff][d] * s_vec[d];
                logits *= scale;

                float wi = __expf(fminf(logits - m_i_val, EXP_CLIP)) / fmaxf(l_i_val, DENOM_EPS);
                float wj = __expf(fminf(logits - sh_mj[jOff], EXP_CLIP)) / fmaxf(sh_lj[jOff], DENOM_EPS);

                #pragma unroll
                for (int d=0; d<D; ++d){
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
        for (int d=0; d<D; ++d)
            atomicAdd(&gVsBH[k0*D + d], grad_acc[d]);
    }
}

// ===================== scatter-grad Vq2, Vr2, Vs2 ======================


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
    int N, int D, float scale)
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
    const int64_t stride_BH = (int64_t)N * D;
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
    float q_vec[MAX_D_REG];
    float s_vec[MAX_D_REG], vs2_vec[MAX_D_REG];
    #pragma unroll
    for (int d=0; d<D; ++d){
        q_vec[d]  = QBH[i0_safe*D + d];
        s_vec[d]  = SBH[k0_safe*D + d];
        vs2_vec[d]= Vs2BH[k0_safe*D + d];
    }
    float grad_acc[MAX_D_REG] = {0.0f};

    // ---- shared memory tiles for (j) ----------------------------------
    extern __shared__ float shmem[];
    float* sh_R   = shmem;                   // T_J * D
    float* sh_Vr2 = sh_R   + T_J * D;        // T_J * D
    float* sh_gYr = sh_Vr2 + T_J * D;        // T_J * D   (dYr[j,:])
    float* sh_mj  = (float*)(sh_gYr + T_J * D);  // T_J scalars
    float* sh_lj  = sh_mj  + T_J;

    /* iterate over J tiles --------------------------------------------- */
    for (int jBase=0; jBase < N; jBase+=T_J){
        // cooperative load by all (i,k) threads inside TB
        const int ld_idx = threadIdx.y;  // reuse y-dimension for co-load rows
        if (ld_idx < T_J && (jBase+ld_idx) < N){
            const int jGlob = jBase + ld_idx;
            for (int d=threadIdx.x; d<D; d+=T_I){
                sh_R  [ld_idx*D + d] = RBH  [jGlob*D + d];
                sh_Vr2[ld_idx*D + d] = Vr2BH[jGlob*D + d];
                sh_gYr[ld_idx*D + d] = gYBH [jGlob*D + d];
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
                for (int d=0; d<D; ++d)
                    dot += q_vec[d] * sh_R[jOff*D + d] * s_vec[d];
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
                const float* dYr_vec = &sh_gYr[jOff*D];
                const float* Vr2_vec = &sh_Vr2[jOff*D];
                const float* dYs_vec = &gYBH[k0*D]; // contiguous in global, fine

                #pragma unroll
                for (int d=0; d<D; ++d){
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
        for (int d=0; d<D; ++d)
            atomicAdd(&gVqBH[i0*D + d], grad_acc[d]);
    }
}

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
    int N, int D, float scale)
{
    const int j0 = blockIdx.x * T_I + threadIdx.x;
    const int k0 = blockIdx.y * T_K + threadIdx.y;
    const int bh = blockIdx.z;
    
    // ALL threads must participate in cooperative loading - use active flag instead of early return
    const bool active = (j0 < N && k0 < N);
    // Clamped indices for safe memory access during cooperative loading
    const int j0_safe = min(j0, N - 1);
    const int k0_safe = min(k0, N - 1);

    const int64_t stride_BH = (int64_t)N * D;
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

    float r_vec[MAX_D_REG];
    float s_vec[MAX_D_REG];
    float vs2_vec[MAX_D_REG];
    float gy_k_vec[MAX_D_REG];
    #pragma unroll
    for (int d=0; d<D; ++d){
        r_vec[d]    = RBH[j0_safe*D + d];
        s_vec[d]    = SBH[k0_safe*D + d];
        vs2_vec[d]  = Vs2BH[k0_safe*D + d];
        gy_k_vec[d] = gYBH[k0_safe*D + d];
    }

    float grad_acc[MAX_D_REG] = {0.0f};

    extern __shared__ float shmem[];
    float* sh_Q   = shmem;                    // T_J * D
    float* sh_Vq2 = sh_Q   + T_J * D;         // T_J * D
    float* sh_gYq = sh_Vq2 + T_J * D;         // T_J * D
    float* sh_mi  = (float*)(sh_gYq + T_J * D); // T_J
    float* sh_li  = sh_mi + T_J;              // T_J

    for (int iBase=0; iBase < N; iBase+=T_J){
        const int li = threadIdx.y;
        if (li < T_J && (iBase + li) < N){
            const int iGlob = iBase + li;
            for (int d=threadIdx.x; d<D; d+=T_I){
                sh_Q  [li*D + d] = QBH [iGlob*D + d];
                sh_Vq2[li*D + d] = Vq2BH[iGlob*D + d];
                sh_gYq[li*D + d] = gYBH [iGlob*D + d];
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
                for (int d=0; d<D; ++d)
                    dot += sh_Q[iOff*D + d] * r_vec[d] * s_vec[d];
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

                const float* dy_q_vec = &sh_gYq[iOff*D];
                const float* vq2_vec  = &sh_Vq2[iOff*D];

                #pragma unroll
                for (int d=0; d<D; ++d){
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
        for (int d=0; d<D; ++d)
            atomicAdd(&gVrBH[j0*D + d], grad_acc[d]);
    }
}

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
    int N, int D, float scale)
{
    const int i0 = blockIdx.x * T_I + threadIdx.x;
    const int k0 = blockIdx.y * T_K + threadIdx.y;
    const int bh = blockIdx.z;
    
    // ALL threads must participate in cooperative loading - use active flag instead of early return
    const bool active = (i0 < N && k0 < N);
    // Clamped indices for safe memory access during cooperative loading
    const int i0_safe = min(i0, N - 1);
    const int k0_safe = min(k0, N - 1);

    const int64_t stride_BH = (int64_t)N * D;
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

    float q_vec[MAX_D_REG];
    float s_vec[MAX_D_REG];
    float vq2_vec[MAX_D_REG];
    float gy_i_vec[MAX_D_REG];
    #pragma unroll
    for (int d=0; d<D; ++d){
        q_vec[d]    = QBH[i0_safe*D + d];
        s_vec[d]    = SBH[k0_safe*D + d];
        vq2_vec[d]  = Vq2BH[i0_safe*D + d];
        gy_i_vec[d] = gYBH [i0_safe*D + d];
    }

    float grad_acc[MAX_D_REG] = {0.0f};

    extern __shared__ float shmem[];
    float* sh_R   = shmem;                    // T_J * D
    float* sh_Vr2 = sh_R   + T_J * D;         // T_J * D
    float* sh_gYr = sh_Vr2 + T_J * D;         // T_J * D
    float* sh_mj  = (float*)(sh_gYr + T_J * D);
    float* sh_lj  = sh_mj + T_J;

    for (int jBase=0; jBase < N; jBase+=T_J){
        int lj = threadIdx.y;
        if (lj < T_J && (jBase + lj) < N){
            int jGlob = jBase + lj;
            #pragma unroll
            for (int d=threadIdx.x; d<D; d+=T_I){
                sh_R  [lj*D + d] = RBH [jGlob*D + d];
                sh_Vr2[lj*D + d] = Vr2BH[jGlob*D + d];
                sh_gYr[lj*D + d] = gYBH[jGlob*D + d];
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
                for (int d=0; d<D; ++d)
                    dot += q_vec[d] * sh_R[jOff*D + d] * s_vec[d];
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

                const float* vr2_vec = &sh_Vr2[jOff*D];
                const float* dy_r_vec = &sh_gYr[jOff*D];

                #pragma unroll
                for (int d=0; d<D; ++d){
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
        for (int d=0; d<D; ++d)
            atomicAdd(&gVsBH[k0*D + d], grad_acc[d]);
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
 * QS_grad_kernel - Two-mode kernel for Jacobian corrections + gradQ/gradS
 * 
 * CORRECTION_ONLY=true:  Computes sum_q[i] and sum_s[k] correction sums
 *                        using 2D block reduction (replaces 3D jacobian_corrections)
 * CORRECTION_ONLY=false: Computes gradQ and gradS using precomputed corrections
 *                        (same as the original QS_grad_fused)
 *
 * Thread mapping: (i, k) grid, streams j-tiles in shared memory
 */
template<bool CORRECTION_ONLY, int BLOCK_I, int BLOCK_J, int BLOCK_K, int REG_CAP = MAX_D_REG>
__global__ void QS_grad_kernel(
    /* ---- operands ---------------------------------------------------- */
    const float* __restrict__ Q,   // [B,H,N,D]
    const float* __restrict__ R,   // [B,H,N,D]
    const float* __restrict__ S,   // [B,H,N,D]
    const float* __restrict__ Vq1, const float* __restrict__ Vq2,
    const float* __restrict__ Vr1, const float* __restrict__ Vr2,
    const float* __restrict__ Vs1, const float* __restrict__ Vs2,
    const float* __restrict__ gradY,   // [B,H,N,D]   (∂L/∂Y)
    /* ---- softmax statistics ----------------------------------------- */
    const float* __restrict__ m_i, const float* __restrict__ l_i,
    const float* __restrict__ m_j, const float* __restrict__ l_j,
    const float* __restrict__ m_k, const float* __restrict__ l_k,
    /* ---- correction sums (written when CORRECTION_ONLY, read otherwise) */
    float* __restrict__ sum_q,
    float* __restrict__ sum_r,
    float* __restrict__ sum_s,
    /* ---- gradient outputs (only used when !CORRECTION_ONLY) ---------- */
    float* __restrict__ gradQ,          // [B,H,N,D]
    float* __restrict__ gradS,          // [B,H,N,D]
    /* ---- sizes ------------------------------------------------------- */
    int  N, int D, float scale)
{
    const int i0 = blockIdx.x * BLOCK_I + threadIdx.x;
    const int k0 = blockIdx.y * BLOCK_K + threadIdx.y;
    const int bh = blockIdx.z;
    const bool valid = (i0 < N && k0 < N);

    // Per (B,H) base pointers
    const int64_t stride_BH = (int64_t)N * D;
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

    // Load i-indexed and k-indexed vectors into registers
    float Qi[REG_CAP] = {0.0f}, Sk[REG_CAP] = {0.0f};
    float Vq1i[REG_CAP] = {0.0f}, Vq2i[REG_CAP] = {0.0f};
    float Vs1k[REG_CAP] = {0.0f}, Vs2k[REG_CAP] = {0.0f};
    float dYi[REG_CAP] = {0.0f}, dYk[REG_CAP] = {0.0f};
    float mi = 0.0f, li = 1.0f, mk = 0.0f, lk = 1.0f;

    if (valid) {
        for (int d = 0; d < D; ++d) {
            Qi[d]   = Qbh[i0*D + d];
            Sk[d]   = Sbh[k0*D + d];
            Vq1i[d] = Vq1bh[i0*D + d];
            Vq2i[d] = Vq2bh[i0*D + d];
            Vs1k[d] = Vs1bh[k0*D + d];
            Vs2k[d] = Vs2bh[k0*D + d];
            dYi[d]  = gYbh[i0*D + d];
            dYk[d]  = gYbh[k0*D + d];
        }
        mi = miBH[i0];  li = liBH[i0];
        mk = mkBH[k0];  lk = lkBH[k0];
    }

    // Mode-dependent state
    float reg_sum_q = 0.0f;   // correction mode accumulators
    float reg_sum_s = 0.0f;
    float sumQi = 0.0f, sumSk = 0.0f;    // gradient mode: precomputed corrections
    float gradQ_acc[REG_CAP];              // gradient mode: output accumulators
    float gradS_acc[REG_CAP];
    if constexpr (!CORRECTION_ONLY) {
        if (valid) {
            sumQi = sum_qBH[i0];
            sumSk = sum_sBH[k0];
        }
        for (int d = 0; d < REG_CAP; ++d) { gradQ_acc[d] = 0.0f; gradS_acc[d] = 0.0f; }
    }

    // Shared memory layout: [R | Vr1 | Vr2 | gYj | mj | lj | sumr_buf/sumr_read]
    extern __shared__ float shmem[];
    float* sh_R    = shmem;
    float* sh_Vr1  = sh_R   + BLOCK_J * D;
    float* sh_Vr2  = sh_Vr1 + BLOCK_J * D;
    float* sh_gYj  = sh_Vr2 + BLOCK_J * D;
    float* sh_mj   = sh_gYj + BLOCK_J * D;
    float* sh_lj   = sh_mj  + BLOCK_J;
    float* sh_sumr = sh_lj  + BLOCK_J;  // gradient mode: reads sum_r per j
                                          // correction mode: smem accumulator for sum_r per j

    // Stream j-tiles through shared memory
    for (int jBase = 0; jBase < N; jBase += BLOCK_J) {
        // In correction mode, zero the sum_r accumulator for this tile
        if constexpr (CORRECTION_ONLY) {
            const int lid = threadIdx.x + threadIdx.y * BLOCK_I;
            if (lid < BLOCK_J) sh_sumr[lid] = 0.0f;
        }

        // Cooperative load of j-tile (all threads participate for __syncthreads)
        for (int ld = threadIdx.y; ld < BLOCK_J && (jBase + ld) < N; ld += BLOCK_K) {
            const int jGlob = jBase + ld;
            for (int d = threadIdx.x; d < D; d += BLOCK_I) {
                sh_R[ld*D + d]   = Rbh[jGlob*D + d];
                sh_Vr1[ld*D + d] = Vr1bh[jGlob*D + d];
                sh_Vr2[ld*D + d] = Vr2bh[jGlob*D + d];
                sh_gYj[ld*D + d] = gYbh[jGlob*D + d];
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

        // Process each j in the tile
        for (int jOff = 0; jOff < BLOCK_J && (jBase + jOff) < N; ++jOff) {
            // Fused computation: logit + 7 unique dot products (d1-d6)
            float dot = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f, d4 = 0.f, d5 = 0.f, d6 = 0.f;
            for (int d = 0; d < D; ++d) {
                const float rj  = sh_R[jOff*D + d];
                const float vr1 = sh_Vr1[jOff*D + d];
                const float vr2 = sh_Vr2[jOff*D + d];
                const float gyj = sh_gYj[jOff*D + d];

                dot += Qi[d] * rj * Sk[d];
                d1 += dYi[d] * vr1 * Vs1k[d];
                d2 += gyj * Vq1i[d] * Vs1k[d];
                d3 += dYk[d] * Vq1i[d] * vr1;
                d4 += dYi[d] * vr2 * Vs2k[d];
                d5 += gyj * Vq2i[d] * Vs2k[d];
                d6 += dYk[d] * Vq2i[d] * vr2;
            }

            // Softmax weights
            const float logits = dot * scale;
            const float Aq = __expf(fminf(logits - mi, EXP_CLIP)) / fmaxf(li, DENOM_EPS);
            const float Ar = __expf(fminf(logits - sh_mj[jOff], EXP_CLIP)) / fmaxf(sh_lj[jOff], DENOM_EPS);
            const float As = __expf(fminf(logits - mk, EXP_CLIP)) / fmaxf(lk, DENOM_EPS);

            // Gradient components
            const float gAq = d1 + d5 * As + d6 * Ar;
            const float gAr = d2 + d4 * As + d6 * Aq;
            const float gAs = d3 + d4 * Ar + d5 * Aq;

            if constexpr (CORRECTION_ONLY) {
                // Accumulate all 3 correction sums in one pass:
                // sum_q, sum_s in registers; sum_r via smem atomics
                reg_sum_q += gAq * Aq;
                reg_sum_s += gAs * As;
                if (valid) atomicAdd(&sh_sumr[jOff], gAr * Ar);
            } else {
                // Jacobian-corrected combined gradient
                const float grad_A = (gAq - sumQi) * Aq
                                   + (gAr - sh_sumr[jOff]) * Ar
                                   + (gAs - sumSk) * As;

                // Accumulate to both outputs
                for (int d = 0; d < D; ++d) {
                    const float rj = sh_R[jOff*D + d];
                    gradQ_acc[d] += grad_A * rj * Sk[d];
                    gradS_acc[d] += grad_A * Qi[d] * rj;
                }
            }
        }
        __syncthreads();

        // In correction mode, flush sum_r smem buffer to global after each j-tile
        if constexpr (CORRECTION_ONLY) {
            const int lid = threadIdx.x + threadIdx.y * BLOCK_I;
            if (lid < BLOCK_J && (jBase + lid) < N)
                atomicAdd(&(sum_r + (int64_t)bh * N)[jBase + lid], sh_sumr[lid]);
            __syncthreads();
        }
    }

    // ======== EPILOGUE ========
    if constexpr (CORRECTION_ONLY) {
        // Block reduction for sum_q[i]: reduce reg_sum_q across k-dim (threadIdx.y)
        float* reduce_buf = shmem;  // reuse shared memory (j-tile data done)

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

        // Block reduction for sum_s[k]: reduce reg_sum_s across i-dim (threadIdx.x)
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
        // Write results (atomic due to k-dimension overlap for Q, i-dimension for S)
        float* gQbh = gradQ + bh * stride_BH;
        float* gSbh = gradS + bh * stride_BH;
        if (valid) {
            for (int d = 0; d < D; ++d) {
                atomicAdd(&gQbh[i0*D + d], scale * gradQ_acc[d]);
                atomicAdd(&gSbh[k0*D + d], scale * gradS_acc[d]);
            }
        }
    }
}

/**
 * R_grad_kernel - Two-mode kernel for Jacobian correction sum_r + gradR
 * 
 * CORRECTION_ONLY=true:  Computes sum_r[j] correction sum
 *                        using 2D block reduction (replaces 3D jacobian_corrections)
 * CORRECTION_ONLY=false: Computes gradR using precomputed corrections
 *                        (same as the original R_grad)
 *
 * Thread mapping: (j, k) grid, streams i-tiles in shared memory
 */
template<bool CORRECTION_ONLY, int BLOCK_J, int BLOCK_I, int BLOCK_K, int REG_CAP = MAX_D_REG>
__global__ void R_grad_kernel(
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
    int N, int D, float scale)
{
    const int j0 = blockIdx.x * BLOCK_J + threadIdx.x;
    const int k0 = blockIdx.y * BLOCK_K + threadIdx.y;
    const int bh = blockIdx.z;
    const bool valid = (j0 < N && k0 < N);

    // Per (B,H) base pointers
    const int64_t stride_BH = (int64_t)N * D;
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

    // Load j-indexed and k-indexed vectors into registers
    float Rj[REG_CAP] = {0.0f}, Sk[REG_CAP] = {0.0f};
    float Vr1j[REG_CAP] = {0.0f}, Vr2j[REG_CAP] = {0.0f};
    float Vs1k[REG_CAP] = {0.0f}, Vs2k[REG_CAP] = {0.0f};
    float dYj[REG_CAP] = {0.0f}, dYk[REG_CAP] = {0.0f};
    float mj = 0.0f, lj = 1.0f, mk = 0.0f, lk = 1.0f;

    if (valid) {
        for (int d = 0; d < D; ++d) {
            Rj[d]   = Rbh[j0*D + d];
            Sk[d]   = Sbh[k0*D + d];
            Vr1j[d] = Vr1bh[j0*D + d];
            Vr2j[d] = Vr2bh[j0*D + d];
            Vs1k[d] = Vs1bh[k0*D + d];
            Vs2k[d] = Vs2bh[k0*D + d];
            dYj[d]  = gYbh[j0*D + d];
            dYk[d]  = gYbh[k0*D + d];
        }
        mj = mjBH[j0];  lj = ljBH[j0];
        mk = mkBH[k0];  lk = lkBH[k0];
    }

    // Mode-dependent state
    float reg_sum_r = 0.0f;                   // correction mode accumulator
    float sumRj = 0.0f, sumSk = 0.0f;         // gradient mode: precomputed corrections
    float grad_acc[REG_CAP];                   // gradient mode: output accumulator
    if constexpr (!CORRECTION_ONLY) {
        if (valid) {
            sumRj = sum_rBH[j0];
            sumSk = (sum_s + (int64_t)bh * N)[k0];
        }
        for (int d = 0; d < REG_CAP; ++d) grad_acc[d] = 0.0f;
    }

    // Shared memory layout: [Q | Vq1 | Vq2 | dYi | mi | li | (sumq)]
    extern __shared__ float shmem[];
    float* sh_Q    = shmem;
    float* sh_Vq1  = sh_Q    + BLOCK_I * D;
    float* sh_Vq2  = sh_Vq1  + BLOCK_I * D;
    float* sh_dYi  = sh_Vq2  + BLOCK_I * D;
    float* sh_mi   = sh_dYi  + BLOCK_I * D;
    float* sh_li   = sh_mi   + BLOCK_I;
    float* sh_sumq = sh_li   + BLOCK_I;  // only used when !CORRECTION_ONLY

    // Stream i-tiles through shared memory
    for (int iBase = 0; iBase < N; iBase += BLOCK_I) {
        // Cooperative load of i-tile
        for (int ld = threadIdx.y; ld < BLOCK_I && (iBase + ld) < N; ld += BLOCK_K) {
            const int iGlob = iBase + ld;
            for (int d = threadIdx.x; d < D; d += BLOCK_J) {
                sh_Q[ld*D + d]   = Qbh[iGlob*D + d];
                sh_Vq1[ld*D + d] = Vq1bh[iGlob*D + d];
                sh_Vq2[ld*D + d] = Vq2bh[iGlob*D + d];
                sh_dYi[ld*D + d] = gYbh[iGlob*D + d];
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

        // Process each i in the tile
        for (int iOff = 0; iOff < BLOCK_I && (iBase + iOff) < N; ++iOff) {
            const float* Qi   = sh_Q   + iOff*D;
            const float* Vq1i = sh_Vq1 + iOff*D;
            const float* Vq2i = sh_Vq2 + iOff*D;
            const float* dYi  = sh_dYi + iOff*D;
            const float mi    = sh_mi[iOff];
            const float li    = sh_li[iOff];

            // Fused computation: logit + dot products
            // In correction mode, skip d1, d3, d5 (only needed for gAq/gAs, not gAr)
            float dot = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f, d4 = 0.f, d5 = 0.f, d6 = 0.f;
            for (int d = 0; d < D; ++d) {
                const float qi  = Qi[d];
                const float vq1 = Vq1i[d];
                const float vq2 = Vq2i[d];
                const float dyi = dYi[d];

                dot += qi * Rj[d] * Sk[d];
                if constexpr (!CORRECTION_ONLY) {
                    d1 += dyi * Vr1j[d] * Vs1k[d];
                }
                d2 += dYj[d] * vq1 * Vs1k[d];
                if constexpr (!CORRECTION_ONLY) {
                    d3 += dYk[d] * vq1 * Vr1j[d];
                }
                d4 += dyi * Vr2j[d] * Vs2k[d];
                if constexpr (!CORRECTION_ONLY) {
                    d5 += dYj[d] * vq2 * Vs2k[d];
                }
                d6 += dYk[d] * vq2 * Vr2j[d];
            }

            // Softmax weights
            const float logits = dot * scale;
            const float Aq = __expf(fminf(logits - mi, EXP_CLIP)) / fmaxf(li, DENOM_EPS);
            const float Ar = __expf(fminf(logits - mj, EXP_CLIP)) / fmaxf(lj, DENOM_EPS);
            const float As = __expf(fminf(logits - mk, EXP_CLIP)) / fmaxf(lk, DENOM_EPS);

            // gAr is needed in both modes
            const float gAr = d2 + d4 * As + d6 * Aq;

            if constexpr (CORRECTION_ONLY) {
                // Accumulate correction sum (scalar — no D-loop)
                reg_sum_r += gAr * Ar;
            } else {
                const float sumQi = sh_sumq[iOff];
                const float gAq = d1 + d5 * As + d6 * Ar;
                const float gAs = d3 + d4 * Ar + d5 * Aq;

                // Jacobian-corrected combined gradient
                const float grad_A = (gAq - sumQi) * Aq
                                   + (gAr - sumRj) * Ar
                                   + (gAs - sumSk) * As;

                // Accumulate to gradR
                for (int d = 0; d < D; ++d)
                    grad_acc[d] += grad_A * Qi[d] * Sk[d];
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
            for (int d = 0; d < D; ++d)
                atomicAdd(&gRbh[j0*D + d], scale * grad_acc[d]);
        }
    }
}



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
  const int N = Q.size(2);
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
  // 5. COMPUTE grad_{Vq,Vr,Vs}_1 (GATHER-GRAD KERNELS)
  // ============================================================================
  {
    // --- grad_Vq_1 ---
    constexpr int TI = T_I;
    constexpr int TK = T_K;

    dim3 block_dim(TI, TK);
    dim3 grid_dim((N + TI - 1) / TI, (N + TK - 1) / TK, B * H);
    size_t shmem_bytes = T_J * D * 3 * sizeof(float);  // R + Vr + gradY

    Vq_gather_grad<<<grid_dim, block_dim, shmem_bytes, at::cuda::getCurrentCUDAStream()>>>(
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
        N, D, scale);
  }

  {
    // --- grad_Vr_1 ---
    constexpr int TI = T_I;
    constexpr int TK = T_K;

    dim3 block_dim(TI, TK);
    dim3 grid_dim((N + TI - 1) / TI, (N + TK - 1) / TK, B * H);
    size_t shmem_bytes =
        T_J * D * 3 * sizeof(float) + T_J * 2 * sizeof(float);  // Q + Vq + gradY + mi/li

    Vr_gather_grad<<<grid_dim, block_dim, shmem_bytes, at::cuda::getCurrentCUDAStream()>>>(
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
        N, D, scale);
  }

  {
    // --- grad_Vs_1 ---
    constexpr int TI = T_I;
    constexpr int TK = T_K;

    dim3 block_dim(TI, TK);
    dim3 grid_dim((N + TI - 1) / TI, (N + TK - 1) / TK, B * H);
    size_t shmem_bytes =
        T_J * D * 3 * sizeof(float) + T_J * 2 * sizeof(float);  // R + Vr + gradY + mj/lj

    Vs_gather_grad<<<grid_dim, block_dim, shmem_bytes, at::cuda::getCurrentCUDAStream()>>>(
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
        N, D, scale);
  }

  // ============================================================================
  // 6. COMPUTE grad_{Vq,Vr,Vs}_2 (SCATTER-GRAD KERNELS)
  // ============================================================================
  {
    // --- grad_Vq_2 ---
    constexpr int TI = T_I;
    constexpr int TK = T_K;

    dim3 block_dim(TI, TK);
    dim3 grid_dim((N + TI - 1) / TI, (N + TK - 1) / TK, B * H);
    size_t shmem_bytes =
        T_J * D * 3 * sizeof(float) + T_J * 2 * sizeof(float);

    Vq_scatter_grad<<<grid_dim, block_dim, shmem_bytes, at::cuda::getCurrentCUDAStream()>>>(
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
        N, D, scale);

    AT_CUDA_CHECK(cudaGetLastError());
  }

  {
    // --- grad_Vr_2 ---
    constexpr int TI = T_I;
    constexpr int TK = T_K;

    dim3 block_dim(TI, TK);
    dim3 grid_dim((N + TI - 1) / TI, (N + TK - 1) / TK, B * H);
    size_t shmem_bytes =
        T_J * D * 3 * sizeof(float) + T_J * 2 * sizeof(float);

    Vr_scatter_grad<<<grid_dim, block_dim, shmem_bytes, at::cuda::getCurrentCUDAStream()>>>(
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
        N, D, scale);
  }

  {
    // --- grad_Vs_2 ---
    constexpr int TI = T_I;
    constexpr int TK = T_K;

    dim3 block_dim(TI, TK);
    dim3 grid_dim((N + TI - 1) / TI, (N + TK - 1) / TK, B * H);
    size_t shmem_bytes =
        T_J * D * 3 * sizeof(float) + T_J * 2 * sizeof(float);

    Vs_scatter_grad<<<grid_dim, block_dim, shmem_bytes, at::cuda::getCurrentCUDAStream()>>>(
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
        N, D, scale);
  }


  // ===========================================================================
  // 7. JACOBIAN CORRECTIONS via 2D-tiled correction passes
  //    (replaces the old 3D jacobian_corrections kernel)
  //    Phase 1a: QS_grad_kernel<true> computes sum_q and sum_s
  //    Phase 1b: R_grad_kernel<true>  computes sum_r
  // ============================================================================
  {
    // Correction pass uses SMALLER tiles (8×8 = 64 threads) to reduce register
    // pressure. At D=64, each thread needs ~512 registers for D-dimensional
    // arrays. With 64 threads: 32K regs/block → 2 blocks/SM (good occupancy).
    // With 256 threads: 131K regs/block → register spilling (bad).
    constexpr int corrI = 8;
    constexpr int corrK = 8;
    constexpr int corrJ = 16;       // j-tile streaming size (unchanged)

    TORCH_CHECK(D <= MAX_D_REG,
                "QS_grad_kernel requires D <= ", MAX_D_REG, ", but got D = ", D);

    dim3 block_qs(corrI, corrK);     // 64 threads (vs 256 in gradient pass)
    dim3 grid_qs((N + corrI - 1) / corrI,
                 (N + corrK - 1) / corrK,
                 B * H);

    // Correction pass: sh_sumr used as smem accumulator for sum_r
    const size_t shmem_corr_qs =
        4 * corrJ * D * sizeof(float) +    // sh_R, sh_Vr1, sh_Vr2, sh_gYj
        3 * corrJ * sizeof(float);          // sh_mj, sh_lj, sh_sumr (accumulator)

    QS_grad_kernel<true, corrI, corrJ, corrK>
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
            N, D, scale);

    AT_CUDA_CHECK(cudaGetLastError());
  }

  // NOTE: R_grad_kernel<true> correction launch is NOT needed — the single
  // QS_grad_kernel<true> above computes all three sums (sum_q, sum_r, sum_s)
  // using register accumulation for sum_q/sum_s and smem atomics for sum_r.

  // ===========================================================================
  // 8. COMPUTE grad_Q + grad_S (fused) and grad_R
  //    Phase 2: gradient passes read the now-complete correction sums
  // ============================================================================   

  {
    // --- grad_Q + grad_S fused accumulation ---
    constexpr int tileI = TILE_I;
    constexpr int tileK = TILE_K;
    constexpr int tileJ = 16;      // independent knob for J streaming

    dim3 block_dim(tileI, tileK);
    dim3 grid_dim((N + tileI - 1) / tileI,
                  (N + tileK - 1) / tileK,
                  B * H);

    const size_t shmem_bytes =
        4 * tileJ * D * sizeof(float) +  // R, Vr1, Vr2, gYj
        3 * tileJ * sizeof(float);       // mj, lj, sum_r

    QS_grad_kernel<false, tileI, tileJ, tileK>
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
            N, D, scale);

    AT_CUDA_CHECK(cudaGetLastError());
  }

  {
    // --- grad_R accumulation ---
    constexpr int tileJ = TILE_J;
    constexpr int tileK = TILE_K;
    constexpr int tileI = 16;

    dim3 block_dim(tileJ, tileK);
    dim3 grid_dim((N + tileJ - 1) / tileJ,
                  (N + tileK - 1) / tileK,
                  B * H);

    const size_t shmem_bytes =
        (4 * tileI * D + 3 * tileI) * sizeof(float);  // Q, Vq1, Vq2, dY + stats

    R_grad_kernel<false, tileJ, tileI, tileK>
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
            N, D, scale);

    AT_CUDA_CHECK(cudaGetLastError());
  }

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
// These MUST be passed to the backward pass to avoid redundant computation and ensure
// numerical consistency between forward and backward passes.
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
