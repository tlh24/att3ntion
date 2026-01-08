#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>      
#include <cuda.h>
#include <cuda_runtime.h>
#include "../cpp/manual_att3ntion.h"
#include <tuple>

constexpr float EXP_CLIP = 80.0f;   // safe range for expf in FP32
constexpr float DENOM_EPS = 1e-6f;  // avoid divide-by-zero

// Tunable tile sizes (shared defaults; override per build if needed)
#ifndef TILE_I
#define TILE_I 8
#endif
#ifndef TILE_J
#define TILE_J 8
#endif
#ifndef TILE_K
#define TILE_K 8
#endif

#ifndef MAX_D_REG
#define MAX_D_REG 64
#endif

// Gather/scatter kernels reuse these aliases so callers can override independently.
#ifndef T_I
#define T_I TILE_I
#endif
#ifndef T_J
#define T_J TILE_J
#endif
#ifndef T_K
#define T_K TILE_K
#endif

// ==================== softmax stats ====================== (TODO: optimize)


// Ar = softmax_{i,k}(A)
static __global__ void Ar_tiled_softmax(
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

    // numerical stability constant
    constexpr float EXP_CLIP = 80.0f;   // safe range for expf

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
            __syncthreads(); // CRITICAL: Ensure all threads read m_tile before red_buf is reused

            // --- Parallel reduction to find tile sum_exp (l_tile) ---
            float l_tile_thread = 0.0f;
            for (int flat_idx = tid; flat_idx < TILE_I * TILE_K; flat_idx += block_size) {
                if (p_tile[flat_idx] > -1e29f) { // Check if it's not padding
                    l_tile_thread += expf(fminf(p_tile[flat_idx] - m_tile, EXP_CLIP));
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
            l_block = expf(fminf(m_block - m_new, EXP_CLIP)) * l_block + expf(fminf(m_tile - m_new, EXP_CLIP)) * l_tile;
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


// Aq = softmax_{j,k}(A)
static __global__ void Aq_tiled_softmax(
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

    // numerical stability constant
    constexpr float EXP_CLIP = 80.0f;   // safe range for expf

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
            __syncthreads(); // CRITICAL: Ensure all threads read m_tile before red_buf is reused

            // --- Parallel reduction to find tile sum_exp (l_tile) ---
            float l_tile_thread = 0.0f;
            for (int flat_idx = tid; flat_idx < TILE_J * TILE_K; flat_idx += block_size) {
                if (p_tile[flat_idx] > -1e29f) { // Check if it's not padding
                    l_tile_thread += expf(fminf(p_tile[flat_idx] - m_tile, EXP_CLIP));
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
            l_block = expf(fminf(m_block - m_new, EXP_CLIP)) * l_block + expf(fminf(m_tile - m_new, EXP_CLIP)) * l_tile;
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

// As = softmax_{i,j}(A)
static __global__ void As_tiled_softmax(
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

    // numerical stability constant
    constexpr float EXP_CLIP = 80.0f;   // safe range for expf

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

            // Find tile max (m_tile) - with proper sync in reduction
            float m_tile_thread = -1e30f;
            for (int i = tid; i < TILE_I*TILE_J; i+=block_size) m_tile_thread = fmaxf(m_tile_thread, p_tile[i]);
            red_buf[tid] = m_tile_thread;
            __syncthreads();
            for (int s=block_size/2; s>0; s>>=1) {
                if (tid<s) red_buf[tid] = fmaxf(red_buf[tid], red_buf[tid+s]);
                __syncthreads(); // CRITICAL: sync inside reduction loop
            }
            float m_tile = red_buf[0];
            __syncthreads(); // CRITICAL: Ensure all threads read m_tile before red_buf is reused

            // Find tile sum_exp (l_tile) - with proper sync in reduction
            float l_tile_thread = 0.0f;
            for (int i = tid; i < TILE_I*TILE_J; i+=block_size) {
                if(p_tile[i] > -1e29f) l_tile_thread += expf(fminf(p_tile[i] - m_tile, EXP_CLIP));
            }
            red_buf[tid] = l_tile_thread;
            __syncthreads();
            for (int s=block_size/2; s>0; s>>=1) {
                if (tid<s) red_buf[tid] += red_buf[tid+s];
                __syncthreads(); // CRITICAL: sync inside reduction loop
            }
            float l_tile = red_buf[0];

            // Online update
            float m_new = fmaxf(m_block, m_tile);
            l_block = expf(fminf(m_block - m_new, EXP_CLIP)) * l_block + expf(fminf(m_tile - m_new, EXP_CLIP)) * l_tile;
            m_block = m_new;
            __syncthreads();
        }
    }
    
    if (tid == 0) {
        m_k_out[bh_idx * K + k_idx] = m_block;
        l_k_out[bh_idx * K + k_idx] = l_block;
    }
}

// ===================== gather-grad Vq1, Vr1, Vs1 ======================

__global__ void grad_gather_Vq_kernel(
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

    // numerical stability constants
    constexpr float EXP_CLIP = 80.0f;   // safe range for expf
    constexpr float DENOM_EPS = 1e-6f;  // prevent division by zero

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

__global__ void grad_gather_Vr_kernel(
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

    // numerical stability constants
    constexpr float EXP_CLIP = 80.0f;   // safe range for expf
    constexpr float DENOM_EPS = 1e-6f;  // prevent division by zero

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

__global__ void grad_gather_Vs_kernel(
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

    // numerical stability constants
    constexpr float EXP_CLIP = 80.0f;   // safe range for expf
    constexpr float DENOM_EPS = 1e-6f;  // prevent division by zero

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


__global__ void grad_scatter_Vq_kernel(
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

    // numerical stability constants
    constexpr float EXP_CLIP = 80.0f;   // safe range for expf
    constexpr float DENOM_EPS = 1e-6f;

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

__global__ void grad_scatter_Vr_kernel(
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

__global__ void grad_scatter_Vs_kernel(
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

// ===================== precompute Jacobian corrections + helpers ==================


// Helper for integer division rounded up
__host__ __device__ inline int ceilDiv(int a, int b) {
    return (a + b - 1) / b;
}

// Device helper for 3-way element-wise product and sum reduction (dot product)
__device__ __forceinline__ float dot3(const float* a, const float* b, const float* c, int D) {
    float sum = 0.0f;
    #pragma unroll
    for (int d = 0; d < D; ++d) {
        sum += a[d] * b[d] * c[d];
    }
    return sum;
}

template<int Bq, int Br, int Bk>
__global__ void precompute_jacobian_corrections_kernel(
    // Inputs
    const float* Q,   const float* R,   const float* S,
    const float* Vq1, const float* Vq2,
    const float* Vr1, const float* Vr2,
    const float* Vs1, const float* Vs2,
    const float* grad_out,
    const float* m_i, const float* l_i,
    const float* m_j, const float* l_j,
    const float* m_k, const float* l_k,
    // Outputs
    float* sum_q, float* sum_r, float* sum_s,
    // Dimensions & Scale
    int N, int D, float scale
) {
    // Shared memory for tensor tiles
    extern __shared__ float smem[];
    float* Qi_smem   = smem;                                  // Bq x D
    float* Rj_smem   = Qi_smem   + Bq * D;                    // Br x D
    float* Sk_smem   = Rj_smem   + Br * D;                    // Bk x D
    float* Vq1i_smem = Sk_smem   + Bk * D;                    // Bq x D
    float* Vq2i_smem = Vq1i_smem + Bq * D;                    // Bq x D
    float* Vr1j_smem = Vq2i_smem + Bq * D;                    // Br x D
    float* Vr2j_smem = Vr1j_smem + Br * D;                    // Br x D
    float* Vs1k_smem = Vr2j_smem + Br * D;                    // Bk x D
    float* Vs2k_smem = Vs1k_smem + Bk * D;                    // Bk x D
    float* dYi_smem  = Vs2k_smem + Bk * D;                    // Bq x D
    float* dYj_smem  = dYi_smem  + Bq * D;                    // Br x D
    float* dYk_smem  = dYj_smem  + Br * D;                    // Bk x D

    // Tile base indices
    const int i0 = blockIdx.x * Bq;
    const int j0 = blockIdx.y * Br;
    const int k0 = blockIdx.z * Bk;

    // Thread indices within the tile
    const int ti = threadIdx.x;
    const int tj = threadIdx.y;
    const int tk = threadIdx.z;

    // Linear thread ID for cooperative loading
    const int lid = tk * (Bq * Br) + tj * Bq + ti;
    const int num_threads = Bq * Br * Bk;

    // Load tiles into shared memory
    // Each loop iterates over a type of tile (i-indexed, j-indexed, k-indexed)
    for (int i = lid; i < Bq * D; i += num_threads) {
        int row = i / D;
        int col = i % D;
        if (i0 + row < N) {
            Qi_smem[i]   = Q[(i0 + row) * D + col];
            Vq1i_smem[i] = Vq1[(i0 + row) * D + col];
            Vq2i_smem[i] = Vq2[(i0 + row) * D + col];
            dYi_smem[i]  = grad_out[(i0 + row) * D + col];
        } else {
            // Zero padding if tile extends beyond N
            Qi_smem[i] = 0.0f; Vq1i_smem[i] = 0.0f; Vq2i_smem[i] = 0.0f; dYi_smem[i] = 0.0f;
        }
    }
    for (int i = lid; i < Br * D; i += num_threads) {
        int row = i / D;
        int col = i % D;
        if (j0 + row < N) {
            Rj_smem[i]   = R[(j0 + row) * D + col];
            Vr1j_smem[i] = Vr1[(j0 + row) * D + col];
            Vr2j_smem[i] = Vr2[(j0 + row) * D + col];
            dYj_smem[i]  = grad_out[(j0 + row) * D + col];
        } else {
            Rj_smem[i] = 0.0f; Vr1j_smem[i] = 0.0f; Vr2j_smem[i] = 0.0f; dYj_smem[i] = 0.0f;
        }
    }
    for (int i = lid; i < Bk * D; i += num_threads) {
        int row = i / D;
        int col = i % D;
        if (k0 + row < N) {
            Sk_smem[i]   = S[(k0 + row) * D + col];
            Vs1k_smem[i] = Vs1[(k0 + row) * D + col];
            Vs2k_smem[i] = Vs2[(k0 + row) * D + col];
            dYk_smem[i]  = grad_out[(k0 + row) * D + col];
        } else {
            Sk_smem[i] = 0.0f; Vs1k_smem[i] = 0.0f; Vs2k_smem[i] = 0.0f; dYk_smem[i] = 0.0f;
        }
    }
    __syncthreads();

    // Global indices for this thread's (i, j, k) point
    const int i = i0 + ti;
    const int j = j0 + tj;
    const int k = k0 + tk;

    if (i < N && j < N && k < N) {
        const float* Qi_vec   = &Qi_smem[ti * D];
        const float* Rj_vec   = &Rj_smem[tj * D];
        const float* Sk_vec   = &Sk_smem[tk * D];
        const float* Vq1i_vec = &Vq1i_smem[ti * D];
        const float* Vq2i_vec = &Vq2i_smem[ti * D];
        const float* Vr1j_vec = &Vr1j_smem[tj * D];
        const float* Vr2j_vec = &Vr2j_smem[tj * D];
        const float* Vs1k_vec = &Vs1k_smem[tk * D];
        const float* Vs2k_vec = &Vs2k_smem[tk * D];
        const float* dYi_vec  = &dYi_smem[ti * D];
        const float* dYj_vec  = &dYj_smem[tj * D];
        const float* dYk_vec  = &dYk_smem[tk * D];

        // numerical stability constants
        constexpr float EXP_CLIP = 80.0f;   // safe range for expf
        constexpr float DENOM_EPS = 1e-6f;  // prevent division by zero

        // ---> Logits & Numerators
        float logits = dot3(Qi_vec, Rj_vec, Sk_vec, D) * scale;
        float Aq_num = expf(fminf(logits - m_i[i], EXP_CLIP));
        float Ar_num = expf(fminf(logits - m_j[j], EXP_CLIP));
        float As_num = expf(fminf(logits - m_k[k], EXP_CLIP));
        float Aq = Aq_num / fmaxf(l_i[i], DENOM_EPS);
        float Ar = Ar_num / fmaxf(l_j[j], DENOM_EPS);
        float As = As_num / fmaxf(l_k[k], DENOM_EPS);

        // ---> Compute grad_A components
        // --- grad_Aq ---
        float gAq  = dot3(dYi_vec, Vr1j_vec, Vs1k_vec, D);
              gAq += dot3(dYj_vec, Vq2i_vec, Vs2k_vec, D) * As;
              gAq += dot3(dYk_vec, Vq2i_vec, Vr2j_vec, D) * Ar;
        // --- grad_Ar ---
        float gAr  = dot3(dYj_vec, Vq1i_vec, Vs1k_vec, D);
              gAr += dot3(dYi_vec, Vr2j_vec, Vs2k_vec, D) * As;
              gAr += dot3(dYk_vec, Vq2i_vec, Vr2j_vec, D) * Aq;
        // --- grad_As ---
        float gAs  = dot3(dYk_vec, Vq1i_vec, Vr1j_vec, D);
              gAs += dot3(dYi_vec, Vr2j_vec, Vs2k_vec, D) * Ar;
              gAs += dot3(dYj_vec, Vq2i_vec, Vs2k_vec, D) * Aq;

        // ---> Accumulate dot products
        atomicAdd(&sum_q[i], gAq * Aq);
        atomicAdd(&sum_r[j], gAr * Ar);
        atomicAdd(&sum_s[k], gAs * As);
    }
}



// ===================== grad Q, R, S ====================== (TODO: optimize)

template<int BLOCK_I, int BLOCK_J, int BLOCK_K, int REG_CAP = MAX_D_REG>
__global__ void grad_Q_kernel(
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
    /* ---- correction scalars from pass-1 ----------------------------- */
    const float* __restrict__ sum_q,
    const float* __restrict__ sum_r,
    const float* __restrict__ sum_s,
    /* ---- output ------------------------------------------------------ */
    float* __restrict__ gradQ,          // [B,H,N,D]
    /* ---- sizes ------------------------------------------------------- */
    int  N, int D, float scale)
{
    /* ---- decode indices --------------------------------------------- */
    const int i0 = blockIdx.x * BLOCK_I + threadIdx.x;   // i-index owned by this lane
    const int k0 = blockIdx.y * BLOCK_K + threadIdx.y;   // k-index owned by this lane
    const int bh = blockIdx.z;                       // flattened (batch, head)

    /* NOTE: Do NOT early return - all threads must participate in __syncthreads()
       for cooperative loading. Use 'valid' flag to guard computation instead. */
    const bool valid = (i0 < N && k0 < N);

    /* ---- per (B,H) base pointers & strides -------------------------- */
    const int64_t stride_BH = (int64_t)N * D;
    const float* Qbh   = Q   + (int64_t)bh * stride_BH;
    const float* Rbh   = R   + (int64_t)bh * stride_BH;
    const float* Sbh   = S   + (int64_t)bh * stride_BH;
    const float* Vq1bh = Vq1 + (int64_t)bh * stride_BH;
    const float* Vq2bh = Vq2 + (int64_t)bh * stride_BH;
    const float* Vr1bh = Vr1 + (int64_t)bh * stride_BH;
    const float* Vr2bh = Vr2 + (int64_t)bh * stride_BH;
    const float* Vs1bh = Vs1 + (int64_t)bh * stride_BH;
    const float* Vs2bh = Vs2 + (int64_t)bh * stride_BH;
    const float* gYbh  = gradY + (int64_t)bh * stride_BH;

    const float* miBH  = m_i + (int64_t)bh * N;
    const float* liBH  = l_i + (int64_t)bh * N;
    const float* mjBH  = m_j + (int64_t)bh * N;
    const float* ljBH  = l_j + (int64_t)bh * N;
    const float* mkBH  = m_k + (int64_t)bh * N;
    const float* lkBH  = l_k + (int64_t)bh * N;

    const float* sum_qBH = sum_q + (int64_t)bh * N;
    const float* sum_rBH = sum_r + (int64_t)bh * N;
    const float* sum_sBH = sum_s + (int64_t)bh * N;

    float* gQbh = gradQ + (int64_t)bh * stride_BH;

    /* ---- registers for i-/k-local vectors --------------------------- */
    float Qi [REG_CAP] = {0.0f};
    float Sk [REG_CAP] = {0.0f};
    float Vq1i[REG_CAP] = {0.0f}, Vq2i[REG_CAP] = {0.0f};
    float Vs1k[REG_CAP] = {0.0f}, Vs2k[REG_CAP] = {0.0f};
    float dYi [REG_CAP] = {0.0f}, dYk [REG_CAP] = {0.0f};

    float mi = 0.0f, li = 1.0f, mk = 0.0f, lk = 1.0f;
    float sumQi = 0.0f, sumSk = 0.0f;

    /* Only load per-thread data if this thread owns a valid (i,k) pair */
    if (valid) {
        #pragma unroll
        for (int d=0; d<D; ++d){
            Qi [d] = Qbh [i0*D + d];
            Sk [d] = Sbh [k0*D + d];
            Vq1i[d] = Vq1bh[i0*D + d];
            Vq2i[d] = Vq2bh[i0*D + d];
            Vs1k[d] = Vs1bh[k0*D + d];
            Vs2k[d] = Vs2bh[k0*D + d];
            dYi [d] = gYbh [i0*D + d];
            dYk [d] = gYbh [k0*D + d];
        }

        mi    = miBH [i0];
        li    = liBH [i0];
        mk    = mkBH [k0];
        lk    = lkBH [k0];
        sumQi = sum_qBH[i0];
        sumSk = sum_sBH[k0];
    }

    /* ---- shared memory for J-tiles ---------------------------------- */
    extern __shared__ float shmem[];
    float* sh_R   = shmem;                    // BLOCK_J × D
    float* sh_Vr1 = sh_R   + BLOCK_J * D;
    float* sh_Vr2 = sh_Vr1 + BLOCK_J * D;
    float* sh_gYj = sh_Vr2 + BLOCK_J * D;
    float* sh_mj  = sh_gYj + BLOCK_J * D;         // BLOCK_J
    float* sh_lj  = sh_mj  + BLOCK_J;             // BLOCK_J
    float* sh_sumr= sh_lj  + BLOCK_J;             // BLOCK_J

    /* ---- accumulator for grad_Q[i,:] -------------------------------- */
    float grad_acc[REG_CAP] = {0.0f};

    // numerical stability constants
    constexpr float EXP_CLIP = 80.0f;   // safe range for expf
    constexpr float DENOM_EPS = 1e-6f;  // prevent division by zero

    /* ---- iterate over J in tiles ------------------------------------ */
    for (int jBase = 0; jBase < N; jBase += BLOCK_J)
    {
        /* cooperative load by all (i,k) threads ----------------------- */
        /* NOTE: BLOCK_J may exceed BLOCK_K, so we loop to cover all rows */
        for (int ld = threadIdx.y; ld < BLOCK_J && (jBase + ld) < N; ld += BLOCK_K) {
            const int jGlob = jBase + ld;
            #pragma unroll
            for (int d = threadIdx.x; d < D; d += BLOCK_I){
                sh_R   [ld*D + d] = Rbh  [jGlob*D + d];
                sh_Vr1 [ld*D + d] = Vr1bh[jGlob*D + d];
                sh_Vr2 [ld*D + d] = Vr2bh[jGlob*D + d];
                sh_gYj [ld*D + d] = gYbh [jGlob*D + d];
            }
            if (threadIdx.x == 0){
                sh_mj [ld]  = mjBH  [jGlob];
                sh_lj [ld]  = ljBH  [jGlob];
                sh_sumr[ld] = sum_rBH[jGlob];
            }
        }
        __syncthreads();

        /* ---- loop inside loaded J-tile ------------------------------ */
        for (int jOff = 0; jOff < BLOCK_J && (jBase + jOff) < N; ++jOff){
            /* --- logits & softmax numerators ------------------------ */
            float dot = 0.f;
            #pragma unroll
            for (int d=0; d<D; ++d)
                dot += Qi[d] * sh_R[jOff*D + d] * Sk[d];
            float logits = dot * scale;

            float Aq = __expf(fminf(logits - mi, EXP_CLIP))         / fmaxf(li, DENOM_EPS);
            float Ar = __expf(fminf(logits - sh_mj[jOff], EXP_CLIP)) / fmaxf(sh_lj[jOff], DENOM_EPS);
            float As = __expf(fminf(logits - mk, EXP_CLIP))         / fmaxf(lk, DENOM_EPS);

            /* --- grad_Aq ------------------------------------------- */
            float gAq = 0.f;
            #pragma unroll
            for (int d=0; d<D; ++d)
                gAq += dYi[d] * sh_Vr1[jOff*D + d] * Vs1k[d];
            float tmp = 0.f;
            #pragma unroll
            for (int d=0; d<D; ++d)
                tmp += sh_gYj[jOff*D + d] * Vq2i[d] * Vs2k[d];
            gAq += tmp * As;
            tmp = 0.f;
            #pragma unroll
            for (int d=0; d<D; ++d)
                tmp += dYk[d] * Vq2i[d] * sh_Vr2[jOff*D + d];
            gAq += tmp * Ar;

            /* --- grad_Ar ------------------------------------------- */
            float gAr = 0.f;
            #pragma unroll
            for (int d=0; d<D; ++d)
                gAr += sh_gYj[jOff*D + d] * Vq1i[d] * Vs1k[d];
            tmp = 0.f;
            #pragma unroll
            for (int d=0; d<D; ++d)
                tmp += dYi[d] * sh_Vr2[jOff*D + d] * Vs2k[d];
            gAr += tmp * As;
            tmp = 0.f;
            #pragma unroll
            for (int d=0; d<D; ++d)
                tmp += dYk[d] * Vq2i[d] * sh_Vr2[jOff*D + d];
            gAr += tmp * Aq;

            /* --- grad_As ------------------------------------------- */
            float gAs = 0.f;
            #pragma unroll
            for (int d=0; d<D; ++d)
                gAs += dYk[d] * Vq1i[d] * sh_Vr1[jOff*D + d];
            tmp = 0.f;
            #pragma unroll
            for (int d=0; d<D; ++d)
                tmp += dYi[d] * sh_Vr2[jOff*D + d] * Vs2k[d];
            gAs += tmp * Ar;
            tmp = 0.f;
            #pragma unroll
            for (int d=0; d<D; ++d)
                tmp += sh_gYj[jOff*D + d] * Vq2i[d] * Vs2k[d];
            gAs += tmp * Aq;

            /* --- Jacobian-corrected grad_A ------------------------- */
            float grad_A = (gAq -  sumQi          ) * Aq
                         + (gAr -  sh_sumr[jOff]  ) * Ar
                         + (gAs -  sumSk          ) * As;

            /* --- accumulate contribution to grad_Q[i,:] ------------ */
            #pragma unroll
            for (int d=0; d<D; ++d)
                grad_acc[d] += grad_A * sh_R[jOff*D + d] * Sk[d];
        }
        __syncthreads();
    }

    /* ---- atomic add to global grad_Q ------------------------------- */
    if (valid) {
        #pragma unroll
        for (int d=0; d<D; ++d)
            atomicAdd(&gQbh[i0*D + d], scale * grad_acc[d]);
    }
}

template<int BLOCK_J, int BLOCK_I, int BLOCK_K, int REG_CAP = MAX_D_REG>
__global__ void grad_R_kernel(
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
    /* ---- correction scalars from pass-1 ----------------------------- */
    const float* __restrict__ sum_q,
    const float* __restrict__ sum_r,
    const float* __restrict__ sum_s,
    /* ---- output ------------------------------------------------------ */
    float* __restrict__ gradR,          // [B,H,N,D]
    /* ---- sizes ------------------------------------------------------- */
    int  N, int D, float scale)
{
    /* ---- decode indices --------------------------------------------- */
    const int j0 = blockIdx.x * BLOCK_J + threadIdx.x;   // j-index owned by this lane
    const int k0 = blockIdx.y * BLOCK_K + threadIdx.y;   // k-index owned by this lane
    const int bh = blockIdx.z;                           // flattened (batch, head)

    /* NOTE: Do NOT early return here! All threads must participate in
       __syncthreads() for cooperative loading. Use validity flag instead. */
    const bool valid = (j0 < N && k0 < N);

    /* ---- per (B,H) base pointers & strides -------------------------- */
    const int64_t stride_BH = (int64_t)N * D;
    const float* Qbh   = Q   + (int64_t)bh * stride_BH;
    const float* Rbh   = R   + (int64_t)bh * stride_BH;
    const float* Sbh   = S   + (int64_t)bh * stride_BH;
    const float* Vq1bh = Vq1 + (int64_t)bh * stride_BH;
    const float* Vq2bh = Vq2 + (int64_t)bh * stride_BH;
    const float* Vr1bh = Vr1 + (int64_t)bh * stride_BH;
    const float* Vr2bh = Vr2 + (int64_t)bh * stride_BH;
    const float* Vs1bh = Vs1 + (int64_t)bh * stride_BH;
    const float* Vs2bh = Vs2 + (int64_t)bh * stride_BH;
    const float* gYbh  = gradY + (int64_t)bh * stride_BH;

    const float* miBH  = m_i + (int64_t)bh * N;
    const float* liBH  = l_i + (int64_t)bh * N;
    const float* mjBH  = m_j + (int64_t)bh * N;
    const float* ljBH  = l_j + (int64_t)bh * N;
    const float* mkBH  = m_k + (int64_t)bh * N;
    const float* lkBH  = l_k + (int64_t)bh * N;

    const float* sum_qBH = sum_q + (int64_t)bh * N;
    const float* sum_rBH = sum_r + (int64_t)bh * N;
    const float* sum_sBH = sum_s + (int64_t)bh * N;

    float* gRbh = gradR + (int64_t)bh * stride_BH;

    /* ---- registers for j-/k-local vectors --------------------------- */
    float Rj [REG_CAP] = {0.0f};
    float Sk [REG_CAP] = {0.0f};
    float Vr1j[REG_CAP] = {0.0f}, Vr2j[REG_CAP] = {0.0f};
    float Vs1k[REG_CAP] = {0.0f}, Vs2k[REG_CAP] = {0.0f};
    float dYj [REG_CAP] = {0.0f}, dYk [REG_CAP] = {0.0f};

    float mj = 0.0f, lj = 1.0f, mk = 0.0f, lk = 1.0f;
    float sumRj = 0.0f, sumSk = 0.0f;

    /* Only load per-thread data if this thread owns a valid (j,k) pair */
    if (valid) {
        #pragma unroll
        for (int d=0; d<D; ++d){
            Rj [d] = Rbh [j0*D + d];
            Sk [d] = Sbh [k0*D + d];
            Vr1j[d] = Vr1bh[j0*D + d];
            Vr2j[d] = Vr2bh[j0*D + d];
            Vs1k[d] = Vs1bh[k0*D + d];
            Vs2k[d] = Vs2bh[k0*D + d];
            dYj [d] = gYbh [j0*D + d];
            dYk [d] = gYbh [k0*D + d];
        }

        mj    = mjBH  [j0];
        lj    = ljBH  [j0];
        mk    = mkBH  [k0];
        lk    = lkBH  [k0];
        sumRj = sum_rBH[j0];
        sumSk = sum_sBH[k0];
    }

    /* ---- shared memory for I-tiles ---------------------------------- */
    extern __shared__ float shmem[];
    float* sh_Q    = shmem;                    // BLOCK_I × D
    float* sh_Vq1  = sh_Q    + BLOCK_I * D;
    float* sh_Vq2  = sh_Vq1  + BLOCK_I * D;
    float* sh_dYi  = sh_Vq2  + BLOCK_I * D;
    float* sh_mi   = sh_dYi  + BLOCK_I * D;    // BLOCK_I
    float* sh_li   = sh_mi   + BLOCK_I;        // BLOCK_I
    float* sh_sumq = sh_li   + BLOCK_I;        // BLOCK_I

    /* ---- accumulator for grad_R[j,:] -------------------------------- */
    float grad_acc[REG_CAP] = {0.0f};

    // numerical stability constants
    constexpr float EXP_CLIP = 80.0f;   // safe range for expf
    constexpr float DENOM_EPS = 1e-6f;  // prevent division by zero

    /* ---- iterate over I in tiles ------------------------------------ */
    for (int iBase = 0; iBase < N; iBase += BLOCK_I)
    {
        /* cooperative load by all (j,k) threads */
        /* NOTE: BLOCK_I may exceed BLOCK_K, so we loop to cover all rows */
        for (int ld = threadIdx.y; ld < BLOCK_I && (iBase + ld) < N; ld += BLOCK_K) {
            const int iGlob = iBase + ld;
            #pragma unroll
            for (int d = threadIdx.x; d < D; d += BLOCK_J){
                sh_Q   [ld*D + d] = Qbh  [iGlob*D + d];
                sh_Vq1 [ld*D + d] = Vq1bh[iGlob*D + d];
                sh_Vq2 [ld*D + d] = Vq2bh[iGlob*D + d];
                sh_dYi [ld*D + d] = gYbh [iGlob*D + d];
            }
            if (threadIdx.x == 0){
                sh_mi  [ld] = miBH  [iGlob];
                sh_li  [ld] = liBH  [iGlob];
                sh_sumq[ld] = sum_qBH[iGlob];
            }
        }
        __syncthreads();

        /* ---- loop inside loaded I-tile ------------------------------ */
        for (int iOff = 0; iOff < BLOCK_I && (iBase + iOff) < N; ++iOff){
            const float* Qi   = sh_Q   + iOff*D;
            const float* Vq1i = sh_Vq1 + iOff*D;
            const float* Vq2i = sh_Vq2 + iOff*D;
            const float* dYi  = sh_dYi + iOff*D;
            const float mi    = sh_mi  [iOff];
            const float li    = sh_li  [iOff];
            const float sumQi = sh_sumq[iOff];

            /* --- logits & softmax numerators ------------------------ */
            float dot = 0.f;
            #pragma unroll
            for (int d=0; d<D; ++d)
                dot += Qi[d] * Rj[d] * Sk[d];
            float logits = dot * scale;

            float Aq = __expf(fminf(logits - mi, EXP_CLIP)) / fmaxf(li, DENOM_EPS);
            float Ar = __expf(fminf(logits - mj, EXP_CLIP)) / fmaxf(lj, DENOM_EPS);
            float As = __expf(fminf(logits - mk, EXP_CLIP)) / fmaxf(lk, DENOM_EPS);

            /* --- grad_Aq ------------------------------------------- */
            float gAq = 0.f;
            #pragma unroll
            for (int d=0; d<D; ++d)
                gAq += dYi[d] * Vr1j[d] * Vs1k[d];

            float tmp = 0.f;
            #pragma unroll
            for (int d=0; d<D; ++d)
                tmp += dYj[d] * Vq2i[d] * Vs2k[d];
            gAq += tmp * As;

            tmp = 0.f;
            #pragma unroll
            for (int d=0; d<D; ++d)
                tmp += dYk[d] * Vq2i[d] * Vr2j[d];
            gAq += tmp * Ar;

            /* --- grad_Ar ------------------------------------------- */
            float gAr = 0.f;
            #pragma unroll
            for (int d=0; d<D; ++d)
                gAr += dYj[d] * Vq1i[d] * Vs1k[d];

            tmp = 0.f;
            #pragma unroll
            for (int d=0; d<D; ++d)
                tmp += dYi[d] * Vr2j[d] * Vs2k[d];
            gAr += tmp * As;

            tmp = 0.f;
            #pragma unroll
            for (int d=0; d<D; ++d)
                tmp += dYk[d] * Vq2i[d] * Vr2j[d];
            gAr += tmp * Aq;

            /* --- grad_As ------------------------------------------- */
            float gAs = 0.f;
            #pragma unroll
            for (int d=0; d<D; ++d)
                gAs += dYk[d] * Vq1i[d] * Vr1j[d];

            tmp = 0.f;
            #pragma unroll
            for (int d=0; d<D; ++d)
                tmp += dYi[d] * Vr2j[d] * Vs2k[d];
            gAs += tmp * Ar;

            tmp = 0.f;
            #pragma unroll
            for (int d=0; d<D; ++d)
                tmp += dYj[d] * Vq2i[d] * Vs2k[d];
            gAs += tmp * Aq;

            /* --- Jacobian-corrected grad_A ------------------------- */
            float grad_A = (gAq -  sumQi ) * Aq
                         + (gAr -  sumRj ) * Ar
                         + (gAs -  sumSk ) * As;

            /* --- accumulate contribution to grad_R[j,:] ------------ */
            #pragma unroll
            for (int d=0; d<D; ++d)
                grad_acc[d] += grad_A * Qi[d] * Sk[d];
        }
        __syncthreads();
    }

    /* ---- atomic add to global grad_R ------------------------------- */
    if (valid) {
        #pragma unroll
        for (int d=0; d<D; ++d)
            atomicAdd(&gRbh[j0*D + d], scale * grad_acc[d]);
    }
}

template<int BLOCK_K, int BLOCK_J, int BLOCK_I, int REG_CAP = MAX_D_REG>
__global__ void grad_S_kernel(
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
    /* ---- correction scalars from pass-1 ----------------------------- */
    const float* __restrict__ sum_q,
    const float* __restrict__ sum_r,
    const float* __restrict__ sum_s,
    /* ---- output ------------------------------------------------------ */
    float* __restrict__ gradS,          // [B,H,N,D]
    /* ---- sizes ------------------------------------------------------- */
    int  N, int D, float scale)
{
    /* ---- decode indices --------------------------------------------- */
    const int k0 = blockIdx.x * BLOCK_K + threadIdx.x;   // k-index owned by this lane
    const int i0 = blockIdx.y * BLOCK_I + threadIdx.y;   // i-index owned by this lane
    const int bh = blockIdx.z;                           // flattened (batch, head)

    /* NOTE: Do NOT early return here! All threads must participate in
       __syncthreads() for cooperative loading. Use validity flag instead. */
    const bool valid = (k0 < N && i0 < N);

    /* ---- per (B,H) base pointers & strides -------------------------- */
    const int64_t stride_BH = (int64_t)N * D;
    const float* Qbh   = Q   + (int64_t)bh * stride_BH;
    const float* Rbh   = R   + (int64_t)bh * stride_BH;
    const float* Sbh   = S   + (int64_t)bh * stride_BH;
    const float* Vq1bh = Vq1 + (int64_t)bh * stride_BH;
    const float* Vq2bh = Vq2 + (int64_t)bh * stride_BH;
    const float* Vr1bh = Vr1 + (int64_t)bh * stride_BH;
    const float* Vr2bh = Vr2 + (int64_t)bh * stride_BH;
    const float* Vs1bh = Vs1 + (int64_t)bh * stride_BH;
    const float* Vs2bh = Vs2 + (int64_t)bh * stride_BH;
    const float* gYbh  = gradY + (int64_t)bh * stride_BH;

    const float* miBH  = m_i + (int64_t)bh * N;
    const float* liBH  = l_i + (int64_t)bh * N;
    const float* mjBH  = m_j + (int64_t)bh * N;
    const float* ljBH  = l_j + (int64_t)bh * N;
    const float* mkBH  = m_k + (int64_t)bh * N;
    const float* lkBH  = l_k + (int64_t)bh * N;

    const float* sum_qBH = sum_q + (int64_t)bh * N;
    const float* sum_rBH = sum_r + (int64_t)bh * N;
    const float* sum_sBH = sum_s + (int64_t)bh * N;

    float* gSbh = gradS + (int64_t)bh * stride_BH;

    /* ---- registers for k-/i-local vectors --------------------------- */
    float Sk [REG_CAP] = {0.0f};
    float Qi [REG_CAP] = {0.0f};
    float Vs1k[REG_CAP] = {0.0f}, Vs2k[REG_CAP] = {0.0f};
    float Vq1i[REG_CAP] = {0.0f}, Vq2i[REG_CAP] = {0.0f};
    float dYk [REG_CAP] = {0.0f}, dYi [REG_CAP] = {0.0f};

    float mi = 0.0f, li = 1.0f, mk = 0.0f, lk = 1.0f;
    float sumQi = 0.0f, sumSk = 0.0f;

    /* Only load per-thread data if this thread owns a valid (k,i) pair */
    if (valid) {
        #pragma unroll
        for (int d=0; d<D; ++d){
            Sk [d] = Sbh [k0*D + d];
            Qi [d] = Qbh [i0*D + d];
            Vs1k[d] = Vs1bh[k0*D + d];
            Vs2k[d] = Vs2bh[k0*D + d];
            Vq1i[d] = Vq1bh[i0*D + d];
            Vq2i[d] = Vq2bh[i0*D + d];
            dYk [d] = gYbh [k0*D + d];
            dYi [d] = gYbh [i0*D + d];
        }

        mi    = miBH  [i0];
        li    = liBH  [i0];
        mk    = mkBH  [k0];
        lk    = lkBH  [k0];
        sumQi = sum_qBH[i0];
        sumSk = sum_sBH[k0];
    }

    /* ---- shared memory for J-tiles ---------------------------------- */
    extern __shared__ float shmem[];
    float* sh_R    = shmem;                    // BLOCK_J × D
    float* sh_Vr1  = sh_R    + BLOCK_J * D;
    float* sh_Vr2  = sh_Vr1  + BLOCK_J * D;
    float* sh_dYj  = sh_Vr2  + BLOCK_J * D;
    float* sh_mj   = sh_dYj  + BLOCK_J * D;    // BLOCK_J
    float* sh_lj   = sh_mj   + BLOCK_J;        // BLOCK_J
    float* sh_sumr = sh_lj   + BLOCK_J;        // BLOCK_J

    /* ---- accumulator for grad_S[k,:] -------------------------------- */
    float grad_acc[REG_CAP] = {0.0f};

    // numerical stability constants
    constexpr float EXP_CLIP = 80.0f;   // safe range for expf
    constexpr float DENOM_EPS = 1e-6f;  // prevent division by zero

    /* ---- iterate over J in tiles ------------------------------------ */
    for (int jBase = 0; jBase < N; jBase += BLOCK_J)
    {
        /* cooperative load by all (k,i) threads */
        /* NOTE: BLOCK_J may exceed BLOCK_I, so we loop to cover all rows */
        for (int ld = threadIdx.y; ld < BLOCK_J && (jBase + ld) < N; ld += BLOCK_I) {
            const int jGlob = jBase + ld;
            #pragma unroll
            for (int d = threadIdx.x; d < D; d += BLOCK_K){
                sh_R   [ld*D + d] = Rbh  [jGlob*D + d];
                sh_Vr1 [ld*D + d] = Vr1bh[jGlob*D + d];
                sh_Vr2 [ld*D + d] = Vr2bh[jGlob*D + d];
                sh_dYj [ld*D + d] = gYbh [jGlob*D + d];
            }
            if (threadIdx.x == 0){
                sh_mj  [ld] = mjBH  [jGlob];
                sh_lj  [ld] = ljBH  [jGlob];
                sh_sumr[ld] = sum_rBH[jGlob];
            }
        }
        __syncthreads();

        /* ---- loop inside loaded J-tile ------------------------------ */
        for (int jOff = 0; jOff < BLOCK_J && (jBase + jOff) < N; ++jOff){
            const float* Rj   = sh_R   + jOff*D;
            const float* Vr1j = sh_Vr1 + jOff*D;
            const float* Vr2j = sh_Vr2 + jOff*D;
            const float* dYj  = sh_dYj + jOff*D;
            const float mj    = sh_mj  [jOff];
            const float lj    = sh_lj  [jOff];
            const float sumRj = sh_sumr[jOff];

            /* --- logits & softmax numerators ------------------------ */
            float dot = 0.f;
            #pragma unroll
            for (int d=0; d<D; ++d)
                dot += Qi[d] * Rj[d] * Sk[d];
            float logits = dot * scale;

            float Aq = __expf(fminf(logits - mi, EXP_CLIP)) / fmaxf(li, DENOM_EPS);
            float Ar = __expf(fminf(logits - mj, EXP_CLIP)) / fmaxf(lj, DENOM_EPS);
            float As = __expf(fminf(logits - mk, EXP_CLIP)) / fmaxf(lk, DENOM_EPS);

            /* --- grad_Aq ------------------------------------------- */
            float gAq = 0.f;
            #pragma unroll
            for (int d=0; d<D; ++d)
                gAq += dYi[d] * Vr1j[d] * Vs1k[d];

            float tmp = 0.f;
            #pragma unroll
            for (int d=0; d<D; ++d)
                tmp += dYj[d] * Vq2i[d] * Vs2k[d];
            gAq += tmp * As;

            tmp = 0.f;
            #pragma unroll
            for (int d=0; d<D; ++d)
                tmp += dYk[d] * Vq2i[d] * Vr2j[d];
            gAq += tmp * Ar;

            /* --- grad_Ar ------------------------------------------- */
            float gAr = 0.f;
            #pragma unroll
            for (int d=0; d<D; ++d)
                gAr += dYj[d] * Vq1i[d] * Vs1k[d];

            tmp = 0.f;
            #pragma unroll
            for (int d=0; d<D; ++d)
                tmp += dYi[d] * Vr2j[d] * Vs2k[d];
            gAr += tmp * As;

            tmp = 0.f;
            #pragma unroll
            for (int d=0; d<D; ++d)
                tmp += dYk[d] * Vq2i[d] * Vr2j[d];
            gAr += tmp * Aq;

            /* --- grad_As ------------------------------------------- */
            float gAs = 0.f;
            #pragma unroll
            for (int d=0; d<D; ++d)
                gAs += dYk[d] * Vq1i[d] * Vr1j[d];

            tmp = 0.f;
            #pragma unroll
            for (int d=0; d<D; ++d)
                tmp += dYi[d] * Vr2j[d] * Vs2k[d];
            gAs += tmp * Ar;

            tmp = 0.f;
            #pragma unroll
            for (int d=0; d<D; ++d)
                tmp += dYj[d] * Vq2i[d] * Vs2k[d];
            gAs += tmp * Aq;

            /* --- Jacobian-corrected grad_A ------------------------- */
            float grad_A = (gAq -  sumQi ) * Aq
                         + (gAr -  sumRj ) * Ar
                         + (gAs -  sumSk ) * As;

            /* --- accumulate contribution to grad_S[k,:] ------------ */
            #pragma unroll
            for (int d=0; d<D; ++d)
                grad_acc[d] += grad_A * Qi[d] * Rj[d];
        }
        __syncthreads();
    }

    /* ---- atomic add to global grad_S ------------------------------- */
    if (valid) {
        #pragma unroll
        for (int d=0; d<D; ++d)
            atomicAdd(&gSbh[k0*D + d], scale * grad_acc[d]);
    }
}


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
              double dropout_rate) {
                
  // ============================================================================
  // 1. ENSURE ALL TENSORS ARE CONTIGUOUS
  // ============================================================================
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

  // ============================================================================
  // 2. EXTRACT DIMENSIONS AND CONSTANTS
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
  // 3. ALLOCATE TENSORS
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
  auto m_i = torch::empty({B, H, N}, optionsBH);
  auto l_i = torch::empty({B, H, N}, optionsBH);
  auto m_j = torch::empty({B, H, N}, optionsBH);
  auto l_j = torch::empty({B, H, N}, optionsBH);
  auto m_k = torch::empty({B, H, N}, optionsBH);
  auto l_k = torch::empty({B, H, N}, optionsBH);
  auto sum_q = torch::zeros({B, H, N}, optionsBH);
  auto sum_r = torch::zeros({B, H, N}, optionsBH);
  auto sum_s = torch::zeros({B, H, N}, optionsBH);

  // ============================================================================
  // 4. PRECOMPUTE SOFTMAX STATISTICS   
  // ============================================================================
  constexpr int block_threads = 256;

  // --- Aq (i-centric) stats ---
  {
    const size_t shmem_bytes =
        (D + TILE_J * D + TILE_K * D + TILE_J * TILE_K + block_threads) * sizeof(float);

    dim3 blocks(I, B * H);
    dim3 threads(block_threads);

    Aq_tiled_softmax<<<blocks, threads, shmem_bytes, at::cuda::getCurrentCUDAStream()>>>(
        Q.data_ptr<float>(),
        R.data_ptr<float>(),
        S.data_ptr<float>(),
        m_i.data_ptr<float>(),
        l_i.data_ptr<float>(),
        B, H, N, J, K, D, scale);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "CUDA error in Aq_tiled_softmax: %s\n", cudaGetErrorString(err));
    }
  }

  // --- Ar (j-centric) stats ---
  {
    const size_t shmem_bytes =
        (D                                  /* r_vec */
         + TILE_I * D + TILE_K * D           /* q_tile + s_tile */
         + TILE_I * TILE_K                   /* p_tile */
         + block_threads) * sizeof(float);   /* red_buf */

    dim3 blocks(J, B * H);
    dim3 threads(block_threads);

    Ar_tiled_softmax<<<blocks, threads, shmem_bytes, at::cuda::getCurrentCUDAStream()>>>(
        Q.data_ptr<float>(),
        R.data_ptr<float>(),
        S.data_ptr<float>(),
        m_j.data_ptr<float>(),
        l_j.data_ptr<float>(),
        B, H, N, J, K, D, scale);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "CUDA error in Ar_tiled_softmax: %s\n", cudaGetErrorString(err));
    }
  }

  // --- As (k-centric) stats ---
  {
    const size_t shmem_bytes =
        (D                                  /* s_vec */
         + TILE_I * D + TILE_J * D           /* q_tile + r_tile */
         + TILE_I * TILE_J                   /* p_tile */
         + block_threads) * sizeof(float);   /* red_buf */

    dim3 blocks(K, B * H);
    dim3 threads(block_threads);

    As_tiled_softmax<<<blocks, threads, shmem_bytes, at::cuda::getCurrentCUDAStream()>>>(
        Q.data_ptr<float>(),
        R.data_ptr<float>(),
        S.data_ptr<float>(),
        m_k.data_ptr<float>(),
        l_k.data_ptr<float>(),
        B, H, N, J, K, D, scale);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "CUDA error in As_tiled_softmax: %s\n", cudaGetErrorString(err));
    }
  }

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

    grad_gather_Vq_kernel<<<grid_dim, block_dim, shmem_bytes, at::cuda::getCurrentCUDAStream()>>>(
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

    grad_gather_Vr_kernel<<<grid_dim, block_dim, shmem_bytes, at::cuda::getCurrentCUDAStream()>>>(
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

    grad_gather_Vs_kernel<<<grid_dim, block_dim, shmem_bytes, at::cuda::getCurrentCUDAStream()>>>(
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

    grad_scatter_Vq_kernel<<<grid_dim, block_dim, shmem_bytes, at::cuda::getCurrentCUDAStream()>>>(
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
    AT_CUDA_CHECK(cudaDeviceSynchronize());
    TORCH_CHECK(
        grad_Vq_2.isfinite().all().item<bool>(),
        "grad_Vq_2 contains non-finite values after grad_scatter_Vq_kernel");
  }

  {
    // --- grad_Vr_2 ---
    constexpr int TI = T_I;
    constexpr int TK = T_K;

    dim3 block_dim(TI, TK);
    dim3 grid_dim((N + TI - 1) / TI, (N + TK - 1) / TK, B * H);
    size_t shmem_bytes =
        T_J * D * 3 * sizeof(float) + T_J * 2 * sizeof(float);

    grad_scatter_Vr_kernel<<<grid_dim, block_dim, shmem_bytes, at::cuda::getCurrentCUDAStream()>>>(
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

    grad_scatter_Vs_kernel<<<grid_dim, block_dim, shmem_bytes, at::cuda::getCurrentCUDAStream()>>>(
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
  // 7. PRECOMPUTE JACOBIAN CORRECTIONS for grad_Q, R, S
  // ============================================================================
  {
    // --- Correction sums shared by grad_Q/grad_R/grad_S ---
    constexpr int Bq = 8;
    constexpr int Br = 8;
    constexpr int Bk = 8;

    dim3 grid(ceilDiv(N, Bq), ceilDiv(N, Br), ceilDiv(N, Bk));
    dim3 threads(Bq, Br, Bk);

    const size_t shmem_bytes =
        (Bq * D + Br * D + Bk * D +   // Q, R, S
         Bq * D + Bq * D +            // Vq1, Vq2
         Br * D + Br * D +            // Vr1, Vr2
         Bk * D + Bk * D +            // Vs1, Vs2
         Bq * D + Br * D + Bk * D)    // dY
        * sizeof(float);

    for (int b = 0; b < B; ++b) {
      for (int h = 0; h < H; ++h) {
        precompute_jacobian_corrections_kernel<Bq, Br, Bk><<<grid, threads, shmem_bytes, at::cuda::getCurrentCUDAStream()>>>(
            Q.select(0, b).select(0, h).data_ptr<float>(),
            R.select(0, b).select(0, h).data_ptr<float>(),
            S.select(0, b).select(0, h).data_ptr<float>(),
            Vq_1.select(0, b).select(0, h).data_ptr<float>(),
            Vq_2.select(0, b).select(0, h).data_ptr<float>(),
            Vr_1.select(0, b).select(0, h).data_ptr<float>(),
            Vr_2.select(0, b).select(0, h).data_ptr<float>(),
            Vs_1.select(0, b).select(0, h).data_ptr<float>(),
            Vs_2.select(0, b).select(0, h).data_ptr<float>(),
            grad_output.select(0, b).select(0, h).data_ptr<float>(),
            m_i.select(0, b).select(0, h).data_ptr<float>(),
            l_i.select(0, b).select(0, h).data_ptr<float>(),
            m_j.select(0, b).select(0, h).data_ptr<float>(),
            l_j.select(0, b).select(0, h).data_ptr<float>(),
            m_k.select(0, b).select(0, h).data_ptr<float>(),
            l_k.select(0, b).select(0, h).data_ptr<float>(),
            sum_q.select(0, b).select(0, h).data_ptr<float>(),
            sum_r.select(0, b).select(0, h).data_ptr<float>(),
            sum_s.select(0, b).select(0, h).data_ptr<float>(),
            N, D, scale);
      }
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "CUDA error in precompute_jacobian_corrections launch: %s\n", cudaGetErrorString(err));
    }
  }

  // ===========================================================================
  // 8. COMPUTE grad_Q, R, S
  // ============================================================================   

  {
    // --- grad_Q accumulation ---
    constexpr int tileI = TILE_I;  // reuse existing macro defaults (8)
    constexpr int tileK = TILE_K;
    constexpr int tileJ = 16;      // independent knob for J streaming

    TORCH_CHECK(D <= MAX_D_REG,
                "grad_Q_kernel requires D <= ",
                MAX_D_REG,
                ", but got D = ",
                D);

    dim3 block_dim(tileI, tileK);
    dim3 grid_dim((N + tileI - 1) / tileI,
                  (N + tileK - 1) / tileK,
                  B * H);

    const size_t shmem_bytes =
        4 * tileJ * D * sizeof(float) +  // R, Vr1, Vr2, gYj
        3 * tileJ * sizeof(float);       // mj, lj, sum_r

    grad_Q_kernel<tileI, tileJ, tileK>
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
            N, D, scale);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "CUDA error in grad_Q_kernel launch: %s\n", cudaGetErrorString(err));
    }

    AT_CUDA_CHECK(cudaDeviceSynchronize());
    TORCH_CHECK(
        grad_Vq_2.isfinite().all().item<bool>(),
        "grad_Vq_2 contains non-finite values after grad_Q_kernel");
  }

  {
    // --- grad_R accumulation ---
    constexpr int tileJ = TILE_J;  // reuse macro defaults for target axis
    constexpr int tileK = TILE_K;
    constexpr int tileI = 16;      // independent knob for I streaming

    TORCH_CHECK(D <= MAX_D_REG,
                "grad_R_kernel requires D <= ",
                MAX_D_REG,
                ", but got D = ",
                D);

    dim3 block_dim(tileJ, tileK);
    dim3 grid_dim((N + tileJ - 1) / tileJ,
                  (N + tileK - 1) / tileK,
                  B * H);

    const size_t shmem_bytes =
        (4 * tileI * D + 3 * tileI) * sizeof(float);  // Q, Vq1, Vq2, dY + stats

    grad_R_kernel<tileJ, tileI, tileK>
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

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "CUDA error in grad_R_kernel launch: %s\n", cudaGetErrorString(err));
    }

    AT_CUDA_CHECK(cudaDeviceSynchronize());
    TORCH_CHECK(
        grad_Vq_2.isfinite().all().item<bool>(),
        "grad_Vq_2 contains non-finite values after grad_R_kernel");
  }

  {
    // --- grad_S accumulation ---
    constexpr int tileK = TILE_K;
    constexpr int tileI = TILE_I;
    constexpr int tileJ = 16;      // stream over J independently

    TORCH_CHECK(D <= MAX_D_REG,
                "grad_S_kernel requires D <= ",
                MAX_D_REG,
                ", but got D = ",
                D);

    dim3 block_dim(tileK, tileI);
    dim3 grid_dim((N + tileK - 1) / tileK,
                  (N + tileI - 1) / tileI,
                  B * H);

    const size_t shmem_bytes =
        4 * tileJ * D * sizeof(float) + 3 * tileJ * sizeof(float);

    grad_S_kernel<tileK, tileJ, tileI>
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
            grad_S.data_ptr<float>(),
            N, D, scale);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "CUDA error in grad_S_kernel launch: %s\n", cudaGetErrorString(err));
    }

    AT_CUDA_CHECK(cudaDeviceSynchronize());
    TORCH_CHECK(
        grad_Vq_2.isfinite().all().item<bool>(),
        "grad_Vq_2 contains non-finite values after grad_S_kernel");
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

