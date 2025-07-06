#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>      
#include <cuda.h>
#include <cuda_runtime.h>
#include "../cpp/manual_att3ntion.h"


// Forward declarations
torch::Tensor compute_A_slice_cuda_wrapper(
    const torch::Tensor& Q_slice_gpu,
    const torch::Tensor& R_slice_gpu,
    const torch::Tensor& S_slice_gpu,
    float scale);

torch::Tensor compute_Aq_slice_cuda_wrapper(
    const torch::Tensor& A_slice_gpu);

torch::Tensor compute_Ar_slice_cuda_wrapper(
    const torch::Tensor& A_slice_gpu);

torch::Tensor compute_As_slice_cuda_wrapper(
    const torch::Tensor& A_slice_gpu);
//Kernel Declarations
__global__ void Yq_gather_kernel(
    const float* __restrict__ Q, 
    const float* __restrict__ R, 
    const float* __restrict__ S,
    const float* __restrict__ V1, 
    const float* __restrict__ V2,
    float*       __restrict__ Y,
    int B, int H, int I, int J, int K, int D,
    float scale);

__global__ void Yr_gather_kernel(
    const float* __restrict__ R_query, 
    const float* __restrict__ Q,
    const float* __restrict__ S,
    const float* __restrict__ V1,      
    const float* __restrict__ V2,      
    float*       __restrict__ Y,       
    int B, int H, int I, int J, int K, int D,
    float scale);

__global__ void Ys_gather_kernel(
    const float* __restrict__ S_query, 
    const float* __restrict__ Q,
    const float* __restrict__ R,
    const float* __restrict__ V1,      
    const float* __restrict__ V2,      
    float*       __restrict__ Y,      
    int B, int H, int I, int J, int K, int D,
    float scale);

__global__ void Yq_scatter_kernel_optimized(
    const float* __restrict__ Ar_slice, // [I, J, K]
    const float* __restrict__ As_slice, // [I, J, K]
    const float* __restrict__ Vr_2_slice, // [J, D]
    const float* __restrict__ Vs_2_slice, // [K, D]
    float*       __restrict__ Y_q__slice_out, // [I, D]
    int I, int J, int K, int D);

__global__ void Yr_scatter_kernel_optimized(
    const float* __restrict__ Aq_slice, // [I, J, K]
    const float* __restrict__ As_slice, // [I, J, K]
    const float* __restrict__ Vq_2_slice, // [I, D]
    const float* __restrict__ Vs_2_slice, // [K, D]
    float*       __restrict__ Y_r__slice_out, // [J, D]
    int I, int J, int K, int D);

__global__ void Ys_scatter_kernel_optimized(
    const float* __restrict__ Aq_slice, // [I, J, K]
    const float* __restrict__ Ar_slice, // [I, J, K]
    const float* __restrict__ Vq_2_slice, // [I, D]
    const float* __restrict__ Vr_2_slice, // [J, D]
    float*       __restrict__ Y_s__slice_out, // [K, D]
    int I, int J, int K, int D);

__global__ void gather_grad_Vq1_kernel_optimized(
    const float* __restrict__ gradY_slice,
    const float* __restrict__ Vr_1_slice,
    const float* __restrict__ Vs_1_slice,
    const float* __restrict__ Ar_slice,
    const float* __restrict__ As_slice,
    float*       __restrict__ gradVq1_slice_out,
    int I, int J, int K, int D, int N_grad);

__global__ void gather_grad_Vr1_kernel_optimized(
    const float* __restrict__ gradY_slice,
    const float* __restrict__ Vq_1_slice,
    const float* __restrict__ Vs_1_slice,
    const float* __restrict__ Aq_slice,
    const float* __restrict__ As_slice,
    float*       __restrict__ gradVr1_slice_out,
    int I, int J, int K, int D, int N_grad);

__global__ void gather_grad_Vs1_kernel_optimized(
    const float* __restrict__ gradY_slice,
    const float* __restrict__ Vq_1_slice,
    const float* __restrict__ Vr_1_slice,
    const float* __restrict__ Aq_slice,
    const float* __restrict__ Ar_slice,
    float*       __restrict__ gradVs1_slice_out,
    int I, int J, int K, int D, int N_grad);
__global__ void scatter_grad_Vq2_kernel_optimized(
    const float* __restrict__ gradY_slice,
    const float* __restrict__ Aq_slice,
    const float* __restrict__ Ar_slice,
    const float* __restrict__ As_slice,
    const float* __restrict__ Vr_2_slice,
    const float* __restrict__ Vs_2_slice,
    float*       __restrict__ gradVq2_slice_out,
    int I, int J, int K, int D, int N_grad);

__global__ void scatter_grad_Vr2_kernel_optimized(
    const float* __restrict__ gradY_slice,
    const float* __restrict__ Aq_slice,
    const float* __restrict__ Ar_slice,
    const float* __restrict__ As_slice,
    const float* __restrict__ Vq_2_slice,
    const float* __restrict__ Vs_2_slice,
    float*       __restrict__ gradVr2_slice_out,
    int I, int J, int K, int D, int N_grad);

__global__ void scatter_grad_Vs2_kernel_optimized(
    const float* __restrict__ gradY_slice,
    const float* __restrict__ Aq_slice,
    const float* __restrict__ Ar_slice,
    const float* __restrict__ As_slice,
    const float* __restrict__ Vq_2_slice,
    const float* __restrict__ Vr_2_slice,
    float*       __restrict__ gradVs2_slice_out,
    int I, int J, int K, int D, int N_grad);



__global__ void compute_A_slice_kernel(
    const float* __restrict__ Q_slice_global, 
    const float* __restrict__ R_slice_global, 
    const float* __restrict__ S_slice_global, 
    float*       __restrict__ A_out_global,  
    int I, int J, int K, int D,
    float scale);
__global__ void compute_Aq_slice_kernel(
    const float* __restrict__ A_slice_global, // Input A_slice [I,J,K] on GPU
    float*       __restrict__ Aq_out_global,  // Output Aq_slice [I,J,K] on GPU
    int I, int J, int K);

__global__ void compute_Ar_slice_kernel(
    const float* __restrict__ A_slice_global, // Input A_slice [I,J,K] on GPU
    float*       __restrict__ Ar_out_global,  // Output Ar_slice [I,J,K] on GPU
    int I, int J, int K);

__global__ void compute_As_slice_kernel(
    const float* __restrict__ A_slice_global, // Input A_slice [I,J,K] on GPU
    float*       __restrict__ As_out_global,  // Output As_slice [I,J,K] on GPU
    int I, int J, int K);
__global__ void apply_softmax_backward_kernel(
    // Inputs
    const float* __restrict__ grad_Aq_slice, // [I, J, K], from interim_grads_kernel
    const float* __restrict__ grad_Ar_slice, // [I, J, K], from interim_grads_kernel
    const float* __restrict__ grad_As_slice, // [I, J, K], from interim_grads_kernel
    const float* __restrict__ Aq_slice,      // [I, J, K], from softmax kernels
    const float* __restrict__ Ar_slice,      // [I, J, K], from softmax kernels
    const float* __restrict__ As_slice,      // [I, J, K], from softmax kernels
    // Output
    float* __restrict__ grad_A_slice_out,    // [I, J, K], final grad_A for this slice
    // Dimensions
    int I, int J, int K);

__global__ void grad_R_kernel(
    const float* __restrict__ grad_A, // Shape [B, H, I, J, K]
    const float* __restrict__ Q,      // Shape [B, H, I, D]
    const float* __restrict__ S,      // Shape [B, H, K, D]
    float*       __restrict__ grad_R, // Shape [B, H, J, D] - Output
    const int B, const int H, const int I, const int J, const int K, const int D,
    const float scale);
__global__ void grad_S_kernel(
    const float* __restrict__ grad_A, // Shape [B, H, I, J, K]
    const float* __restrict__ Q,      // Shape [B, H, I, D]
    const float* __restrict__ R,      // Shape [B, H, J, D]
    float*       __restrict__ grad_S, // Shape [B, H, K, D] - Output
    const int B, const int H, const int I, const int J, const int K, const int D,
    const float scale);




// -- Forward Pass --
__global__ void Yq_gather_kernel(
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

// -- Tiled Yq_gather_kernel --
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define TILE_J 16
#define TILE_K 16

__device__ inline float block_reduce_sum(float v, float* buf)
{
    auto blk = cg::this_thread_block();
    buf[blk.thread_rank()] = v;

    cg::sync(blk);
    for (int s = blk.size() / 2; s > 0; s >>= 1) {
        if (blk.thread_rank() < s) buf[blk.thread_rank()] += buf[blk.thread_rank() + s];
        cg::sync(blk);
    }
    return buf[0];
}

extern "C" __global__
void Yq_gather_kernel_tiled(
        const float* __restrict__ Q,
        const float* __restrict__ R,
        const float* __restrict__ S,
        const float* __restrict__ V1,
        const float* __restrict__ V2,
        float*       __restrict__ Y,
        int B,int H,int I,int J,int K,int D,float scale)
{
    /* -------- block ↔ (b,h,i) mapping -------- */
    const int b = blockIdx.z;
    const int h = blockIdx.y;
    const int i = blockIdx.x;
    auto blk   = cg::this_thread_block();     // 32 threads

    /* -------- shared-memory scratch -------- */
    extern __shared__ float smem[];
    float* q_vec   = smem;                             // D
    float* r_tile  = q_vec   + D;                      // TILE_J * D
    float* s_tile  = r_tile  + TILE_J * D;             // TILE_K * D
    float* v1_tile = s_tile  + TILE_K * D;             // TILE_J * D
    float* v2_tile = v1_tile + TILE_J * D;             // TILE_K * D
    float* red_buf = v2_tile + TILE_K * D;             // 32

    /* -------- preload Q[b,h,i,:] -------- */
    const int64_t q_off = (((int64_t)b * H + h) * I + i) * D;
    for (int d = threadIdx.x; d < D; d += blk.size())
        q_vec[d] = Q[q_off + d];
    cg::sync(blk);

    /* -------- pass 1 : max(dot) -------- */
    float max_dot = -1e30f;
    for (int jb = 0; jb < J; jb += TILE_J)
    for (int kb = 0; kb < K; kb += TILE_K)
    {
        /* load R/S tiles */
        for (int jt = 0; jt < TILE_J; ++jt) {
            int j = jb + jt;
            if (j < J) {
                const int64_t r_off = (((int64_t)b * H + h) * J + j) * D;
                for (int d = threadIdx.x; d < D; d += blk.size())
                    r_tile[jt*D + d] = R[r_off + d];
            }
        }
        for (int kt = 0; kt < TILE_K; ++kt) {
            int k = kb + kt;
            if (k < K) {
                const int64_t s_off = (((int64_t)b * H + h) * K + k) * D;
                for (int d = threadIdx.x; d < D; d += blk.size())
                    s_tile[kt*D + d] = S[s_off + d];
            }
        }
        cg::sync(blk);

        /* compute dots inside the 16×16 tile */
        for (int jt = 0; jt < TILE_J && jb+jt < J; ++jt)
        for (int kt = 0; kt < TILE_K && kb+kt < K; ++kt)
        {
            float part = 0.f;
            for (int d = threadIdx.x; d < D; d += blk.size())
                part += q_vec[d] * r_tile[jt*D + d] * s_tile[kt*D + d];
            float dot = block_reduce_sum(part, red_buf);
            if (threadIdx.x == 0) max_dot = fmaxf(max_dot, dot * scale);
            cg::sync(blk);
        }
    }
    /* broadcast max_dot */
    if (threadIdx.x == 0) red_buf[0] = max_dot;
    cg::sync(blk);
    max_dot = red_buf[0];

    /* -------- pass 2 : sum exp -------- */
    float sum_exp = 0.f;
    for (int jb = 0; jb < J; jb += TILE_J)
    for (int kb = 0; kb < K; kb += TILE_K)
    {
        /* reload R/S tiles (same loop as above) */
        for (int jt = 0; jt < TILE_J; ++jt) {
            int j = jb + jt;
            if (j < J) {
                const int64_t r_off = (((int64_t)b * H + h) * J + j) * D;
                for (int d = threadIdx.x; d < D; d += blk.size())
                    r_tile[jt*D + d] = R[r_off + d];
            }
        }
        for (int kt = 0; kt < TILE_K; ++kt) {
            int k = kb + kt;
            if (k < K) {
                const int64_t s_off = (((int64_t)b * H + h) * K + k) * D;
                for (int d = threadIdx.x; d < D; d += blk.size())
                    s_tile[kt*D + d] = S[s_off + d];
            }
        }
        cg::sync(blk);

        for (int jt = 0; jt < TILE_J && jb+jt < J; ++jt)
        for (int kt = 0; kt < TILE_K && kb+kt < K; ++kt)
        {
            float part = 0.f;
            for (int d = threadIdx.x; d < D; d += blk.size())
                part += q_vec[d] * r_tile[jt*D + d] * s_tile[kt*D + d];
            float dot = block_reduce_sum(part, red_buf);
            if (threadIdx.x == 0) sum_exp += expf(dot*scale - max_dot);
            cg::sync(blk);
        }
    }
    if (threadIdx.x == 0) red_buf[0] = sum_exp;
    cg::sync(blk);
    sum_exp = red_buf[0];
    const float inv_denom = 1.f / sum_exp;

    /* -------- pass 3 : final accumulation -------- */
    float y_val = 0.f;
    for (int jb = 0; jb < J; jb += TILE_J)
    for (int kb = 0; kb < K; kb += TILE_K)
    {
        /* load R/S/V tiles */
        for (int jt = 0; jt < TILE_J; ++jt) {
            int j = jb + jt;
            if (j < J) {
                const int64_t base = (((int64_t)b * H + h) * J + j) * D;
                for (int d = threadIdx.x; d < D; d += blk.size()) {
                    r_tile [jt*D + d] = R [base + d];
                    v1_tile[jt*D + d] = V1[base + d];
                }
            }
        }
        for (int kt = 0; kt < TILE_K; ++kt) {
            int k = kb + kt;
            if (k < K) {
                const int64_t base = (((int64_t)b * H + h) * K + k) * D;
                for (int d = threadIdx.x; d < D; d += blk.size()) {
                    s_tile [kt*D + d] = S [base + d];
                    v2_tile[kt*D + d] = V2[base + d];
                }
            }
        }
        cg::sync(blk);

        for (int jt = 0; jt < TILE_J && jb+jt < J; ++jt)
        for (int kt = 0; kt < TILE_K && kb+kt < K; ++kt)
        {
            float part = 0.f;
            for (int d = threadIdx.x; d < D; d += blk.size())
                part += q_vec[d] * r_tile[jt*D + d] * s_tile[kt*D + d];
            float dot = block_reduce_sum(part, red_buf);
            if (threadIdx.x == 0) red_buf[1] = dot;
            cg::sync(blk);
            dot = red_buf[1];

            /* accumulate */
            for (int d = threadIdx.x; d < D; d += blk.size())
                y_val += (expf(dot*scale - max_dot) * inv_denom) *
                         v1_tile[jt*D + d] *
                         v2_tile[kt*D + d];
            cg::sync(blk);
        }
    }

    /* -------- write result -------- */
    for (int d = threadIdx.x; d < D; d += blk.size())
        Y[q_off + d] = y_val;
}

// ... existing code ...
#define TILE_J 16
#define TILE_K 16

#define TILE_I 16

__device__ inline float block_reduce_sum(float v, float* buf)
{
// ... existing code ...
extern "C" __global__
void Yq_gather_kernel_tiled(
        const float* __restrict__ Q,
// ... existing code ...
    for (int d = threadIdx.x; d < D; d += blk.size())
        Y[q_off + d] = y_val;
}

extern "C" __global__
void Yr_gather_kernel_tiled(
        const float* __restrict__ R_query,
        const float* __restrict__ Q,
        const float* __restrict__ S,
        const float* __restrict__ V1, // Vq_1
        const float* __restrict__ V2, // Vs_1
        float*       __restrict__ Y,  // Y_r
        int B,int H,int I,int J,int K,int D,float scale)
{
    /* -------- block ↔ (b,h,j) mapping -------- */
    const int b = blockIdx.z;
    const int h = blockIdx.y;
    const int j = blockIdx.x;
    auto blk   = cg::this_thread_block();

    /* -------- shared-memory scratch -------- */
    extern __shared__ float smem[];
    float* r_query_vec = smem;
    float* q_tile  = r_query_vec + D;
    float* s_tile  = q_tile  + TILE_I * D;
    float* v1_tile = s_tile  + TILE_K * D;
    float* v2_tile = v1_tile + TILE_I * D;
    float* red_buf = v2_tile + TILE_K * D;

    /* -------- preload R_query[b,h,j,:] -------- */
    const int64_t r_query_off = (((int64_t)b * H + h) * J + j) * D;
    for (int d = threadIdx.x; d < D; d += blk.size())
        r_query_vec[d] = R_query[r_query_off + d];
    cg::sync(blk);

    /* -------- pass 1 : max(dot) -------- */
    float max_dot = -1e30f;
    for (int ib = 0; ib < I; ib += TILE_I)
    for (int kb = 0; kb < K; kb += TILE_K)
    {
        /* load Q/S tiles */
        for (int it = 0; it < TILE_I; ++it) {
            int i_ = ib + it;
            if (i_ < I) {
                const int64_t q_off = (((int64_t)b * H + h) * I + i_) * D;
                for (int d = threadIdx.x; d < D; d += blk.size())
                    q_tile[it*D + d] = Q[q_off + d];
            }
        }
        for (int kt = 0; kt < TILE_K; ++kt) {
            int k = kb + kt;
            if (k < K) {
                const int64_t s_off = (((int64_t)b * H + h) * K + k) * D;
                for (int d = threadIdx.x; d < D; d += blk.size())
                    s_tile[kt*D + d] = S[s_off + d];
            }
        }
        cg::sync(blk);

        /* compute dots inside the tile */
        for (int it = 0; it < TILE_I && ib+it < I; ++it)
        for (int kt = 0; kt < TILE_K && kb+kt < K; ++kt)
        {
            float part = 0.f;
            for (int d = threadIdx.x; d < D; d += blk.size())
                part += r_query_vec[d] * q_tile[it*D + d] * s_tile[kt*D + d];
            float dot = block_reduce_sum(part, red_buf);
            if (threadIdx.x == 0) max_dot = fmaxf(max_dot, dot * scale);
            cg::sync(blk);
        }
    }
    if (threadIdx.x == 0) red_buf[0] = max_dot;
    cg::sync(blk);
    max_dot = red_buf[0];

    /* -------- pass 2 : sum exp -------- */
    float sum_exp = 0.f;
    for (int ib = 0; ib < I; ib += TILE_I)
    for (int kb = 0; kb < K; kb += TILE_K)
    {
        /* reload Q/S tiles */
        for (int it = 0; it < TILE_I; ++it) {
            int i_ = ib + it;
            if (i_ < I) {
                const int64_t q_off = (((int64_t)b * H + h) * I + i_) * D;
                for (int d = threadIdx.x; d < D; d += blk.size())
                    q_tile[it*D + d] = Q[q_off + d];
            }
        }
        for (int kt = 0; kt < TILE_K; ++kt) {
            int k = kb + kt;
            if (k < K) {
                const int64_t s_off = (((int64_t)b * H + h) * K + k) * D;
                for (int d = threadIdx.x; d < D; d += blk.size())
                    s_tile[kt*D + d] = S[s_off + d];
            }
        }
        cg::sync(blk);

        for (int it = 0; it < TILE_I && ib+it < I; ++it)
        for (int kt = 0; kt < TILE_K && kb+kt < K; ++kt)
        {
            float part = 0.f;
            for (int d = threadIdx.x; d < D; d += blk.size())
                part += r_query_vec[d] * q_tile[it*D + d] * s_tile[kt*D + d];
            float dot = block_reduce_sum(part, red_buf);
            if (threadIdx.x == 0) sum_exp += expf(dot*scale - max_dot);
            cg::sync(blk);
        }
    }
    if (threadIdx.x == 0) red_buf[0] = sum_exp;
    cg::sync(blk);
    sum_exp = red_buf[0];
    const float inv_denom = 1.f / sum_exp;

    /* -------- pass 3 : final accumulation -------- */
    float y_val = 0.f;
    for (int ib = 0; ib < I; ib += TILE_I)
    for (int kb = 0; kb < K; kb += TILE_K)
    {
        /* load Q/S/V tiles */
        for (int it = 0; it < TILE_I; ++it) {
            int i_ = ib + it;
            if (i_ < I) {
                const int64_t base = (((int64_t)b * H + h) * I + i_) * D;
                for (int d = threadIdx.x; d < D; d += blk.size()) {
                    q_tile[it*D + d] = Q[base + d];
                    v1_tile[it*D + d] = V1[base + d];
                }
            }
        }
        for (int kt = 0; kt < TILE_K; ++kt) {
            int k = kb + kt;
            if (k < K) {
                const int64_t base = (((int64_t)b * H + h) * K + k) * D;
                for (int d = threadIdx.x; d < D; d += blk.size()) {
                    s_tile[kt*D + d] = S[base + d];
                    v2_tile[kt*D + d] = V2[base + d];
                }
            }
        }
        cg::sync(blk);

        for (int it = 0; it < TILE_I && ib+it < I; ++it)
        for (int kt = 0; kt < TILE_K && kb+kt < K; ++kt)
        {
            float part = 0.f;
            for (int d = threadIdx.x; d < D; d += blk.size())
                part += r_query_vec[d] * q_tile[it*D + d] * s_tile[kt*D + d];
            float dot = block_reduce_sum(part, red_buf);
            if (threadIdx.x == 0) red_buf[1] = dot;
            cg::sync(blk);
            dot = red_buf[1];

            /* accumulate */
            for (int d = threadIdx.x; d < D; d += blk.size())
                y_val += (expf(dot*scale - max_dot) * inv_denom) *
                         v1_tile[it*D + d] *
                         v2_tile[kt*D + d];
            cg::sync(blk);
        }
    }

    /* -------- write result -------- */
    for (int d = threadIdx.x; d < D; d += blk.size())
        Y[r_query_off + d] = y_val;
}

extern "C" __global__
void Ys_gather_kernel_tiled(
        const float* __restrict__ S_query,
        const float* __restrict__ Q,
        const float* __restrict__ R,
        const float* __restrict__ V1, // Vq_1
        const float* __restrict__ V2, // Vr_1
        float*       __restrict__ Y,  // Y_s
        int B,int H,int I,int J,int K,int D,float scale)
{
    /* -------- block ↔ (b,h,k) mapping -------- */
    const int b = blockIdx.z;
    const int h = blockIdx.y;
    const int k = blockIdx.x;
    auto blk   = cg::this_thread_block();

    /* -------- shared-memory scratch -------- */
    extern __shared__ float smem[];
    float* s_query_vec = smem;
    float* q_tile  = s_query_vec + D;
    float* r_tile  = q_tile  + TILE_I * D;
    float* v1_tile = r_tile  + TILE_J * D;
    float* v2_tile = v1_tile + TILE_I * D;
    float* red_buf = v2_tile + TILE_J * D;

    /* -------- preload S_query[b,h,k,:] -------- */
    const int64_t s_query_off = (((int64_t)b * H + h) * K + k) * D;
    for (int d = threadIdx.x; d < D; d += blk.size())
        s_query_vec[d] = S_query[s_query_off + d];
    cg::sync(blk);

    /* -------- pass 1 : max(dot) -------- */
    float max_dot = -1e30f;
    for (int ib = 0; ib < I; ib += TILE_I)
    for (int jb = 0; jb < J; jb += TILE_J)
    {
        /* load Q/R tiles */
        for (int it = 0; it < TILE_I; ++it) {
            int i_ = ib + it;
            if (i_ < I) {
                const int64_t q_off = (((int64_t)b * H + h) * I + i_) * D;
                for (int d = threadIdx.x; d < D; d += blk.size())
                    q_tile[it*D + d] = Q[q_off + d];
            }
        }
        for (int jt = 0; jt < TILE_J; ++jt) {
            int j = jb + jt;
            if (j < J) {
                const int64_t r_off = (((int64_t)b * H + h) * J + j) * D;
                for (int d = threadIdx.x; d < D; d += blk.size())
                    r_tile[jt*D + d] = R[r_off + d];
            }
        }
        cg::sync(blk);

        /* compute dots inside the tile */
        for (int it = 0; it < TILE_I && ib+it < I; ++it)
        for (int jt = 0; jt < TILE_J && jb+jt < J; ++jt)
        {
            float part = 0.f;
            for (int d = threadIdx.x; d < D; d += blk.size())
                part += s_query_vec[d] * q_tile[it*D + d] * r_tile[jt*D + d];
            float dot = block_reduce_sum(part, red_buf);
            if (threadIdx.x == 0) max_dot = fmaxf(max_dot, dot * scale);
            cg::sync(blk);
        }
    }
    if (threadIdx.x == 0) red_buf[0] = max_dot;
    cg::sync(blk);
    max_dot = red_buf[0];

    /* -------- pass 2 : sum exp -------- */
    float sum_exp = 0.f;
    for (int ib = 0; ib < I; ib += TILE_I)
    for (int jb = 0; jb < J; jb += TILE_J)
    {
        /* reload Q/R tiles */
        for (int it = 0; it < TILE_I; ++it) {
            int i_ = ib + it;
            if (i_ < I) {
                const int64_t q_off = (((int64_t)b * H + h) * I + i_) * D;
                for (int d = threadIdx.x; d < D; d += blk.size())
                    q_tile[it*D + d] = Q[q_off + d];
            }
        }
        for (int jt = 0; jt < TILE_J; ++jt) {
            int j = jb + jt;
            if (j < J) {
                const int64_t r_off = (((int64_t)b * H + h) * J + j) * D;
                for (int d = threadIdx.x; d < D; d += blk.size())
                    r_tile[jt*D + d] = R[r_off + d];
            }
        }
        cg::sync(blk);

        for (int it = 0; it < TILE_I && ib+it < I; ++it)
        for (int jt = 0; jt < TILE_J && jb+jt < J; ++jt)
        {
            float part = 0.f;
            for (int d = threadIdx.x; d < D; d += blk.size())
                part += s_query_vec[d] * q_tile[it*D + d] * r_tile[jt*D + d];
            float dot = block_reduce_sum(part, red_buf);
            if (threadIdx.x == 0) sum_exp += expf(dot*scale - max_dot);
            cg::sync(blk);
        }
    }
    if (threadIdx.x == 0) red_buf[0] = sum_exp;
    cg::sync(blk);
    sum_exp = red_buf[0];
    const float inv_denom = 1.f / sum_exp;

    /* -------- pass 3 : final accumulation -------- */
    float y_val = 0.f;
    for (int ib = 0; ib < I; ib += TILE_I)
    for (int jb = 0; jb < J; jb += TILE_J)
    {
        /* load Q/R/V tiles */
        for (int it = 0; it < TILE_I; ++it) {
            int i_ = ib + it;
            if (i_ < I) {
                const int64_t base = (((int64_t)b * H + h) * I + i_) * D;
                for (int d = threadIdx.x; d < D; d += blk.size()) {
                    q_tile [it*D + d] = Q[base + d];
                    v1_tile[it*D + d] = V1[base + d];
                }
            }
        }
        for (int jt = 0; jt < TILE_J; ++jt) {
            int j = jb + jt;
            if (j < J) {
                const int64_t base = (((int64_t)b * H + h) * J + j) * D;
                for (int d = threadIdx.x; d < D; d += blk.size()) {
                    r_tile [jt*D + d] = R [base + d];
                    v2_tile[jt*D + d] = V2[base + d];
                }
            }
        }
        cg::sync(blk);

        for (int it = 0; it < TILE_I && ib+it < I; ++it)
        for (int jt = 0; jt < TILE_J && jb+jt < J; ++jt)
        {
            float part = 0.f;
            for (int d = threadIdx.x; d < D; d += blk.size())
                part += s_query_vec[d] * q_tile[it*D + d] * r_tile[jt*D + d];
            float dot = block_reduce_sum(part, red_buf);
            if (threadIdx.x == 0) red_buf[1] = dot;
            cg::sync(blk);
            dot = red_buf[1];

            /* accumulate */
            for (int d = threadIdx.x; d < D; d += blk.size())
                y_val += (expf(dot*scale - max_dot) * inv_denom) *
                         v1_tile[it*D + d] *
                         v2_tile[jt*D + d];
            cg::sync(blk);
        }
    }

    /* -------- write result -------- */
    for (int d = threadIdx.x; d < D; d += blk.size())
        Y[s_query_off + d] = y_val;
}


__global__ void Yr_gather_kernel(
    const float* __restrict__ R_query, // R is the 'query' for this dimension
    const float* __restrict__ Q,
    const float* __restrict__ S,
    const float* __restrict__ V1,      // Vq_1
    const float* __restrict__ V2,      // Vs_1
    float*       __restrict__ Y,       // Y_r
    int B, int H, int I, int J, int K, int D,
    float scale){
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
__global__ void Yr_gather_kernel(
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
__global__ void Ys_gather_kernel(
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

    // --- GATHER Calls --- 
    // Y_q Gather for fixed_dim = 0 
    // {
    //     const int64_t N = (int64_t)B*H*I*D;
    //     const dim3 blocks((N + threads - 1) / threads);
    //     Yq_gather_kernel<<<blocks, threads>>>(
    //         Q.data_ptr<float>(), R.data_ptr<float>(), S.data_ptr<float>(), //Query Q
    //         Vr_1.data_ptr<float>(), Vs_1.data_ptr<float>(),
    //         Y_q.data_ptr<float>(), 
    //         B, H, I, J, K, D, scale);
    // }
    // // Y_r Gather for fixed_dim = 1 
    // {
    //     const int64_t N = (int64_t)B*H*J*D;
    //     const dim3 blocks((N + threads - 1) / threads);
    //     Yr_gather_kernel<<<blocks, threads>>>(
    //         R.data_ptr<float>(), Q.data_ptr<float>(), S.data_ptr<float>(), 
    //         Vq_1.data_ptr<float>(), Vs_1.data_ptr<float>(), 
    //         Y_r.data_ptr<float>(), 
    //         B, H, I, J, K, D, scale);
    // }
    // // Y_s Gather for fixed_dim = 2 
    // {
    //     const int64_t N = (int64_t)B*H*K*D;
    //     const dim3 blocks((N + threads - 1) / threads);
    //     Ys_gather_kernel<<<blocks, threads>>>(
    //         S.data_ptr<float>(), Q.data_ptr<float>(), R.data_ptr<float>(), //Query S
    //         Vq_1.data_ptr<float>(), Vr_1.data_ptr<float>(), 
    //         Y_s.data_ptr<float>(), 
    //         B, H, I, J, K, D, scale);
    // }

    // --- TILED GATHER Calls --- 
    // Y_q Gather Tiled
    {
        int TpB = (D < 32) ? D : 32;
        dim3 grid(I, H, B);
        dim3 block(TpB);
        size_t smem_size = sizeof(float) * ( D + TILE_J*D + TILE_K*D + TILE_J*D + TILE_K*D + TpB );

        Yq_gather_kernel_tiled<<<grid, block, smem_size>>>(
            Q.data_ptr<float>(), R.data_ptr<float>(), S.data_ptr<float>(),
            Vr_1.data_ptr<float>(), Vs_1.data_ptr<float>(),
            Y_q.data_ptr<float>(), 
            B, H, I, J, K, D, scale);
    }

    // Y_r Gather Tiled
    {
        int TpB = (D < 32) ? D : 32;
        dim3 grid(J, H, B);
        dim3 block(TpB);
        size_t smem_size = sizeof(float) * ( D + TILE_I*D + TILE_K*D + TILE_I*D + TILE_K*D + TpB );
        Yr_gather_kernel_tiled<<<grid, block, smem_size>>>(
            R.data_ptr<float>(), Q.data_ptr<float>(), S.data_ptr<float>(), 
            Vq_1.data_ptr<float>(), Vs_1.data_ptr<float>(), 
            Y_r.data_ptr<float>(), 
            B, H, I, J, K, D, scale);
    }
    // Y_s Gather Tiled
    {
        int TpB = (D < 32) ? D : 32;
        dim3 grid(K, H, B);
        dim3 block(TpB);
        size_t smem_size = sizeof(float) * ( D + TILE_I*D + TILE_J*D + TILE_I*D + TILE_J*D + TpB );
        Ys_gather_kernel_tiled<<<grid, block, smem_size>>>(
            S.data_ptr<float>(), Q.data_ptr<float>(), R.data_ptr<float>(),
            Vq_1.data_ptr<float>(), Vr_1.data_ptr<float>(), 
            Y_s.data_ptr<float>(), 
            B, H, I, J, K, D, scale);
    }


    // --- OPTIMIZED SCATTER Calls (Slice-wise) --- 
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            // Get GPU slices for current (b,h)
            auto Q_slice_gpu = Q.select(0, b).select(0, h);
            auto R_slice_gpu = R.select(0, b).select(0, h);
            auto S_slice_gpu = S.select(0, b).select(0, h);
            auto Vq_2_slice_gpu = Vq_2.select(0, b).select(0, h);
            auto Vr_2_slice_gpu = Vr_2.select(0, b).select(0, h);
            auto Vs_2_slice_gpu = Vs_2.select(0, b).select(0, h);

            // Pre-compute A and softmax slices
            torch::Tensor A_slice_gpu = compute_A_slice_cuda_wrapper(Q_slice_gpu, R_slice_gpu, S_slice_gpu, scale);
            torch::Tensor Aq_slice_gpu = compute_Aq_slice_cuda_wrapper(A_slice_gpu);
            torch::Tensor Ar_slice_gpu = compute_Ar_slice_cuda_wrapper(A_slice_gpu);
            torch::Tensor As_slice_gpu = compute_As_slice_cuda_wrapper(A_slice_gpu);

            // Y_q_ scatter
            {
                auto Y_q__slice_gpu = Y_q_.select(0, b).select(0, h);
                const int64_t N = (int64_t)I*D;
                const dim3 blocks((N + threads - 1) / threads);
                Yq_scatter_kernel_optimized<<<blocks, threads>>>(
                    Ar_slice_gpu.data_ptr<float>(), As_slice_gpu.data_ptr<float>(),
                    Vr_2_slice_gpu.data_ptr<float>(), Vs_2_slice_gpu.data_ptr<float>(),
                    Y_q__slice_gpu.data_ptr<float>(),
                    I, J, K, D
                );
            }
            // Y_r_ scatter
            {
                auto Y_r__slice_gpu = Y_r_.select(0, b).select(0, h);
                const int64_t N = (int64_t)J*D;
                const dim3 blocks((N + threads - 1) / threads);
                Yr_scatter_kernel_optimized<<<blocks, threads>>>(
                    Aq_slice_gpu.data_ptr<float>(), As_slice_gpu.data_ptr<float>(),
                    Vq_2_slice_gpu.data_ptr<float>(), Vs_2_slice_gpu.data_ptr<float>(),
                    Y_r__slice_gpu.data_ptr<float>(),
                    I, J, K, D
                );
            }
            // Y_s_ scatter
            {
                auto Y_s__slice_gpu = Y_s_.select(0, b).select(0, h);
                const int64_t N = (int64_t)K*D;
                const dim3 blocks((N + threads - 1) / threads);
                Ys_scatter_kernel_optimized<<<blocks, threads>>>(
                    Aq_slice_gpu.data_ptr<float>(), Ar_slice_gpu.data_ptr<float>(),
                    Vq_2_slice_gpu.data_ptr<float>(), Vr_2_slice_gpu.data_ptr<float>(),
                    Y_s__slice_gpu.data_ptr<float>(),
                    I, J, K, D
                );
            }
        }
    }


    cudaDeviceSynchronize(); 
    return std::make_tuple(Y_q, Y_r, Y_s, Y_q_, Y_r_, Y_s_);}




// --- Backward Pass ---
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

// __global__ void apply_softmax_backward_kernel(
//     const float* __restrict__ grad_Aq_slice_in, // [I, J, K]
//     const float* __restrict__ grad_Ar_slice_in, // [I, J, K]
//     const float* __restrict__ grad_As_slice_in, // [I, J, K]
//     const float* __restrict__ Aq_slice_in,      // [I, J, K]
//     const float* __restrict__ Ar_slice_in,      // [I, J, K]
//     const float* __restrict__ As_slice_in,      // [I, J, K]
//     float* __restrict__ grad_A_slice_out,    // [I, J, K]
//     int I_dim, int J_dim, int K_dim
// ) {
//     // Map 3D thread indices to (i, j, k) for grad_A_slice_out
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     int j = blockIdx.y * blockDim.y + threadIdx.y;
//     int k = blockIdx.z * blockDim.z + threadIdx.z;

//     // Boundary check
//     if (i >= I_dim || j >= J_dim || k >= K_dim) {
//         return;
//     }

//     int64_t ijk_idx = (int64_t)i * J_dim * K_dim + (int64_t)j * K_dim + k;
//     float final_grad_A_val = 0.0f;

//     // --- 2.1 Contribution from Aq (Softmax over j, k for fixed i) ---
//     // sum_q = sum_{j',k'} (grad_Aq[i,j',k'] * Aq[i,j',k'])
//     // This sum is specific to each 'i'.
//     // All threads with the same 'i' (blockIdx.x * blockDim.x + threadIdx.x) participate.
//     // We need a reduction across J_dim * K_dim for each 'i'.

//     // For simplicity in this kernel, each thread (i,j,k) calculates its part for sum_q, sum_r, sum_s.
//     // More optimized: a dedicated reduction kernel or block-wide reduction for each sum_q[i], sum_r[j], sum_s[k].
//     // This version is less optimal for the sum_q/r/s but simpler to write initially.
//     // It recomputes sums, which is not ideal.

//     // --- Contribution from grad_Aq ---
//     float sum_q_for_ijk = 0.0f;
//     for (int j_prime = 0; j_prime < J_dim; ++j_prime) {
//         for (int k_prime = 0; k_prime < K_dim; ++k_prime) {
//             int64_t i_jprime_kprime_idx = (int64_t)i * J_dim * K_dim + (int64_t)j_prime * K_dim + k_prime;
//             sum_q_for_ijk += grad_Aq_slice_in[i_jprime_kprime_idx] * Aq_slice_in[i_jprime_kprime_idx];
//         }
//     }
//     final_grad_A_val += (grad_Aq_slice_in[ijk_idx] - sum_q_for_ijk) * Aq_slice_in[ijk_idx];


//     // --- Contribution from grad_Ar ---
//     float sum_r_for_ijk = 0.0f;
//     for (int i_prime = 0; i_prime < I_dim; ++i_prime) {
//         for (int k_prime = 0; k_prime < K_dim; ++k_prime) {
//             int64_t iprime_j_kprime_idx = (int64_t)i_prime * J_dim * K_dim + (int64_t)j * K_dim + k_prime;
//             sum_r_for_ijk += grad_Ar_slice_in[iprime_j_kprime_idx] * Ar_slice_in[iprime_j_kprime_idx];
//         }
//     }
//     final_grad_A_val += (grad_Ar_slice_in[ijk_idx] - sum_r_for_ijk) * Ar_slice_in[ijk_idx];
    

//     // --- Contribution from grad_As ---
//     float sum_s_for_ijk = 0.0f;
//     for (int i_prime = 0; i_prime < I_dim; ++i_prime) {
//         for (int j_prime = 0; j_prime < J_dim; ++j_prime) {
//             int64_t iprime_jprime_k_idx = (int64_t)i_prime * J_dim * K_dim + (int64_t)j_prime * K_dim + k;
//             sum_s_for_ijk += grad_As_slice_in[iprime_jprime_k_idx] * As_slice_in[iprime_jprime_k_idx];
//         }
//     }
//     final_grad_A_val += (grad_As_slice_in[ijk_idx] - sum_s_for_ijk) * As_slice_in[ijk_idx];

//     grad_A_slice_out[ijk_idx] = final_grad_A_val;
// }

// torch::Tensor apply_softmax_backward_cuda_wrapper(
//     const torch::Tensor& grad_Aq_slice_gpu, // [I, J, K]
//     const torch::Tensor& grad_Ar_slice_gpu, // [I, J, K]
//     const torch::Tensor& grad_As_slice_gpu, // [I, J, K]
//     const torch::Tensor& Aq_slice_gpu,      // [I, J, K]
//     const torch::Tensor& Ar_slice_gpu,      // [I, J, K]
//     const torch::Tensor& As_slice_gpu       // [I, J, K]
// ) {
//     TORCH_CHECK(grad_Aq_slice_gpu.is_cuda(), "grad_Aq_slice_gpu must be CUDA");
//     TORCH_CHECK(grad_Aq_slice_gpu.dim() == 3 && Aq_slice_gpu.dim() == 3, "Inputs must be 3D");
    
//     const int I = grad_Aq_slice_gpu.size(0);
//     const int J = grad_Aq_slice_gpu.size(1);
//     const int K = grad_Aq_slice_gpu.size(2);

//     TORCH_CHECK(grad_Ar_slice_gpu.sizes() == grad_Aq_slice_gpu.sizes(), "grad_Ar shape mismatch");
//     // ... add other shape consistency checks ...

//     auto options = grad_Aq_slice_gpu.options();
//     torch::Tensor grad_A_slice_out_gpu = torch::zeros({I, J, K}, options);

//     constexpr int BLOCK_DIM_I = 8;
//     constexpr int BLOCK_DIM_J = 8;
//     constexpr int BLOCK_DIM_K = 8; 
//     dim3 blockDim(BLOCK_DIM_I, BLOCK_DIM_J, BLOCK_DIM_K);
//     dim3 gridDim(
//         (I + BLOCK_DIM_I - 1) / BLOCK_DIM_I,
//         (J + BLOCK_DIM_J - 1) / BLOCK_DIM_J,
//         (K + BLOCK_DIM_K - 1) / BLOCK_DIM_K
//     );

//     // Ensure inputs are contiguous
//     auto grad_Aq_cont = grad_Aq_slice_gpu.contiguous();
//     auto grad_Ar_cont = grad_Ar_slice_gpu.contiguous();
//     auto grad_As_cont = grad_As_slice_gpu.contiguous();
//     auto Aq_cont = Aq_slice_gpu.contiguous();
//     auto Ar_cont = Ar_slice_gpu.contiguous();
//     auto As_cont = As_slice_gpu.contiguous();

//     apply_softmax_backward_kernel<<<gridDim, blockDim>>>(
//         grad_Aq_cont.data_ptr<float>(),
//         grad_Ar_cont.data_ptr<float>(),
//         grad_As_cont.data_ptr<float>(),
//         Aq_cont.data_ptr<float>(),
//         Ar_cont.data_ptr<float>(),
//         As_cont.data_ptr<float>(),
//         grad_A_slice_out_gpu.data_ptr<float>(),
//         I, J, K
//     );

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



