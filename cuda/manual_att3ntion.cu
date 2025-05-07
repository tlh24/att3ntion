#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>      
#include <cuda.h>
#include <cuda_runtime.h>
#include "../cpp/manual_att3ntion.h"


// Forward declarations
__global__ void gather_dim0_kernel(
    const float* __restrict__ Q, 
    const float* __restrict__ R, 
    const float* __restrict__ S,
    const float* __restrict__ V1, 
    const float* __restrict__ V2,
    float*       __restrict__ Y,
    int B, int H, int I, int J, int K, int D,
    float scale);

__global__ void gather_dim1_kernel(
    const float* __restrict__ R_query, 
    const float* __restrict__ Q,
    const float* __restrict__ S,
    const float* __restrict__ V1,      
    const float* __restrict__ V2,      
    float*       __restrict__ Y,       
    int B, int H, int I, int J, int K, int D,
    float scale);

__global__ void gather_dim2_kernel(
    const float* __restrict__ S_query, 
    const float* __restrict__ Q,
    const float* __restrict__ R,
    const float* __restrict__ V1,      
    const float* __restrict__ V2,      
    float*       __restrict__ Y,      
    int B, int H, int I, int J, int K, int D,
    float scale);

__global__ void scatter_dim0_kernel(
    const float* Q,
    const float* R,
    const float* S,
    const float* Vr_2, 
    const float* Vs_2, 
    float*       Y_q_, 
    int B, int H, int I, int J, int K, int D,
    float scale);

__global__ void scatter_dim1_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ R,
    const float* __restrict__ S,
    const float* __restrict__ Vq_2, 
    const float* __restrict__ Vs_2, 
    float*       __restrict__ Y_r_, 
    int B, int H, int I, int J, int K, int D,
    float scale);

__global__ void scatter_dim2_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ R,
    const float* __restrict__ S,
    const float* __restrict__ Vq_2, 
    const float* __restrict__ Vr_2, 
    float*       __restrict__ Y_s_, 
    int B, int H, int I, int J, int K, int D,
    float scale);

__global__ void gather_grad_Vq1_kernel(
    const float* gradY, 
    const float* Q,     
    const float* R,     
    const float* S,     
    const float* Vr_1,  
    const float* Vs_1,  
    float*       gradVq1, 
    int B, int H, int I, int J, int K, int D, int N_grad,
    float scale);

__global__ void gather_grad_Vr1_kernel(
    const float* gradY,
    const float* Q,
    const float* R,
    const float* S,
    const float* Vq_1,
    const float* Vs_1,
    float*       gradVr1,
    int B, int H, int I, int J, int K, int D, int N_grad,
    float scale);

__global__ void gather_grad_Vs1_kernel(
    const float* gradY,
    const float* Q,
    const float* R,
    const float* S,
    const float* Vq_1,
    const float* Vr_1,
    float*       gradVs1,
    int B, int H, int I, int J, int K, int D, int N_grad,
    float scale);

__global__ void scatter_grad_Vq2_kernel(
    const float* gradY, 
    const float* Q,     
    const float* R,     
    const float* S,     
    const float* Vr_2,  
    const float* Vs_2,  
    float*       gradVq2, 
    int B, int H, int I, int J, int K, int D, int N_grad,
    float scale);

__global__ void scatter_grad_Vr2_kernel(
    const float* gradY,
    const float* Q,
    const float* R,
    const float* S,
    const float* Vq_2,
    const float* Vs_2,
    float*       gradVr2,
    int B, int H, int I, int J, int K, int D, int N_grad,
    float scale);

__global__ void scatter_grad_Vs2_kernel(
    const float* gradY,
    const float* Q,
    const float* R,
    const float* S,
    const float* Vq_2,
    const float* Vr_2,
    float*       gradVs2,
    int B, int H, int I, int J, int K, int D, int N_grad,
    float scale);

__global__ void compute_grad_Q_kernel_from_gradA(
    const float* __restrict__ grad_A, // Shape [B, H, I, J, K]
    const float* __restrict__ R,      // Shape [B, H, J, D]
    const float* __restrict__ S,      // Shape [B, H, K, D]
    float*       __restrict__ grad_Q, // Shape [B, H, I, D] - Output
    const int B, const int H, const int I, const int J, const int K, const int D,
    const float scale);
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

// -- Forward Kernels --
// Y_q Gather for fixed_dim = 0 
__global__ void gather_dim0_kernel(
    const float* __restrict__ Q, 
    const float* __restrict__ R, 
    const float* __restrict__ S,
    const float* __restrict__ V1, 
    const float* __restrict__ V2,
    float*       __restrict__ Y,
    int B, int H, int I, int J, int K, int D,
    float scale)
{
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
    Y[idx] = y_val;
}

// Y_r Gather for fixed_dim = 1 
__global__ void gather_dim1_kernel(
    const float* __restrict__ R_query, // R is the 'query' for this dimension
    const float* __restrict__ Q,
    const float* __restrict__ S,
    const float* __restrict__ V1,      // Vq_1
    const float* __restrict__ V2,      // Vs_1
    float*       __restrict__ Y,       // Y_r
    int B, int H, int I, int J, int K, int D,
    float scale)
{
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

    Y[idx] = y_val;
}

// Y_s Gather for fixed_dim = 2 
__global__ void gather_dim2_kernel(
    const float* __restrict__ S_query, // S is the 'query' for this dimension
    const float* __restrict__ Q,
    const float* __restrict__ R,
    const float* __restrict__ V1,      // Vq_1
    const float* __restrict__ V2,      // Vr_1
    float*       __restrict__ Y,       // Y_s
    int B, int H, int I, int J, int K, int D,
    float scale)
{
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

    Y[idx] = y_val;
}

// -- Scatter Helpers -- 
// Computes dot product Q[i]*R[j]*S[k] for specific indices
__device__ inline float compute_dot_product_cuda(
    const float* __restrict__ Q, 
    const float* __restrict__ R, 
    const float* __restrict__ S,
    int b, int h, int i, int j, int k, 
    int B, int H, int I, int J, int K, int D)
{
    // Calculate base pointers using passed dimensions
    const float* q_vec = Q + (((int64_t)b * H + h) * I + i) * D;
    const float* r_vec = R + (((int64_t)b * H + h) * J + j) * D;
    const float* s_vec = S + (((int64_t)b * H + h) * K + k) * D;

    float dot = 0.0f;
    #pragma unroll 4
    for (int d = 0; d < D; ++d) {
        dot += q_vec[d] * r_vec[d] * s_vec[d];
    }
    return dot;
}

// Compute single softmax attention value for eg. Ar[i,j,k] (fixed_dim=1) 
__device__ inline float compute_single_softmax_attn_cuda(
    const float* __restrict__ Q, 
    const float* __restrict__ R, 
    const float* __restrict__ S,
    int b, int h, int i_target, int j_target, int k_target,
    int B, int H, int I, int J, int K, int D,
    float scale,
    int fixed_dim // 1 for Ar (softmax over i, k for fixed j), 2 for As (softmax over i, j for fixed k)
)
{
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
    
    return expf(target_dot * scale - max_val) / sum_exp;
}

// Y_q_ Scatter for fixed_dim = 0 
__global__ void scatter_dim0_kernel(
    const float* Q,
    const float* R,
    const float* S,
    const float* Vr_2, 
    const float* Vs_2, 
    float*       Y_q_, 
    int B, int H, int I, int J, int K, int D,
    float scale)
{
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

    Y_q_[idx] = accum_val;
}

// Y_r_ Scatter for fixed_dim = 1 
__global__ void scatter_dim1_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ R,
    const float* __restrict__ S,
    const float* __restrict__ Vq_2, 
    const float* __restrict__ Vs_2, 
    float*       __restrict__ Y_r_, 
    int B, int H, int I, int J, int K, int D,
    float scale)
{
    // global thread index [0 .. B*H*J*D) 
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)B*H*J*D) return;

    // decode (b,h,j,d) from idx - this thread computes Y_r_[b,h,j,d]
    int d = idx % D;
    int tmp = idx / D;
    int j = tmp % J; 
    tmp = tmp / J;
    int h = tmp % H;
    int b = tmp / H;

    float accum_val = 0.0f;

    // Iterate over the dimensions we sum over (i and k)
    for (int i = 0; i < I; ++i) {
        const float* vq2_vec = Vq_2 + (((int64_t)b * H + h) * I + i) * D;
        for (int k = 0; k < K; ++k) {
            const float* vs2_vec = Vs_2 + (((int64_t)b * H + h) * K + k) * D;

            // Aq[b,h,i,j,k] (softmax over j',k' for fixed i)
            float attn_aq = compute_single_softmax_attn_cuda(Q, R, S, b, h, i, j, k, B, H, I, J, K, D, scale, 0); 
            
            // As[b,h,i,j,k] (softmax over i',j' for fixed k)
            float attn_as = compute_single_softmax_attn_cuda(Q, R, S, b, h, i, j, k, B, H, I, J, K, D, scale, 2);

            // Get value components for the specific d
            float vq2_val = vq2_vec[d];
            float vs2_val = vs2_vec[d];

            // Aq * As * Vq_2 * Vs_2
            accum_val += attn_aq * attn_as * vq2_val * vs2_val;
        }
    }

    Y_r_[idx] = accum_val;
}

// Y_s_ Scatter for fixed_dim = 2 
__global__ void scatter_dim2_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ R,
    const float* __restrict__ S,
    const float* __restrict__ Vq_2, 
    const float* __restrict__ Vr_2, 
    float*       __restrict__ Y_s_, 
    int B, int H, int I, int J, int K, int D,
    float scale)
{
    // global thread index [0 .. B*H*K*D)
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)B*H*K*D) return;

    // decode (b,h,k,d) from idx - this thread computes Y_s_[b,h,k,d]
    int d = idx % D;
    int tmp = idx / D;
    int k = tmp % K; // Fixed index for this output element
    tmp = tmp / K;
    int h = tmp % H;
    int b = tmp / H;

    float accum_val = 0.0f;

    // Iterate over the dimensions we sum over (i and j)
    for (int i = 0; i < I; ++i) {
        const float* vq2_vec = Vq_2 + (((int64_t)b * H + h) * I + i) * D;
        for (int j = 0; j < J; ++j) {
            const float* vr2_vec = Vr_2 + (((int64_t)b * H + h) * J + j) * D;

            // Aq[b,h,i,j,k] (softmax over j',k' for fixed i)
            float attn_aq = compute_single_softmax_attn_cuda(Q, R, S, b, h, i, j, k, B, H, I, J, K, D, scale, 0); 
            
            // Ar[b,h,i,j,k] (softmax over i',k' for fixed j)
            float attn_ar = compute_single_softmax_attn_cuda(Q, R, S, b, h, i, j, k, B, H, I, J, K, D, scale, 1);

            // Get value components for the specific d
            float vq2_val = vq2_vec[d];
            float vr2_val = vr2_vec[d];

            // Aq * Ar * Vq_2 * Vr_2
            accum_val += attn_aq * attn_ar * vq2_val * vr2_val;
        }
    }

    Y_s_[idx] = accum_val;
}

// -- Forward pass --
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> forward_cuda(
    at::Tensor Q, at::Tensor R, at::Tensor S,
    at::Tensor Vq_1, at::Tensor Vq_2,
    at::Tensor Vr_1, at::Tensor Vr_2,
    at::Tensor Vs_1, at::Tensor Vs_2,
    double dropout_rate) 
{
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
    {
        const int64_t N = (int64_t)B*H*I*D;
        const dim3 blocks((N + threads - 1) / threads);
        gather_dim0_kernel<<<blocks, threads>>>(
            Q.data_ptr<float>(), R.data_ptr<float>(), S.data_ptr<float>(), //Query Q
            Vr_1.data_ptr<float>(), Vs_1.data_ptr<float>(),
            Y_q.data_ptr<float>(), 
            B, H, I, J, K, D, scale);
    }
    // Y_r Gather for fixed_dim = 1 
    {
        const int64_t N = (int64_t)B*H*J*D;
        const dim3 blocks((N + threads - 1) / threads);
        gather_dim1_kernel<<<blocks, threads>>>(
            R.data_ptr<float>(), Q.data_ptr<float>(), S.data_ptr<float>(), 
            Vq_1.data_ptr<float>(), Vs_1.data_ptr<float>(), 
            Y_r.data_ptr<float>(), 
            B, H, I, J, K, D, scale);
    }
    // Y_s Gather for fixed_dim = 2 
    {
        const int64_t N = (int64_t)B*H*K*D;
        const dim3 blocks((N + threads - 1) / threads);
        gather_dim2_kernel<<<blocks, threads>>>(
            S.data_ptr<float>(), Q.data_ptr<float>(), R.data_ptr<float>(), //Query S
            Vq_1.data_ptr<float>(), Vr_1.data_ptr<float>(), 
            Y_s.data_ptr<float>(), 
            B, H, I, J, K, D, scale);
    }

    // --- SCATTER Calls --- 
    // Y_q_[B,H,I,D]
    {
        const int64_t N = (int64_t)B*H*I*D;
        const dim3 blocks((N + threads - 1) / threads);
        scatter_dim0_kernel<<<blocks, threads>>>(
            Q.data_ptr<float>(), R.data_ptr<float>(), S.data_ptr<float>(),
            Vr_2.data_ptr<float>(), Vs_2.data_ptr<float>(), 
            Y_q_.data_ptr<float>(), 
            B, H, I, J, K, D, scale);
    }
    // Y_r_[B,H,J,D]
    {
        const int64_t N = (int64_t)B*H*J*D;
        const dim3 blocks((N + threads - 1) / threads);
        scatter_dim1_kernel<<<blocks, threads>>>(
            Q.data_ptr<float>(), R.data_ptr<float>(), S.data_ptr<float>(),
            Vq_2.data_ptr<float>(), Vs_2.data_ptr<float>(), 
            Y_r_.data_ptr<float>(), 
            B, H, I, J, K, D, scale);
    }
    // Y_s_[B,H,K,D]
    {
        const int64_t N = (int64_t)B*H*K*D;
        const dim3 blocks((N + threads - 1) / threads);
        scatter_dim2_kernel<<<blocks, threads>>>(
            Q.data_ptr<float>(), R.data_ptr<float>(), S.data_ptr<float>(),
            Vq_2.data_ptr<float>(), Vr_2.data_ptr<float>(), 
            Y_s_.data_ptr<float>(), 
            B, H, I, J, K, D, scale);
    }

    cudaDeviceSynchronize(); 
    return std::make_tuple(Y_q, Y_r, Y_s, Y_q_, Y_r_, Y_s_);
}

// --- Backward Kernels ---

// Kernel to compute gradient for Vq_1
// Mirrors the logic of compute_grad_Vq_1 in the C++ version
__global__ void gather_grad_Vq1_kernel(
    const float* gradY, 
    const float* Q,     
    const float* R,     
    const float* S,     
    const float* Vr_1,  
    const float* Vs_1,  
    float*       gradVq1, 
    int B, int H, int I, int J, int K, int D, int N_grad,
    float scale)
{
    // global thread index maps to [B,H,I,D]
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)B*H*I*D) return;

    // decode (b,h,i,d) - This thread calculates gradVq1[b,h,i,d]
    int d = idx % D;
    int tmp = idx / D;
    int i_target = tmp % I; // This is the target 'i' for the gradient
    tmp = tmp / I;
    int h = tmp % H;
    int b = tmp / H;

    float grad_accum = 0.0f;

    // Base offset for gradY for the current (b, h)
    int64_t gradY_offset_bh = ((int64_t)b * H + h) * N_grad * D;

    // --- 1. Contribution from Y_r path (dL/dY_r) ---
    // dL/dVq_1[i] += sum_{j} [ dL/dY_r[j] * sum_{k} ( Ar[i,j,k] * Vs_1[k] ) ]
    for (int j = 0; j < J; ++j) { // Loop over the source gradient index 'j' from Y_r
        // Check bounds for gradY access
        if (j >= N_grad) continue; 

        // Get dL/dY_r[b,h,j,d] from the combined gradient tensor
        float dy_r = gradY[gradY_offset_bh + (int64_t)j * D + d];

        if (dy_r != 0.0f) { // Optimization
            float term1_sum_k = 0.0f;
            for (int k = 0; k < K; ++k) {
                // Calculate Ar[b,h,i_target,j,k] (softmax over i', k' for fixed j)
                float attn_ar = compute_single_softmax_attn_cuda(Q, R, S, b, h, i_target, j, k, B, H, I, J, K, D, scale, 1);

                // Get Vs_1[b,h,k,d]
                const float* vs1_vec = Vs_1 + (((int64_t)b * H + h) * K + k) * D;
                float vs1_val = vs1_vec[d];

                term1_sum_k += attn_ar * vs1_val;
            }
             grad_accum += dy_r * term1_sum_k;
        }
    }

    // --- 2. Contribution from Y_s path (dL/dY_s) ---
    // dL/dVq_1[i] += sum_{k} [ dL/dY_s[k] * sum_{j} ( As[i,j,k] * Vr_1[j] ) ]
    for (int k = 0; k < K; ++k) { // Loop over the source gradient index 'k' from Y_s
        // Check bounds for gradY access
        if (k >= N_grad) continue;

        // Get dL/dY_s[b,h,k,d] from the combined gradient tensor
        float dy_s = gradY[gradY_offset_bh + (int64_t)k * D + d];

        if (dy_s != 0.0f) { // Optimization
             float term2_sum_j = 0.0f;
            for (int j = 0; j < J; ++j) {
                // Calculate As[b,h,i_target,j,k] (softmax over i', j' for fixed k)
                float attn_as = compute_single_softmax_attn_cuda(Q, R, S, b, h, i_target, j, k, B, H, I, J, K, D, scale, 2);

                // Get Vr_1[b,h,j,d]
                const float* vr1_vec = Vr_1 + (((int64_t)b * H + h) * J + j) * D;
                float vr1_val = vr1_vec[d];
                
                term2_sum_j += attn_as * vr1_val;
            }
             grad_accum += dy_s * term2_sum_j;
        }
    }

    // Write the final accumulated gradient for Vq_1[b,h,i_target,d]
    gradVq1[idx] = grad_accum;
}

// Kernel to compute gradient for Vr_1
// Mirrors the logic of compute_grad_Vr_1 in the C++ version
__global__ void gather_grad_Vr1_kernel(
    const float* gradY,
    const float* Q,
    const float* R,
    const float* S,
    const float* Vq_1,
    const float* Vs_1,
    float*       gradVr1,
    int B, int H, int I, int J, int K, int D, int N_grad,
    float scale)
{
    // global thread index maps to [B,H,J,D]
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)B*H*J*D) return;

    // decode (b,h,j,d) - This thread calculates gradVr1[b,h,j,d]
    int d = idx % D;
    int tmp = idx / D;
    int j_target = tmp % J; // This is the target 'j' for the gradient
    tmp = tmp / J;
    int h = tmp % H;
    int b = tmp / H;

    float grad_accum = 0.0f;

    // Base offset for gradY for the current (b, h)
    int64_t gradY_offset_bh = ((int64_t)b * H + h) * N_grad * D;

    // --- 1. Contribution from Y_q path (dL/dY_q) ---
    // dL/dVr_1[j] += sum_{i} [ dL/dY_q[i] * sum_{k} ( Aq[i,j,k] * Vs_1[k] ) ]
    for (int i = 0; i < I; ++i) { // Loop over the source gradient index 'i' from Y_q
        if (i >= N_grad) continue;
        float dy_q = gradY[gradY_offset_bh + (int64_t)i * D + d];

        if (dy_q != 0.0f) {
            float term1_sum_k = 0.0f;
            for (int k = 0; k < K; ++k) {
                // Calculate Aq[b,h,i,j_target,k] (softmax over j', k' for fixed i)
                float attn_aq = compute_single_softmax_attn_cuda(Q, R, S, b, h, i, j_target, k, B, H, I, J, K, D, scale, 0);
                // Get Vs_1[b,h,k,d]
                const float* vs1_vec = Vs_1 + (((int64_t)b * H + h) * K + k) * D;
                float vs1_val = vs1_vec[d];
                term1_sum_k += attn_aq * vs1_val;
            }
            grad_accum += dy_q * term1_sum_k;
        }
    }

    // --- 2. Contribution from Y_s path (dL/dY_s) ---
    // dL/dVr_1[j] += sum_{k} [ dL/dY_s[k] * sum_{i} ( As[i,j,k] * Vq_1[i] ) ]
    for (int k = 0; k < K; ++k) { // Loop over the source gradient index 'k' from Y_s
        if (k >= N_grad) continue;
        float dy_s = gradY[gradY_offset_bh + (int64_t)k * D + d];

        if (dy_s != 0.0f) {
            float term2_sum_i = 0.0f;
            for (int i = 0; i < I; ++i) {
                // Calculate As[b,h,i,j_target,k] (softmax over i', j' for fixed k)
                float attn_as = compute_single_softmax_attn_cuda(Q, R, S, b, h, i, j_target, k, B, H, I, J, K, D, scale, 2);
                // Get Vq_1[b,h,i,d]
                const float* vq1_vec = Vq_1 + (((int64_t)b * H + h) * I + i) * D;
                float vq1_val = vq1_vec[d];
                term2_sum_i += attn_as * vq1_val;
            }
            grad_accum += dy_s * term2_sum_i;
        }
    }

    gradVr1[idx] = grad_accum;
}

// Kernel to compute gradient for Vs_1
// Mirrors the logic of compute_grad_Vs_1 in the C++ version
__global__ void gather_grad_Vs1_kernel(
    const float* gradY,
    const float* Q,
    const float* R,
    const float* S,
    const float* Vq_1,
    const float* Vr_1,
    float*       gradVs1,
    int B, int H, int I, int J, int K, int D, int N_grad,
    float scale)
{
    // global thread index maps to [B,H,K,D]
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)B*H*K*D) return;

    // decode (b,h,k,d) - This thread calculates gradVs1[b,h,k,d]
    int d = idx % D;
    int tmp = idx / D;
    int k_target = tmp % K; // This is the target 'k' for the gradient
    tmp = tmp / K;
    int h = tmp % H;
    int b = tmp / H;

    float grad_accum = 0.0f;

    // Base offset for gradY for the current (b, h)
    int64_t gradY_offset_bh = ((int64_t)b * H + h) * N_grad * D;

    // --- 1. Contribution from Y_q path (dL/dY_q) ---
    // dL/dVs_1[k] += sum_{i} [ dL/dY_q[i] * sum_{j} ( Aq[i,j,k] * Vr_1[j] ) ]
    for (int i = 0; i < I; ++i) { // Loop over the source gradient index 'i' from Y_q
        if (i >= N_grad) continue;
        float dy_q = gradY[gradY_offset_bh + (int64_t)i * D + d];

        if (dy_q != 0.0f) {
            float term1_sum_j = 0.0f;
            for (int j = 0; j < J; ++j) {
                // Calculate Aq[b,h,i,j,k_target] (softmax over j', k' for fixed i)
                float attn_aq = compute_single_softmax_attn_cuda(Q, R, S, b, h, i, j, k_target, B, H, I, J, K, D, scale, 0);
                // Get Vr_1[b,h,j,d]
                const float* vr1_vec = Vr_1 + (((int64_t)b * H + h) * J + j) * D;
                float vr1_val = vr1_vec[d];
                term1_sum_j += attn_aq * vr1_val;
            }
            grad_accum += dy_q * term1_sum_j;
        }
    }

    // --- 2. Contribution from Y_r path (dL/dY_r) ---
    // dL/dVs_1[k] += sum_{j} [ dL/dY_r[j] * sum_{i} ( Ar[i,j,k] * Vq_1[i] ) ]
    for (int j = 0; j < J; ++j) { // Loop over the source gradient index 'j' from Y_r
        if (j >= N_grad) continue;
        float dy_r = gradY[gradY_offset_bh + (int64_t)j * D + d];

        if (dy_r != 0.0f) {
            float term2_sum_i = 0.0f;
            for (int i = 0; i < I; ++i) {
                // Calculate Ar[b,h,i,j,k_target] (softmax over i', k' for fixed j)
                float attn_ar = compute_single_softmax_attn_cuda(Q, R, S, b, h, i, j, k_target, B, H, I, J, K, D, scale, 1);
                // Get Vq_1[b,h,i,d]
                const float* vq1_vec = Vq_1 + (((int64_t)b * H + h) * I + i) * D;
                float vq1_val = vq1_vec[d];
                term2_sum_i += attn_ar * vq1_val;
            }
            grad_accum += dy_r * term2_sum_i;
        }
    }

    gradVs1[idx] = grad_accum;
}

// Kernel to compute gradient for Vq_2
// Mirrors the logic of compute_grad_Vq_2 in the C++ version
__global__ void scatter_grad_Vq2_kernel(
    const float* gradY, 
    const float* Q,     
    const float* R,     
    const float* S,     
    const float* Vr_2,  
    const float* Vs_2,  
    float*       gradVq2, 
    int B, int H, int I, int J, int K, int D, int N_grad,
    float scale)
{
    // global thread index maps to [B,H,I,D]
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)B*H*I*D) return;

    // decode (b,h,i,d) - This thread calculates gradVq2[b,h,i,d]
    int d = idx % D;
    int tmp = idx / D;
    int i_target = tmp % I; // This is the target 'i' for the gradient
    tmp = tmp / I;
    int h = tmp % H;
    int b = tmp / H;

    float grad_accum = 0.0f;

    // Base offset for gradY for the current (b, h)
    // gradY contains the upstream gradients dL/dY_r_ and dL/dY_s_
    int64_t gradY_offset_bh = ((int64_t)b * H + h) * N_grad * D;

    // --- 1. Contribution from Y_r_ path (dL/dY_r_) ---
    // dL/dVq_2[i] += sum_{j,k} [ dL/dY_r_[j] * Aq[i,j,k] * As[i,j,k] * Vs_2[k] ]
    for (int j = 0; j < J; ++j) { // Loop over the source gradient index 'j' from Y_r_
        if (j >= N_grad) continue; // Check bounds for gradY access
        float dy_r = gradY[gradY_offset_bh + (int64_t)j * D + d];

        if (dy_r != 0.0f) { // Optimization
            for (int k = 0; k < K; ++k) {
                // Calculate Aq[b,h,i_target,j,k] (softmax over j', k' for fixed i)
                float attn_aq = compute_single_softmax_attn_cuda(Q, R, S, b, h, i_target, j, k, B, H, I, J, K, D, scale, 0);
                // Calculate As[b,h,i_target,j,k] (softmax over i', j' for fixed k)
                float attn_as = compute_single_softmax_attn_cuda(Q, R, S, b, h, i_target, j, k, B, H, I, J, K, D, scale, 2);
                // Get Vs_2[b,h,k,d]
                const float* vs2_vec = Vs_2 + (((int64_t)b * H + h) * K + k) * D;
                float vs2_val = vs2_vec[d];

                grad_accum += dy_r * attn_aq * attn_as * vs2_val;
            }
        }
    }

    // --- 2. Contribution from Y_s_ path (dL/dY_s_) ---
    // dL/dVq_2[i] += sum_{k,j} [ dL/dY_s_[k] * Aq[i,j,k] * Ar[i,j,k] * Vr_2[j] ]
    for (int k = 0; k < K; ++k) { // Loop over the source gradient index 'k' from Y_s_
        if (k >= N_grad) continue; // Check bounds for gradY access
        float dy_s = gradY[gradY_offset_bh + (int64_t)k * D + d];

        if (dy_s != 0.0f) { // Optimization
            for (int j = 0; j < J; ++j) {
                 // Calculate Aq[b,h,i_target,j,k] (softmax over j', k' for fixed i)
                 float attn_aq = compute_single_softmax_attn_cuda(Q, R, S, b, h, i_target, j, k, B, H, I, J, K, D, scale, 0);
                 // Calculate Ar[b,h,i_target,j,k] (softmax over i', k' for fixed j)
                 float attn_ar = compute_single_softmax_attn_cuda(Q, R, S, b, h, i_target, j, k, B, H, I, J, K, D, scale, 1);
                 // Get Vr_2[b,h,j,d]
                 const float* vr2_vec = Vr_2 + (((int64_t)b * H + h) * J + j) * D;
                 float vr2_val = vr2_vec[d];

                 grad_accum += dy_s * attn_aq * attn_ar * vr2_val;
            }
        }
    }

    // Write the final accumulated gradient for Vq_2[b,h,i_target,d]
    gradVq2[idx] = grad_accum;
}

// Kernel to compute gradient for Vr_2
// Mirrors the logic of compute_grad_Vr_2 in the C++ version
__global__ void scatter_grad_Vr2_kernel(
    const float* gradY, 
    const float* Q,     
    const float* R,     
    const float* S,     
    const float* Vq_2,  
    const float* Vs_2,  
    float*       gradVr2, 
    int B, int H, int I, int J, int K, int D, int N_grad,
    float scale)
{
    // global thread index maps to [B,H,J,D]
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)B*H*J*D) return;

    // decode (b,h,j,d) - This thread calculates gradVr2[b,h,j,d]
    int d = idx % D;
    int tmp = idx / D;
    int j_target = tmp % J; // This is the target 'j' for the gradient
    tmp = tmp / J;
    int h = tmp % H;
    int b = tmp / H;

    float grad_accum = 0.0f;

    // Base offset for gradY for the current (b, h)
    int64_t gradY_offset_bh = ((int64_t)b * H + h) * N_grad * D;

    // --- 1. Contribution from Y_q_ path (dL/dY_q_) ---
    // dL/dVr_2[j] += sum_{i,k} [ dL/dY_q_[i] * Ar[i,j,k] * As[i,j,k] * Vs_2[k] ]
    for (int i = 0; i < I; ++i) { // Loop over the source gradient index 'i' from Y_q_
        if (i >= N_grad) continue; // Check bounds for gradY access
        float dy_q = gradY[gradY_offset_bh + (int64_t)i * D + d];

        if (dy_q != 0.0f) { // Optimization
            for (int k = 0; k < K; ++k) {
                // Calculate Ar[b,h,i,j_target,k] (softmax over i', k' for fixed j)
                float attn_ar = compute_single_softmax_attn_cuda(Q, R, S, b, h, i, j_target, k, B, H, I, J, K, D, scale, 1);
                // Calculate As[b,h,i,j_target,k] (softmax over i', j' for fixed k)
                float attn_as = compute_single_softmax_attn_cuda(Q, R, S, b, h, i, j_target, k, B, H, I, J, K, D, scale, 2);
                // Get Vs_2[b,h,k,d]
                const float* vs2_vec = Vs_2 + (((int64_t)b * H + h) * K + k) * D;
                float vs2_val = vs2_vec[d];

                grad_accum += dy_q * attn_ar * attn_as * vs2_val;
            }
        }
    }

    // --- 2. Contribution from Y_s_ path (dL/dY_s_) ---
    // dL/dVr_2[j] += sum_{k,i} [ dL/dY_s_[k] * Aq[i,j,k] * Ar[i,j,k] * Vq_2[i] ]
    for (int k = 0; k < K; ++k) { // Loop over the source gradient index 'k' from Y_s_
        if (k >= N_grad) continue; // Check bounds for gradY access
        float dy_s = gradY[gradY_offset_bh + (int64_t)k * D + d];

        if (dy_s != 0.0f) { // Optimization
            for (int i = 0; i < I; ++i) {
                // Calculate Aq[b,h,i,j_target,k] (softmax over j', k' for fixed i)
                float attn_aq = compute_single_softmax_attn_cuda(Q, R, S, b, h, i, j_target, k, B, H, I, J, K, D, scale, 0);
                // Calculate Ar[b,h,i,j_target,k] (softmax over i', k' for fixed j)
                float attn_ar = compute_single_softmax_attn_cuda(Q, R, S, b, h, i, j_target, k, B, H, I, J, K, D, scale, 1);
                // Get Vq_2[b,h,i,d]
                const float* vq2_vec = Vq_2 + (((int64_t)b * H + h) * I + i) * D;
                float vq2_val = vq2_vec[d];

                grad_accum += dy_s * attn_aq * attn_ar * vq2_val;
            }
        }
    }

    // Write the final accumulated gradient for Vr_2[b,h,j_target,d]
    gradVr2[idx] = grad_accum;
}

// Kernel to compute gradient for Vs_2
// Mirrors the logic of compute_grad_Vs_2 in the C++ version
__global__ void scatter_grad_Vs2_kernel(
    const float* gradY, 
    const float* Q,     
    const float* R,     
    const float* S,     
    const float* Vq_2,  
    const float* Vr_2,  
    float*       gradVs2, 
    int B, int H, int I, int J, int K, int D, int N_grad,
    float scale)
{
    // global thread index maps to [B,H,K,D]
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)B*H*K*D) return;

    // decode (b,h,k,d) - This thread calculates gradVs2[b,h,k,d]
    int d = idx % D;
    int tmp = idx / D;
    int k_target = tmp % K; // This is the target 'k' for the gradient
    tmp = tmp / K;
    int h = tmp % H;
    int b = tmp / H;

    float grad_accum = 0.0f;

    // Base offset for gradY for the current (b, h)
    int64_t gradY_offset_bh = ((int64_t)b * H + h) * N_grad * D;

    // --- 1. Contribution from Y_q_ path (dL/dY_q_) ---
    // dL/dVs_2[k] += sum_{i,j} [ dL/dY_q_[i] * Ar[i,j,k] * As[i,j,k] * Vr_2[j] ]
    for (int i = 0; i < I; ++i) { // Loop over the source gradient index 'i' from Y_q_
        if (i >= N_grad) continue; // Check bounds for gradY access
        float dy_q = gradY[gradY_offset_bh + (int64_t)i * D + d];

        if (dy_q != 0.0f) { // Optimization
            for (int j = 0; j < J; ++j) {
                // Calculate Ar[b,h,i,j,k_target] (softmax over i', k' for fixed j)
                float attn_ar = compute_single_softmax_attn_cuda(Q, R, S, b, h, i, j, k_target, B, H, I, J, K, D, scale, 1);
                // Calculate As[b,h,i,j,k_target] (softmax over i', j' for fixed k)
                float attn_as = compute_single_softmax_attn_cuda(Q, R, S, b, h, i, j, k_target, B, H, I, J, K, D, scale, 2);
                // Get Vr_2[b,h,j,d]
                const float* vr2_vec = Vr_2 + (((int64_t)b * H + h) * J + j) * D;
                float vr2_val = vr2_vec[d];

                grad_accum += dy_q * attn_ar * attn_as * vr2_val;
            }
        }
    }

    // --- 2. Contribution from Y_r_ path (dL/dY_r_) ---
    // dL/dVs_2[k] += sum_{j,i} [ dL/dY_r_[j] * Aq[i,j,k] * As[i,j,k] * Vq_2[i] ]
    for (int j = 0; j < J; ++j) { // Loop over the source gradient index 'j' from Y_r_
        if (j >= N_grad) continue; // Check bounds for gradY access
        float dy_r = gradY[gradY_offset_bh + (int64_t)j * D + d];

        if (dy_r != 0.0f) { // Optimization
            for (int i = 0; i < I; ++i) {
                // Calculate Aq[b,h,i,j,k_target] (softmax over j', k' for fixed i)
                float attn_aq = compute_single_softmax_attn_cuda(Q, R, S, b, h, i, j, k_target, B, H, I, J, K, D, scale, 0);
                // Calculate As[b,h,i,j,k_target] (softmax over i', j' for fixed k)
                float attn_as = compute_single_softmax_attn_cuda(Q, R, S, b, h, i, j, k_target, B, H, I, J, K, D, scale, 2);
                // Get Vq_2[b,h,i,d]
                const float* vq2_vec = Vq_2 + (((int64_t)b * H + h) * I + i) * D;
                float vq2_val = vq2_vec[d];

                grad_accum += dy_r * attn_aq * attn_as * vq2_val;
            }
        }
    }

    // Write the final accumulated gradient for Vs_2[b,h,k_target,d]
    gradVs2[idx] = grad_accum;
}

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
    // Shared memory for the current JxK plane of A_slice and for reduction
    // Requires J_dim * K_dim floats for the plane, and 
    // THREADS_PER_BLOCK floats for the reduction scratchpad if THREADS_PER_BLOCK > warpSize
    extern __shared__ float s_data[]; 
    
    // Current 'i' index this block is responsible for
    int i_current = blockIdx.x;

    // Ensure this block is within the valid range of 'i'
    if (i_current >= I_dim) {
        return;
    }

    // Base pointer to the current A_slice[i_current, :, :] in global memory
    const float* current_A_plane_global = A_slice_global + (int64_t)i_current * J_dim * K_dim;
    // Base pointer for output Aq_out[i_current, :, :]
    float* current_Aq_plane_global = Aq_out_global + (int64_t)i_current * J_dim * K_dim;

    // --- Load A_slice[i_current, :, :] into shared memory s_A_plane ---
    // s_A_plane will be the first J_dim * K_dim elements of s_data
    float* s_A_plane = s_data; 
    int plane_size = J_dim * K_dim;
    int tid_in_block = threadIdx.y * blockDim.x + threadIdx.x; // Linear thread ID within the block
    int threads_in_block = blockDim.x * blockDim.y;

    for (int idx = tid_in_block; idx < plane_size; idx += threads_in_block) {
        s_A_plane[idx] = current_A_plane_global[idx];
    }
    __syncthreads(); // Ensure all data is loaded into s_A_plane

    // --- Find max_val in s_A_plane for numerical stability (Parallel Reduction) ---
    // s_reduction_pad will be after s_A_plane in s_data
    float* s_reduction_pad = s_data + plane_size; 
                                                
    float thread_max_val = -FLT_MAX; // Initialize with a very small number
    for (int idx = tid_in_block; idx < plane_size; idx += threads_in_block) {
        if (s_A_plane[idx] > thread_max_val) {
            thread_max_val = s_A_plane[idx];
        }
    }
    s_reduction_pad[tid_in_block] = thread_max_val;
    __syncthreads();

    // Perform reduction in shared memory (assumes threads_in_block is power of 2 for simplicity here)
    // A more robust reduction handles non-power-of-2 block sizes.
    for (int offset = threads_in_block / 2; offset > 0; offset >>= 1) {
        if (tid_in_block < offset) {
            if (s_reduction_pad[tid_in_block + offset] > s_reduction_pad[tid_in_block]) {
                 s_reduction_pad[tid_in_block] = s_reduction_pad[tid_in_block + offset];
            }
        }
        __syncthreads();
    }
    float plane_max_val = s_reduction_pad[0]; // Max value for the current plane
    __syncthreads(); // Ensure all threads see the correct plane_max_val


    // --- Compute sum_exp for the plane (Parallel Reduction) ---
    float thread_sum_exp = 0.0f;
    for (int idx = tid_in_block; idx < plane_size; idx += threads_in_block) {
        thread_sum_exp += expf(s_A_plane[idx] - plane_max_val);
    }
    s_reduction_pad[tid_in_block] = thread_sum_exp;
    __syncthreads();

    for (int offset = threads_in_block / 2; offset > 0; offset >>= 1) {
        if (tid_in_block < offset) {
            s_reduction_pad[tid_in_block] += s_reduction_pad[tid_in_block + offset];
        }
        __syncthreads();
    }
    float plane_sum_exp = s_reduction_pad[0];
    __syncthreads();

    // --- Compute softmax values and write to global memory ---
    if (plane_sum_exp == 0.0f) plane_sum_exp = 1e-20f; // Avoid division by zero

    for (int idx = tid_in_block; idx < plane_size; idx += threads_in_block) {
        float softmax_val = expf(s_A_plane[idx] - plane_max_val) / plane_sum_exp;
        current_Aq_plane_global[idx] = softmax_val;
    }
}

// Add this C++ wrapper function, for example, after compute_A_slice_cuda_wrapper

torch::Tensor compute_Aq_slice_cuda_wrapper(
    const torch::Tensor& A_slice_gpu // Assumed to be on GPU, shape [I,J,K]
) {
    TORCH_CHECK(A_slice_gpu.is_cuda(), "A_slice_gpu must be a CUDA tensor");
    TORCH_CHECK(A_slice_gpu.dim() == 3, "A_slice_gpu must be 3D [I,J,K]");

    const int I = A_slice_gpu.size(0);
    const int J = A_slice_gpu.size(1);
    const int K = A_slice_gpu.size(2);

    // Allocate output tensor Aq_slice_out_gpu [I,J,K] on GPU
    auto options = A_slice_gpu.options(); // Inherit dtype and device
    torch::Tensor Aq_slice_out_gpu = torch::zeros({I, J, K}, options);

    // Kernel launch configuration for compute_Aq_slice_kernel
    // Each block handles one 'i'-plane (JxK elements)
    dim3 gridDim(I); // I blocks in total, one for each i-plane

    // Threads per block: Try to cover J*K elements.
    // Max threads per block is 1024.
    // Using a 1D block for simplicity, can be optimized to 2D.
    int threads_per_block = std::min(1024, J * K);
    // Ensure threads_per_block is a power of 2 for simpler reduction, or use a more general reduction.
    // For this example, let's pick a common size like 256 or 512 if J*K is large enough.
    // Or, make it precisely J*K if small enough.
    // For robust reduction, block size should be a power of two if using the simple reduction logic.
    // Let's choose a common block size, e.g., 256. The kernel loops if plane_size > threads_per_block.
    threads_per_block = 256; // Example
    if (J * K < threads_per_block && J*K > 0) { // If plane is smaller, use its size (power of 2 padding might be better for reduction)
        // A more robust way is to ensure threads_per_block is a power of 2 for the reduction used.
        // For now, we'll use a fixed size and the kernel loop handles it.
    }


    dim3 blockDim(threads_per_block); // 1D block of threads

    // Calculate shared memory size:
    // J*K floats for s_A_plane + threads_per_block floats for s_reduction_pad
    size_t shared_mem_size = (J * K + threads_per_block) * sizeof(float);
    
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
        // throw std::runtime_error(std::string("CUDA kernel launch failed in compute_Aq_slice_cuda_wrapper: ") + cudaGetErrorString(err));
    }
    // cudaDeviceSynchronize(); // For debugging

    return Aq_slice_out_gpu;
}

// Kernel to compute Ar_slice (softmax over i, k for each fixed j)
// Each block processes one 'j' plane.
__global__ void compute_Ar_slice_kernel(
    const float* __restrict__ A_slice_global, // Input A_slice [I,J,K] (global mem)
    float*       __restrict__ Ar_out_global,  // Output Ar_slice [I,J,K] (global mem)
    int I_dim, int J_dim, int K_dim
) {
    // Shared memory: I_dim * K_dim for the plane + reduction pad
    extern __shared__ float s_data[]; 
    
    int j_current = blockIdx.x; // Current 'j' index this block handles

    if (j_current >= J_dim) return;

    // --- Load A_slice[:, j_current, :] into shared memory s_A_plane ---
    // This plane is non-contiguous in global memory. Careful loading is needed.
    float* s_A_plane = s_data; 
    int plane_size = I_dim * K_dim;
    int tid_in_block = threadIdx.x; // Using 1D block
    int threads_in_block = blockDim.x;

    for (int idx = tid_in_block; idx < plane_size; idx += threads_in_block) {
        // Map linear index 'idx' back to (i, k) within the plane
        int i_load = idx / K_dim;
        int k_load = idx % K_dim;
        
        // Calculate global memory index for A_slice[i_load, j_current, k_load]
        int64_t global_idx = (int64_t)i_load * J_dim * K_dim + (int64_t)j_current * K_dim + k_load;
        
        s_A_plane[idx] = A_slice_global[global_idx];
    }
    __syncthreads(); 

    // --- Perform Reduction for max_val and sum_exp (same as Aq kernel) ---
    float* s_reduction_pad = s_data + plane_size; 
    float thread_max_val = -FLT_MAX; 
    for (int idx = tid_in_block; idx < plane_size; idx += threads_in_block) {
        thread_max_val = fmaxf(thread_max_val, s_A_plane[idx]);
    }
    s_reduction_pad[tid_in_block] = thread_max_val;
    __syncthreads();
    // Reduction for max_val
    for (int offset = threads_in_block / 2; offset > 0; offset >>= 1) {
        if (tid_in_block < offset) {
            s_reduction_pad[tid_in_block] = fmaxf(s_reduction_pad[tid_in_block], s_reduction_pad[tid_in_block + offset]);
        }
        __syncthreads();
    }
    float plane_max_val = s_reduction_pad[0]; 
    __syncthreads(); 

    float thread_sum_exp = 0.0f;
    for (int idx = tid_in_block; idx < plane_size; idx += threads_in_block) {
        thread_sum_exp += expf(s_A_plane[idx] - plane_max_val);
    }
    s_reduction_pad[tid_in_block] = thread_sum_exp;
    __syncthreads();
    // Reduction for sum_exp
    for (int offset = threads_in_block / 2; offset > 0; offset >>= 1) {
        if (tid_in_block < offset) {
            s_reduction_pad[tid_in_block] += s_reduction_pad[tid_in_block + offset];
        }
        __syncthreads();
    }
    float plane_sum_exp = s_reduction_pad[0];
    __syncthreads();

    // --- Compute softmax values and write to global memory ---
    if (plane_sum_exp == 0.0f) plane_sum_exp = 1e-20f; 

    for (int idx = tid_in_block; idx < plane_size; idx += threads_in_block) {
        float softmax_val = expf(s_A_plane[idx] - plane_max_val) / plane_sum_exp;
        
        // Map linear index 'idx' back to (i, k)
        int i_write = idx / K_dim;
        int k_write = idx % K_dim;
        
        // Calculate global memory index for Ar_out[i_write, j_current, k_write]
        int64_t global_idx = (int64_t)i_write * J_dim * K_dim + (int64_t)j_current * K_dim + k_write;
        Ar_out_global[global_idx] = softmax_val;
    }
}


// Kernel to compute As_slice (softmax over i, j for each fixed k)
// Each block processes one 'k' plane.
__global__ void compute_As_slice_kernel(
    const float* __restrict__ A_slice_global, // Input A_slice [I,J,K] (global mem)
    float*       __restrict__ As_out_global,  // Output As_slice [I,J,K] (global mem)
    int I_dim, int J_dim, int K_dim
) {
    // Shared memory: I_dim * J_dim for the plane + reduction pad
    extern __shared__ float s_data[]; 
    
    int k_current = blockIdx.x; // Current 'k' index this block handles

    if (k_current >= K_dim) return;

    // --- Load A_slice[:, :, k_current] into shared memory s_A_plane ---
    // This plane is also non-contiguous.
    float* s_A_plane = s_data; 
    int plane_size = I_dim * J_dim;
    int tid_in_block = threadIdx.x;
    int threads_in_block = blockDim.x;

    for (int idx = tid_in_block; idx < plane_size; idx += threads_in_block) {
        // Map linear index 'idx' back to (i, j) within the plane
        int i_load = idx / J_dim;
        int j_load = idx % J_dim;
        
        // Calculate global memory index for A_slice[i_load, j_load, k_current]
        int64_t global_idx = (int64_t)i_load * J_dim * K_dim + (int64_t)j_load * K_dim + k_current;
        
        s_A_plane[idx] = A_slice_global[global_idx];
    }
    __syncthreads(); 

    // --- Perform Reduction for max_val and sum_exp (same as Aq/Ar kernel) ---
    float* s_reduction_pad = s_data + plane_size; 
    float thread_max_val = -FLT_MAX; 
    for (int idx = tid_in_block; idx < plane_size; idx += threads_in_block) {
        thread_max_val = fmaxf(thread_max_val, s_A_plane[idx]);
    }
    s_reduction_pad[tid_in_block] = thread_max_val;
    __syncthreads();
    for (int offset = threads_in_block / 2; offset > 0; offset >>= 1) {
        if (tid_in_block < offset) {
            s_reduction_pad[tid_in_block] = fmaxf(s_reduction_pad[tid_in_block], s_reduction_pad[tid_in_block + offset]);
        }
        __syncthreads();
    }
    float plane_max_val = s_reduction_pad[0]; 
    __syncthreads(); 

    float thread_sum_exp = 0.0f;
    for (int idx = tid_in_block; idx < plane_size; idx += threads_in_block) {
        thread_sum_exp += expf(s_A_plane[idx] - plane_max_val);
    }
    s_reduction_pad[tid_in_block] = thread_sum_exp;
    __syncthreads();
    for (int offset = threads_in_block / 2; offset > 0; offset >>= 1) {
        if (tid_in_block < offset) {
            s_reduction_pad[tid_in_block] += s_reduction_pad[tid_in_block + offset];
        }
        __syncthreads();
    }
    float plane_sum_exp = s_reduction_pad[0];
    __syncthreads();

    // --- Compute softmax values and write to global memory ---
    if (plane_sum_exp == 0.0f) plane_sum_exp = 1e-20f; 

    for (int idx = tid_in_block; idx < plane_size; idx += threads_in_block) {
        float softmax_val = expf(s_A_plane[idx] - plane_max_val) / plane_sum_exp;
        
        // Map linear index 'idx' back to (i, j)
        int i_write = idx / J_dim;
        int j_write = idx % J_dim;
        
        // Calculate global memory index for As_out[i_write, j_write, k_current]
        int64_t global_idx = (int64_t)i_write * J_dim * K_dim + (int64_t)j_write * K_dim + k_current;
        As_out_global[global_idx] = softmax_val;
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

    // Threads per block: Cover I*K elements. Use similar logic as Aq
    int plane_size = I * K;
    int threads_per_block = 256; // Example fixed size
    // Potentially adjust based on plane_size if needed, ensuring power of 2 for simple reduction.
     if (plane_size < threads_per_block && plane_size > 0) { 
          // Adapt or use fixed size; fixed 256 used here.
     }

    dim3 blockDim(threads_per_block); 

    // Shared memory: I*K floats for s_A_plane + threads_per_block floats for reduction pad
    size_t shared_mem_size = (plane_size + threads_per_block) * sizeof(float);
    
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
    // cudaDeviceSynchronize(); // For debugging

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

    // Threads per block: Cover I*J elements.
    int plane_size = I * J;
    int threads_per_block = 256; // Example fixed size
     if (plane_size < threads_per_block && plane_size > 0) { 
          // Adapt or use fixed size
     }

    dim3 blockDim(threads_per_block); 

    // Shared memory: I*J floats for s_A_plane + threads_per_block floats for reduction pad
    size_t shared_mem_size = (plane_size + threads_per_block) * sizeof(float);
    
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
    // cudaDeviceSynchronize(); // For debugging

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

    // Ensure inputs are contiguous (important for pointer access)
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
    // cudaDeviceSynchronize(); // For debugging

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

__global__ void apply_softmax_backward_kernel(
    // Inputs
    const float* __restrict__ grad_Aq_slice_in, // [I, J, K]
    const float* __restrict__ grad_Ar_slice_in, // [I, J, K]
    const float* __restrict__ grad_As_slice_in, // [I, J, K]
    const float* __restrict__ Aq_slice_in,      // [I, J, K]
    const float* __restrict__ Ar_slice_in,      // [I, J, K]
    const float* __restrict__ As_slice_in,      // [I, J, K]
    // Output
    float* __restrict__ grad_A_slice_out,    // [I, J, K]
    // Dimensions
    int I_dim, int J_dim, int K_dim
) {
    // Map 3D thread indices to (i, j, k) for grad_A_slice_out
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    // Boundary check
    if (i >= I_dim || j >= J_dim || k >= K_dim) {
        return;
    }

    int64_t ijk_idx = (int64_t)i * J_dim * K_dim + (int64_t)j * K_dim + k;
    float final_grad_A_val = 0.0f;

    // --- 2.1 Contribution from Aq (Softmax over j, k for fixed i) ---
    // sum_q = sum_{j',k'} (grad_Aq[i,j',k'] * Aq[i,j',k'])
    // This sum is specific to each 'i'.
    // All threads with the same 'i' (blockIdx.x * blockDim.x + threadIdx.x) participate.
    // We need a reduction across J_dim * K_dim for each 'i'.

    // For simplicity in this kernel, each thread (i,j,k) calculates its part for sum_q, sum_r, sum_s.
    // More optimized: a dedicated reduction kernel or block-wide reduction for each sum_q[i], sum_r[j], sum_s[k].
    // This version is less optimal for the sum_q/r/s but simpler to write initially.
    // It recomputes sums, which is not ideal.

    // --- Contribution from grad_Aq ---
    float sum_q_for_ijk = 0.0f;
    for (int j_prime = 0; j_prime < J_dim; ++j_prime) {
        for (int k_prime = 0; k_prime < K_dim; ++k_prime) {
            int64_t i_jprime_kprime_idx = (int64_t)i * J_dim * K_dim + (int64_t)j_prime * K_dim + k_prime;
            sum_q_for_ijk += grad_Aq_slice_in[i_jprime_kprime_idx] * Aq_slice_in[i_jprime_kprime_idx];
        }
    }
    final_grad_A_val += (grad_Aq_slice_in[ijk_idx] - sum_q_for_ijk) * Aq_slice_in[ijk_idx];


    // --- Contribution from grad_Ar ---
    float sum_r_for_ijk = 0.0f;
    for (int i_prime = 0; i_prime < I_dim; ++i_prime) {
        for (int k_prime = 0; k_prime < K_dim; ++k_prime) {
            int64_t iprime_j_kprime_idx = (int64_t)i_prime * J_dim * K_dim + (int64_t)j * K_dim + k_prime;
            sum_r_for_ijk += grad_Ar_slice_in[iprime_j_kprime_idx] * Ar_slice_in[iprime_j_kprime_idx];
        }
    }
    final_grad_A_val += (grad_Ar_slice_in[ijk_idx] - sum_r_for_ijk) * Ar_slice_in[ijk_idx];
    

    // --- Contribution from grad_As ---
    float sum_s_for_ijk = 0.0f;
    for (int i_prime = 0; i_prime < I_dim; ++i_prime) {
        for (int j_prime = 0; j_prime < J_dim; ++j_prime) {
            int64_t iprime_jprime_k_idx = (int64_t)i_prime * J_dim * K_dim + (int64_t)j_prime * K_dim + k;
            sum_s_for_ijk += grad_As_slice_in[iprime_jprime_k_idx] * As_slice_in[iprime_jprime_k_idx];
        }
    }
    final_grad_A_val += (grad_As_slice_in[ijk_idx] - sum_s_for_ijk) * As_slice_in[ijk_idx];

    grad_A_slice_out[ijk_idx] = final_grad_A_val;
}

torch::Tensor apply_softmax_backward_cuda_wrapper(
    // Inputs (GPU tensors)
    const torch::Tensor& grad_Aq_slice_gpu, // [I, J, K]
    const torch::Tensor& grad_Ar_slice_gpu, // [I, J, K]
    const torch::Tensor& grad_As_slice_gpu, // [I, J, K]
    const torch::Tensor& Aq_slice_gpu,      // [I, J, K]
    const torch::Tensor& Ar_slice_gpu,      // [I, J, K]
    const torch::Tensor& As_slice_gpu       // [I, J, K]
) {
    TORCH_CHECK(grad_Aq_slice_gpu.is_cuda(), "grad_Aq_slice_gpu must be CUDA");
    // ... add checks for other inputs ...
    TORCH_CHECK(grad_Aq_slice_gpu.dim() == 3 && Aq_slice_gpu.dim() == 3, "Inputs must be 3D");
    
    const int I = grad_Aq_slice_gpu.size(0);
    const int J = grad_Aq_slice_gpu.size(1);
    const int K = grad_Aq_slice_gpu.size(2);

    TORCH_CHECK(grad_Ar_slice_gpu.sizes() == grad_Aq_slice_gpu.sizes(), "grad_Ar shape mismatch");
    // ... add other shape consistency checks ...

    auto options = grad_Aq_slice_gpu.options();
    torch::Tensor grad_A_slice_out_gpu = torch::zeros({I, J, K}, options);

    // Kernel launch configuration (maps to each element of grad_A_slice_out)
    constexpr int BLOCK_DIM_I = 8;
    constexpr int BLOCK_DIM_J = 8;
    constexpr int BLOCK_DIM_K = 8; 
    dim3 blockDim(BLOCK_DIM_I, BLOCK_DIM_J, BLOCK_DIM_K);
    dim3 gridDim(
        (I + BLOCK_DIM_I - 1) / BLOCK_DIM_I,
        (J + BLOCK_DIM_J - 1) / BLOCK_DIM_J,
        (K + BLOCK_DIM_K - 1) / BLOCK_DIM_K
    );

    // Ensure inputs are contiguous
    auto grad_Aq_cont = grad_Aq_slice_gpu.contiguous();
    auto grad_Ar_cont = grad_Ar_slice_gpu.contiguous();
    auto grad_As_cont = grad_As_slice_gpu.contiguous();
    auto Aq_cont = Aq_slice_gpu.contiguous();
    auto Ar_cont = Ar_slice_gpu.contiguous();
    auto As_cont = As_slice_gpu.contiguous();

    apply_softmax_backward_kernel<<<gridDim, blockDim>>>(
        grad_Aq_cont.data_ptr<float>(),
        grad_Ar_cont.data_ptr<float>(),
        grad_As_cont.data_ptr<float>(),
        Aq_cont.data_ptr<float>(),
        Ar_cont.data_ptr<float>(),
        As_cont.data_ptr<float>(),
        grad_A_slice_out_gpu.data_ptr<float>(),
        I, J, K
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in apply_softmax_backward_cuda_wrapper: %s\n", cudaGetErrorString(err));
    }
    // cudaDeviceSynchronize(); // For debugging

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

    // --- Write output ---
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

    // --- Write output ---
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

    // Write output
    grad_S[idx] = scale * sum_for_grad_s;
}

// Modify the main backward_cuda function:
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
          torch::Tensor, torch::Tensor,
          torch::Tensor, torch::Tensor,
          torch::Tensor, torch::Tensor>
backward_cuda(
    torch::Tensor grad_output, // GPU
    torch::Tensor Q,           // GPU
    torch::Tensor R,           // GPU
    torch::Tensor S,           // GPU
    torch::Tensor Vq_1,        // GPU
    torch::Tensor Vq_2,        // GPU
    torch::Tensor Vr_1,        // GPU
    torch::Tensor Vr_2,        // GPU
    torch::Tensor Vs_1,        // GPU
    torch::Tensor Vs_2,        // GPU
    double dropout_rate)
{
  // Make sure inputs are contiguous
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
  auto grad_R   = torch::zeros_like(R); // Still a placeholder for now
  auto grad_S   = torch::zeros_like(S); // Still a placeholder for now
  auto grad_Vq_1 = torch::zeros_like(Vq_1);
  auto grad_Vq_2 = torch::zeros_like(Vq_2); 
  auto grad_Vr_1 = torch::zeros_like(Vr_1);
  auto grad_Vr_2 = torch::zeros_like(Vr_2);
  auto grad_Vs_1 = torch::zeros_like(Vs_1);
  auto grad_Vs_2 = torch::zeros_like(Vs_2);

  // 2) extract dims + scale
  const int B = Q.size(0);
  const int H = Q.size(1);
  const int I = Q.size(2); // Slice dim for Q
  const int J = R.size(2); // Slice dim for R
  const int K = S.size(2); // Slice dim for S
  const int D = Q.size(3);
  const int N_grad = grad_output.size(2); 

  const float scale = 1.0f / sqrtf((float)D);
  const int threads = 256; 

  // 3) Launch gather/scatter kernels for grad_V* (these operate on full tensors)
  {
    const int64_t N_kernel_Vq1 = (int64_t)B * H * I * D; 
    const dim3 blocks_Vq1((N_kernel_Vq1 + threads - 1) / threads);
    gather_grad_Vq1_kernel<<<blocks_Vq1, threads>>>(
        grad_output.data_ptr<float>(), Q.data_ptr<float>(), R.data_ptr<float>(), S.data_ptr<float>(),
        Vr_1.data_ptr<float>(), Vs_1.data_ptr<float>(), grad_Vq_1.data_ptr<float>(),
        B, H, I, J, K, D, N_grad, scale);
  }
   { 
      const int64_t N_kernel_Vr1 = (int64_t)B * H * J * D;
      const dim3 blocks_Vr1((N_kernel_Vr1 + threads - 1) / threads);
      gather_grad_Vr1_kernel<<<blocks_Vr1, threads>>>( 
          grad_output.data_ptr<float>(), Q.data_ptr<float>(), R.data_ptr<float>(), S.data_ptr<float>(), 
          Vq_1.data_ptr<float>(), Vs_1.data_ptr<float>(), grad_Vr_1.data_ptr<float>(), 
          B, H, I, J, K, D, N_grad, scale); 
  }
   { 
      const int64_t N_kernel_Vs1 = (int64_t)B * H * K * D;
      const dim3 blocks_Vs1((N_kernel_Vs1 + threads - 1) / threads);
      gather_grad_Vs1_kernel<<<blocks_Vs1, threads>>>( 
          grad_output.data_ptr<float>(), Q.data_ptr<float>(), R.data_ptr<float>(), S.data_ptr<float>(), 
          Vq_1.data_ptr<float>(), Vr_1.data_ptr<float>(), grad_Vs_1.data_ptr<float>(),
          B, H, I, J, K, D, N_grad, scale); 
  }
  { 
      const int64_t N_kernel_Vq2 = (int64_t)B * H * I * D; 
      const dim3 blocks_Vq2((N_kernel_Vq2 + threads - 1) / threads);
      scatter_grad_Vq2_kernel<<<blocks_Vq2, threads>>>( 
          grad_output.data_ptr<float>(), Q.data_ptr<float>(), R.data_ptr<float>(), S.data_ptr<float>(), 
          Vr_2.data_ptr<float>(), Vs_2.data_ptr<float>(), grad_Vq_2.data_ptr<float>(),
          B, H, I, J, K, D, N_grad, scale); 
  }
  { 
      const int64_t N_kernel_Vr2 = (int64_t)B * H * J * D; 
      const dim3 blocks_Vr2((N_kernel_Vr2 + threads - 1) / threads);
      scatter_grad_Vr2_kernel<<<blocks_Vr2, threads>>>( 
          grad_output.data_ptr<float>(), Q.data_ptr<float>(), R.data_ptr<float>(), S.data_ptr<float>(), 
          Vq_2.data_ptr<float>(), Vs_2.data_ptr<float>(), grad_Vr_2.data_ptr<float>(),
          B, H, I, J, K, D, N_grad, scale); 
  }
  { 
      const int64_t N_kernel_Vs2 = (int64_t)B * H * K * D; 
      const dim3 blocks_Vs2((N_kernel_Vs2 + threads - 1) / threads);
      scatter_grad_Vs2_kernel<<<blocks_Vs2, threads>>>( 
          grad_output.data_ptr<float>(), Q.data_ptr<float>(), R.data_ptr<float>(), S.data_ptr<float>(), 
          Vq_2.data_ptr<float>(), Vr_2.data_ptr<float>(), grad_Vs_2.data_ptr<float>(),
          B, H, I, J, K, D, N_grad, scale); 
  }
  // Error check after V-gradient kernels
  cudaError_t v_err = cudaGetLastError();
  if (v_err != cudaSuccess) {
    fprintf(stderr, "CUDA error after V-gradient kernels in backward_cuda: %s\n", cudaGetErrorString(v_err));
  }


  // --- 4. Compute full grad_A tensor on GPU by processing slice by slice ---
  auto grad_A_batched_gpu = torch::zeros({B, H, I, J, K}, Q.options()); // Allocate full grad_A on GPU

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

          // --- Chain of CUDA calls to get final_grad_A_slice_gpu ---
          torch::Tensor A_slice_gpu = compute_A_slice_cuda_wrapper(
              Q_slice_gpu, R_slice_gpu, S_slice_gpu, scale
          );
          
          torch::Tensor Aq_slice_gpu = compute_Aq_slice_cuda_wrapper(A_slice_gpu);
          torch::Tensor Ar_slice_gpu = compute_Ar_slice_cuda_wrapper(A_slice_gpu);
          torch::Tensor As_slice_gpu = compute_As_slice_cuda_wrapper(A_slice_gpu);
          
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

  // --- 5. Launch kernel for full grad_Q using the full grad_A_batched_gpu ---
  {
      const int64_t N_kernel_Q = (int64_t)B * H * I * D; 
      const dim3 blocks_Q((N_kernel_Q + threads - 1) / threads);
      // Note: Using 'grad_Q_kernel' as it's named in your .cu file, assuming it's compute_grad_Q_kernel_from_gradA
      grad_Q_kernel<<<blocks_Q, threads>>>( 
          grad_A_batched_gpu.data_ptr<float>(), 
          R.data_ptr<float>(),                  
          S.data_ptr<float>(),                  
          grad_Q.data_ptr<float>(),             
          B, H, I, J, K, D, scale);      
  }  
  
  cudaError_t gq_err = cudaGetLastError();
  if (gq_err != cudaSuccess) {
    fprintf(stderr, "CUDA error after grad_Q_kernel in backward_cuda: %s\n", cudaGetErrorString(gq_err));
  }

    // --- 6. Launch kernel for full grad_R using the full grad_A_batched_gpu --- (NEW LAUNCH STYLE)
  {
      const int64_t N_kernel_R = (int64_t)B * H * J * D; // Output size for grad_R
      const dim3 blocks_R((N_kernel_R + threads - 1) / threads);
      grad_R_kernel<<<blocks_R, threads>>>( // Direct launch of grad_R_kernel
          grad_A_batched_gpu.data_ptr<float>(), // Full grad_A
          Q.data_ptr<float>(),                  // Full Q
          S.data_ptr<float>(),                  // Full S
          grad_R.data_ptr<float>(),             // Write to pre-allocated grad_R
          B, H, I, J, K, D, scale);
   }
  cudaError_t gr_err = cudaGetLastError();
  if (gr_err != cudaSuccess) {
     fprintf(stderr, "CUDA error after grad_R_kernel in backward_cuda: %s\n", cudaGetErrorString(gr_err));
  }

  // --- 7. Launch kernel for full grad_S using the full grad_A_batched_gpu --- (NEW)
  {
      const int64_t N_kernel_S = (int64_t)B * H * K * D; // Output size for grad_S
      const dim3 blocks_S((N_kernel_S + threads - 1) / threads);
      grad_S_kernel<<<blocks_S, threads>>>( // Direct launch of grad_S_kernel
          grad_A_batched_gpu.data_ptr<float>(), // Full grad_A
          Q.data_ptr<float>(),                  // Full Q
          R.data_ptr<float>(),                  // Full R
          grad_S.data_ptr<float>(),             // Write to pre-allocated grad_S
          B, H, I, J, K, D, scale);
  }
  cudaError_t gs_err = cudaGetLastError();
  if (gs_err != cudaSuccess) {
     fprintf(stderr, "CUDA error after grad_S_kernel in backward_cuda: %s\n", cudaGetErrorString(gs_err));
  }

  // TODO: Implement and launch slice-wise kernels for grad_R, grad_S using final_grad_A_slice_gpu
  //       and accumulate into grad_R, grad_S within the b,h loop.
  //       Or, launch full kernels for grad_R, grad_S using grad_A_batched_gpu after the loop.

  cudaDeviceSynchronize(); // Wait for all kernel completions

  return std::make_tuple(
      grad_Q, grad_R, grad_S,
      grad_Vq_1, grad_Vq_2, 
      grad_Vr_1, grad_Vr_2,
      grad_Vs_1, grad_Vs_2
  );
}


