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

// Forward declaration for backward Vq1 kernel
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

// Forward declarations for other backward kernels (add new ones here)
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

// Forward declaration for scatter backward Vq2 kernel
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

// Forward declarations for scatter backward Vr2 and Vs2 kernels
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

// Forward declaration for NEW grad_Q kernel (accepts grad_A)
__global__ void compute_grad_Q_kernel_from_gradA(
    const float* __restrict__ grad_A, // Shape [B, H, I, J, K]
    const float* __restrict__ R,      // Shape [B, H, J, D]
    const float* __restrict__ S,      // Shape [B, H, K, D]
    float*       __restrict__ grad_Q, // Shape [B, H, I, D] - Output
    const int B, const int H, const int I, const int J, const int K, const int D,
    const float scale
);
__global__ void compute_A_slice_kernel(
    const float* __restrict__ Q_slice_global, 
    const float* __restrict__ R_slice_global, 
    const float* __restrict__ S_slice_global, 
    float*       __restrict__ A_out_global,  
    int I, int J, int K, int D,
    float scale
);


// Gather for fixed_dim = 0  (i.e. output shape [B,H,I,D]) i.e. Y_q
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
    //   Q[b,h,i,:]   at Q + (((b*H + h)*I + i)*D)
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

    // write out
    Y[idx] = y_val;
}

// Gather for fixed_dim = 1 (i.e. output shape [B,H,J,D]) i.e. Y_r
// Corresponds to compute_Y_gather(..., Vq_1, Vs_1, 1) in C++
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
    // global thread index [0 .. B*H*J*D)
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)B*H*J*D) return;

    // decode (b,h,j,d) from idx
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

    // --- 1) find max for numerical‐stable softmax over (i,k) ---
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

    // write out
    Y[idx] = y_val;
}

// Gather for fixed_dim = 2 (i.e. output shape [B,H,K,D]) i.e. Y_s
// Corresponds to compute_Y_gather(..., Vq_1, Vr_1, 2) in C++
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

    // --- 1) find max for numerical‐stable softmax over (i,j) ---
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

    // write out
    Y[idx] = y_val;
}

// --- Helper functions for Scatter Kernels ---

// Device helper: Compute dot product Q[i]*R[j]*S[k] for specific indices
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
        // Accumulate directly as float
        dot += q_vec[d] * r_vec[d] * s_vec[d];
    }
    return dot;
}

// Device helper: Compute single softmax attention value Ar[i,j,k] (fixed_dim=1) or As[i,j,k] (fixed_dim=2)
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
    if (fixed_dim == 0) { // Aq: Softmax over j, k for fixed i_target
        for (int j_idx = 0; j_idx < J; ++j_idx) {
            for (int k_idx = 0; k_idx < K; ++k_idx) {
                float dot = compute_dot_product_cuda(Q, R, S, b, h, i_target, j_idx, k_idx, B, H, I, J, K, D);
                max_val = fmaxf(max_val, dot * scale);
            }
        }
    }
    else if (fixed_dim == 1) { // Ar: Softmax over i, k for fixed j_target
        for (int i_idx = 0; i_idx < I; ++i_idx) {
            for (int k_idx = 0; k_idx < K; ++k_idx) {
                float dot = compute_dot_product_cuda(Q, R, S, b, h, i_idx, j_target, k_idx, B, H, I, J, K, D);
                max_val = fmaxf(max_val, dot * scale);
            }
        }
    } else { // fixed_dim == 2: As: Softmax over i, j for fixed k_target
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

// Scatter for fixed_dim = 0 (Output Y_q_)
// Corresponds to compute_Y_scatter_q in C++
__global__ void scatter_dim0_kernel(
    const float* Q,
    const float* R,
    const float* S,
    const float* Vr_2, // Input Value Tensor 1
    const float* Vs_2, // Input Value Tensor 2
    float*       Y_q_, // Output Tensor
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

            // Compute Ar[b,h,i,j,k] (softmax over i',k' for fixed j)
            float attn_ar = compute_single_softmax_attn_cuda(Q, R, S, b, h, i, j, k, B, H, I, J, K, D, scale, 1);
            
            // Compute As[b,h,i,j,k] (softmax over i',j' for fixed k)
            float attn_as = compute_single_softmax_attn_cuda(Q, R, S, b, h, i, j, k, B, H, I, J, K, D, scale, 2);

            // Get value components for the specific d
            float vr2_val = vr2_vec[d];
            float vs2_val = vs2_vec[d];

            // Accumulate: Ar * As * Vr_2 * Vs_2
            accum_val += attn_ar * attn_as * vr2_val * vs2_val;
        }
    }

    // Write the final accumulated value to the output tensor Y_q_
    // Note: This is an assignment, not addition. Assumes Y_q_ is initialized to zero.
    Y_q_[idx] = accum_val;
}

// Scatter for fixed_dim = 1 (Output Y_r_)
// Corresponds to compute_Y_scatter_r in C++
__global__ void scatter_dim1_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ R,
    const float* __restrict__ S,
    const float* __restrict__ Vq_2, // Input Value Tensor 1
    const float* __restrict__ Vs_2, // Input Value Tensor 2
    float*       __restrict__ Y_r_, // Output Tensor
    int B, int H, int I, int J, int K, int D,
    float scale)
{
    // global thread index [0 .. B*H*J*D) 
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)B*H*J*D) return;

    // decode (b,h,j,d) from idx - this thread computes Y_r_[b,h,j,d]
    int d = idx % D;
    int tmp = idx / D;
    int j = tmp % J; // Fixed index for this output element
    tmp = tmp / J;
    int h = tmp % H;
    int b = tmp / H;

    float accum_val = 0.0f;

    // Iterate over the dimensions we sum over (i and k)
    for (int i = 0; i < I; ++i) {
        const float* vq2_vec = Vq_2 + (((int64_t)b * H + h) * I + i) * D;
        for (int k = 0; k < K; ++k) {
            const float* vs2_vec = Vs_2 + (((int64_t)b * H + h) * K + k) * D;

            // Compute Aq[b,h,i,j,k] (softmax over j',k' for fixed i)
            float attn_aq = compute_single_softmax_attn_cuda(Q, R, S, b, h, i, j, k, B, H, I, J, K, D, scale, 0); 
            
            // Compute As[b,h,i,j,k] (softmax over i',j' for fixed k)
            float attn_as = compute_single_softmax_attn_cuda(Q, R, S, b, h, i, j, k, B, H, I, J, K, D, scale, 2);

            // Get value components for the specific d
            float vq2_val = vq2_vec[d];
            float vs2_val = vs2_vec[d];

            // Accumulate: Aq * As * Vq_2 * Vs_2
            accum_val += attn_aq * attn_as * vq2_val * vs2_val;
        }
    }

    // Write the final accumulated value to the output tensor Y_r_
    Y_r_[idx] = accum_val;
}

// Scatter for fixed_dim = 2 (Output Y_s_)
// Corresponds to compute_Y_scatter_s in C++
__global__ void scatter_dim2_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ R,
    const float* __restrict__ S,
    const float* __restrict__ Vq_2, // Input Value Tensor 1
    const float* __restrict__ Vr_2, // Input Value Tensor 2
    float*       __restrict__ Y_s_, // Output Tensor
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

            // Compute Aq[b,h,i,j,k] (softmax over j',k' for fixed i)
            float attn_aq = compute_single_softmax_attn_cuda(Q, R, S, b, h, i, j, k, B, H, I, J, K, D, scale, 0); 
            
            // Compute Ar[b,h,i,j,k] (softmax over i',k' for fixed j)
            float attn_ar = compute_single_softmax_attn_cuda(Q, R, S, b, h, i, j, k, B, H, I, J, K, D, scale, 1);

            // Get value components for the specific d
            float vq2_val = vq2_vec[d];
            float vr2_val = vr2_vec[d];

            // Accumulate: Aq * Ar * Vq_2 * Vr_2
            accum_val += attn_aq * attn_ar * vq2_val * vr2_val;
        }
    }

    // Write the final accumulated value to the output tensor Y_s_
    Y_s_[idx] = accum_val;
}

// the CUDA entry‐point for forward pass
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

    const float scale = 1.0f / sqrtf((float)D); // Use sqrtf for float

    // allocate outputs on GPU
    auto opts = Q.options();
    auto Y_q  = torch::zeros({B,H,I,D}, opts);
    auto Y_r  = torch::zeros({B,H,J,D}, opts);
    auto Y_s  = torch::zeros({B,H,K,D}, opts);
    auto Y_q_ = torch::zeros({B,H,I,D}, opts); // Allocate output for scatter
    auto Y_r_ = torch::zeros({B,H,J,D}, opts);
    auto Y_s_ = torch::zeros({B,H,K,D}, opts);

    const int threads = 256;

    // --- GATHER Calls --- 
    // dim=0
    {
        const int64_t N = (int64_t)B*H*I*D;
        const dim3 blocks((N + threads - 1) / threads);
        gather_dim0_kernel<<<blocks, threads>>>(
            Q.data_ptr<float>(), R.data_ptr<float>(), S.data_ptr<float>(),
            Vr_1.data_ptr<float>(), Vs_1.data_ptr<float>(), // Values Vr_1, Vs_1
            Y_q.data_ptr<float>(), // Output Y_q
            B, H, I, J, K, D, scale);
    }
    // dim=1
    {
        const int64_t N = (int64_t)B*H*J*D;
        const dim3 blocks((N + threads - 1) / threads);
        gather_dim1_kernel<<<blocks, threads>>>(
            R.data_ptr<float>(), Q.data_ptr<float>(), S.data_ptr<float>(), // Query R
            Vq_1.data_ptr<float>(), Vs_1.data_ptr<float>(), // Values Vq_1, Vs_1
            Y_r.data_ptr<float>(), // Output Y_r
            B, H, I, J, K, D, scale);
    }
    // dim=2
    {
        const int64_t N = (int64_t)B*H*K*D;
        const dim3 blocks((N + threads - 1) / threads);
        gather_dim2_kernel<<<blocks, threads>>>(
            S.data_ptr<float>(), Q.data_ptr<float>(), R.data_ptr<float>(), // Query S
            Vq_1.data_ptr<float>(), Vr_1.data_ptr<float>(), // Values Vq_1, Vr_1
            Y_s.data_ptr<float>(), // Output Y_s
            B, H, I, J, K, D, scale);
    }

    // --- SCATTER Calls --- 
    // dim=0: Output Y_q_[B,H,I,D]. Inputs: Q,R,S, Vr_2, Vs_2
    {
        const int64_t N = (int64_t)B*H*I*D;
        const dim3 blocks((N + threads - 1) / threads);
        scatter_dim0_kernel<<<blocks, threads>>>(
            Q.data_ptr<float>(), R.data_ptr<float>(), S.data_ptr<float>(),
            Vr_2.data_ptr<float>(), Vs_2.data_ptr<float>(), // Values Vr_2, Vs_2
            Y_q_.data_ptr<float>(), // Output Y_q_
            B, H, I, J, K, D, scale);
    }
    // dim=1: Output Y_r_[B,H,J,D]. Inputs: Q,R,S, Vq_2, Vs_2
    {
        const int64_t N = (int64_t)B*H*J*D;
        const dim3 blocks((N + threads - 1) / threads);
        scatter_dim1_kernel<<<blocks, threads>>>(
            Q.data_ptr<float>(), R.data_ptr<float>(), S.data_ptr<float>(),
            Vq_2.data_ptr<float>(), Vs_2.data_ptr<float>(), // Values Vq_2, Vs_2
            Y_r_.data_ptr<float>(), // Output Y_r_
            B, H, I, J, K, D, scale);
    }
    // dim=2: Output Y_s_[B,H,K,D]. Inputs: Q,R,S, Vq_2, Vr_2
    {
        const int64_t N = (int64_t)B*H*K*D;
        const dim3 blocks((N + threads - 1) / threads);
        scatter_dim2_kernel<<<blocks, threads>>>(
            Q.data_ptr<float>(), R.data_ptr<float>(), S.data_ptr<float>(),
            Vq_2.data_ptr<float>(), Vr_2.data_ptr<float>(), // Values Vq_2, Vr_2
            Y_s_.data_ptr<float>(), // Output Y_s_
            B, H, I, J, K, D, scale);
    }

    cudaDeviceSynchronize(); // Wait for all kernels to complete
    
    // Return the tuple of all intermediate tensors
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

torch::Tensor compute_grad_A_cpu_and_copy(
    torch::Tensor grad_output_gpu, // GPU tensor
    torch::Tensor Q_gpu,           // GPU tensor
    torch::Tensor R_gpu,           // GPU tensor
    torch::Tensor S_gpu,           // GPU tensor
    torch::Tensor Vq_1_gpu,        // GPU tensor
    torch::Tensor Vq_2_gpu,        // GPU tensor
    torch::Tensor Vr_1_gpu,        // GPU tensor
    torch::Tensor Vr_2_gpu,        // GPU tensor
    torch::Tensor Vs_1_gpu,        // GPU tensor
    torch::Tensor Vs_2_gpu,        // GPU tensor
    float scale
) {
        const int B = Q_gpu.size(0);
        const int H = Q_gpu.size(1);
        // I, J, K are slice dimensions, not full tensor
        // These will be determined when we select slices for (b,h)

        torch::Tensor grad_A_final_cpu_result = torch::zeros({B, H, Q_gpu.size(2), R_gpu.size(2), S_gpu.size(2)}, 
                                                             Q_gpu.options().device(torch::kCPU));

        auto grad_output_cpu_full = grad_output_gpu.cpu();
        auto Q_cpu_full = Q_gpu.cpu();
        auto R_cpu_full = R_gpu.cpu();
        auto S_cpu_full = S_gpu.cpu();
        auto Vq_1_cpu_full = Vq_1_gpu.cpu();
        auto Vq_2_cpu_full = Vq_2_gpu.cpu();
        auto Vr_1_cpu_full = Vr_1_gpu.cpu();
        auto Vr_2_cpu_full = Vr_2_gpu.cpu();
        auto Vs_1_cpu_full = Vs_1_gpu.cpu();
        auto Vs_2_cpu_full = Vs_2_gpu.cpu();

        for (int b = 0; b < B; ++b) {
            for (int h = 0; h < H; ++h) {
                // GPU Slices for Q,R,S for A_slice computation
                auto Q_slice_gpu_current = Q_gpu.select(0, b).select(0, h);
                auto R_slice_gpu_current = R_gpu.select(0, b).select(0, h);
                auto S_slice_gpu_current = S_gpu.select(0, b).select(0, h);

                // CPU Slices for reference computation and remaining parts of grad_A_single
                auto Q_slice_cpu_ref = Q_cpu_full.select(0, b).select(0, h);
                auto R_slice_cpu_ref = R_cpu_full.select(0, b).select(0, h);
                auto S_slice_cpu_ref = S_cpu_full.select(0, b).select(0, h);
                auto grad_output_slice_cpu = grad_output_cpu_full.select(0, b).select(0, h);
                auto Vq_1_slice_cpu = Vq_1_cpu_full.select(0, b).select(0, h);
                auto Vq_2_slice_cpu = Vq_2_cpu_full.select(0, b).select(0, h);
                auto Vr_1_slice_cpu = Vr_1_cpu_full.select(0, b).select(0, h);
                auto Vr_2_slice_cpu = Vr_2_cpu_full.select(0, b).select(0, h);
                auto Vs_1_slice_cpu = Vs_1_cpu_full.select(0, b).select(0, h);
                auto Vs_2_slice_cpu = Vs_2_cpu_full.select(0, b).select(0, h);

                // Step 1: Compute A_slice on GPU
                torch::Tensor A_slice_from_gpu = compute_A_slice_cuda_wrapper(
                    Q_slice_gpu_current, R_slice_gpu_current, S_slice_gpu_current, scale
                );
                torch::Tensor A_slice_cpu_variant = A_slice_from_gpu.cpu();

                // Step 2: Compute Aq_slice on GPU using A_slice_from_gpu
                // torch::Tensor Aq_slice_from_gpu = compute_Aq_slice_cuda_wrapper(A_slice_from_gpu);
                // torch::Tensor Aq_slice_cpu_variant = Aq_slice_from_gpu.cpu();

                // Compute reference A_slice, Aq_slice, Ar_slice, As_slice on CPU for comparison and remaining inputs
                torch::Tensor A_slice_ref_cpu, Aq_slice_ref_cpu, Ar_slice_ref_cpu, As_slice_ref_cpu;
                std::tie(A_slice_ref_cpu, Aq_slice_ref_cpu, Ar_slice_ref_cpu, As_slice_ref_cpu) =
                    compute_attention_tensors_single(Q_slice_cpu_ref, R_slice_cpu_ref, S_slice_cpu_ref, scale);

                // ----> Verification Points <----
                if (!torch::allclose(A_slice_cpu_variant, A_slice_ref_cpu, 1e-4, 1e-5)) { // Adjusted tolerance
                           fprintf(stderr, "CUDA DEBUG: Mismatch in A_slice for b=%d, h=%d. Max diff: %e\n", 
                                   b, h, (A_slice_cpu_variant - A_slice_ref_cpu).abs().max().item<float>());
                       }

                // Print a success message if they match for a specific b,h for easier debugging
                // else if (b==0 && h==0) {
                //      fprintf(stdout, "CUDA DEBUG: A_slice and Aq_slice MATCH for b=0, h=0!\n");
                // }


                // Call compute_grad_A_single using GPU-computed A_slice and Aq_slice,
                // but CPU-computed Ar_slice and As_slice for this incremental test.
                auto grad_A_slice_for_final_result = compute_grad_A_single(
                    grad_output_slice_cpu, 
                    Q_slice_cpu_ref, R_slice_cpu_ref, S_slice_cpu_ref,
                    Vq_1_slice_cpu, Vq_2_slice_cpu, Vr_1_slice_cpu, Vr_2_slice_cpu, Vs_1_slice_cpu, Vs_2_slice_cpu,
                    A_slice_cpu_variant,    // <<< GPU-computed A_slice (via CPU)
                    Aq_slice_ref_cpu,   //
                    Ar_slice_ref_cpu,       // CPU-computed Ar_slice
                    As_slice_ref_cpu,       // CPU-computed As_slice
                    b, h
                );
                
               grad_A_final_cpu_result.select(0, b).select(0, h).copy_(grad_A_slice_for_final_result);
            }
        }
       return grad_A_final_cpu_result.to(torch::kCUDA);
}

// Minimal backward function that returns zero gradients[placeholder]
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
          torch::Tensor, torch::Tensor,
          torch::Tensor, torch::Tensor,
          torch::Tensor, torch::Tensor>
backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor Q, torch::Tensor R, torch::Tensor S,
    torch::Tensor Vq_1, torch::Tensor Vq_2,
    torch::Tensor Vr_1, torch::Tensor Vr_2,
    torch::Tensor Vs_1, torch::Tensor Vs_2,
    double dropout_rate)
{
  // Make sure inputs are contiguous (good practice, though PyTorch often handles this)
  grad_output = grad_output.contiguous();
  Q = Q.contiguous();  
  R = R.contiguous();  
  S = S.contiguous();
  Vq_1 = Vq_1.contiguous();
  Vq_2 = Vq_2.contiguous(); // Ensure Vq_2 is contiguous
  Vr_1 = Vr_1.contiguous();
  Vr_2 = Vr_2.contiguous(); // Ensure Vr_2 is contiguous
  Vs_1 = Vs_1.contiguous();
  Vs_2 = Vs_2.contiguous(); // Ensure Vs_2 is contiguous

  // 1) allocate outputs
  auto grad_Q   = torch::zeros_like(Q);
  auto grad_R   = torch::zeros_like(R);
  auto grad_S   = torch::zeros_like(S);
  auto grad_Vq_1 = torch::zeros_like(Vq_1);
  auto grad_Vq_2 = torch::zeros_like(Vq_2); // Allocate the target gradient tensor
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
  const int N_grad = grad_output.size(2); // Use the provided grad_output size

  const float scale = 1.0f / sqrtf((float)D); // Use sqrtf for float
  const int threads = 256; // Standard block size

  // 3) Launch the kernel for grad_Vq_1
  // The kernel computes grad_Vq_1 which has shape [B, H, I, D]
  {
    const int64_t N_kernel = (int64_t)B * H * I * D; // Total number of elements in grad_Vq_1
    const dim3 blocks((N_kernel + threads - 1) / threads); // Calculate number of blocks needed

    gather_grad_Vq1_kernel<<<blocks, threads>>>(
        grad_output.data_ptr<float>(), // Combined gradient input
        Q.data_ptr<float>(),
        R.data_ptr<float>(),
        S.data_ptr<float>(),
        Vr_1.data_ptr<float>(),        // Value tensor needed for Y_s path
        Vs_1.data_ptr<float>(),        // Value tensor needed for Y_r path
        grad_Vq_1.data_ptr<float>(),   // Output gradient tensor
        B, H, I, J, K, D, N_grad, scale); // Pass dimensions and scale
  }

  // Launch kernel for grad_Vr_1 (Gather)
  {
      const int64_t N_kernel = (int64_t)B * H * J * D;
      const dim3 blocks((N_kernel + threads - 1) / threads);
      // Assuming gather_grad_Vr1_kernel exists and is implemented
      gather_grad_Vr1_kernel<<<blocks, threads>>>( 
          grad_output.data_ptr<float>(), 
          Q.data_ptr<float>(), 
          R.data_ptr<float>(), 
          S.data_ptr<float>(), 
          Vq_1.data_ptr<float>(), // Vq_1 is needed
          Vs_1.data_ptr<float>(), // Vs_1 is needed
          grad_Vr_1.data_ptr<float>(), // Output 
          B, H, I, J, K, D, N_grad, scale); 
  }

  // Launch kernel for grad_Vs_1 (Gather)
  {
      const int64_t N_kernel = (int64_t)B * H * K * D;
      const dim3 blocks((N_kernel + threads - 1) / threads);
      // Assuming gather_grad_Vs1_kernel exists and is implemented
      gather_grad_Vs1_kernel<<<blocks, threads>>>( 
          grad_output.data_ptr<float>(), 
          Q.data_ptr<float>(), 
          R.data_ptr<float>(), 
          S.data_ptr<float>(), 
          Vq_1.data_ptr<float>(), // Vq_1 is needed
          Vr_1.data_ptr<float>(), // Vr_1 is needed
          grad_Vs_1.data_ptr<float>(), // Output 
          B, H, I, J, K, D, N_grad, scale); 
  }

  // Launch kernel for grad_Vq_2 (Scatter)
  {
      const int64_t N_kernel = (int64_t)B * H * I * D; 
      const dim3 blocks((N_kernel + threads - 1) / threads);
      scatter_grad_Vq2_kernel<<<blocks, threads>>>( 
          grad_output.data_ptr<float>(), // Upstream grads dL/dY_r_, dL/dY_s_ 
          Q.data_ptr<float>(), 
          R.data_ptr<float>(), 
          S.data_ptr<float>(), 
          Vr_2.data_ptr<float>(),        // Value tensor needed for Y_s_ path
          Vs_2.data_ptr<float>(),        // Value tensor needed for Y_r_ path
          grad_Vq_2.data_ptr<float>(),   // Output gradient tensor
          B, H, I, J, K, D, N_grad, scale); // Pass dimensions and scale
  }
  
  // Launch kernel for grad_Vr_2 (Scatter)
  {
      const int64_t N_kernel = (int64_t)B * H * J * D; 
      const dim3 blocks((N_kernel + threads - 1) / threads);
      scatter_grad_Vr2_kernel<<<blocks, threads>>>( 
          grad_output.data_ptr<float>(), 
          Q.data_ptr<float>(), 
          R.data_ptr<float>(), 
          S.data_ptr<float>(), 
          Vq_2.data_ptr<float>(),        // Value tensor needed for Y_s_ path
          Vs_2.data_ptr<float>(),        // Value tensor needed for Y_q_ path
          grad_Vr_2.data_ptr<float>(),   // Output gradient tensor
          B, H, I, J, K, D, N_grad, scale); // Pass dimensions and scale
  }

  // Launch kernel for grad_Vs_2 (Scatter)
  {
      const int64_t N_kernel = (int64_t)B * H * K * D; 
      const dim3 blocks((N_kernel + threads - 1) / threads);
      scatter_grad_Vs2_kernel<<<blocks, threads>>>( 
          grad_output.data_ptr<float>(), 
          Q.data_ptr<float>(), 
          R.data_ptr<float>(), 
          S.data_ptr<float>(), 
          Vq_2.data_ptr<float>(),        // Value tensor needed for Y_r_ path
          Vr_2.data_ptr<float>(),        // Value tensor needed for Y_q_ path
          grad_Vs_2.data_ptr<float>(),   // Output gradient tensor
          B, H, I, J, K, D, N_grad, scale); // Pass dimensions and scale
  }

// --- 4. Compute grad_A (Temporarily using CPU + Copy) ---
  // This is inefficient but allows testing the grad_Q kernel in isolation first.
  // In the future, grad_A should be computed by a dedicated CUDA kernel.
  torch::Tensor grad_A_gpu = compute_grad_A_cpu_and_copy(
      grad_output, Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2, scale
  );

  // --- 5. Launch kernel for grad_Q using the computed grad_A ---
 {
      const int64_t N_kernel = (int64_t)B * H * I * D; // Output size
      const dim3 blocks((N_kernel + threads - 1) / threads);
      grad_Q_kernel<<<blocks, threads>>>(
          grad_A_gpu.data_ptr<float>(),  // Input grad_A [B,H,I,J,K]
          R.data_ptr<float>(),           // Input R [B,H,J,D]
          S.data_ptr<float>(),           // Input S [B,H,K,D]
          grad_Q.data_ptr<float>(),      // Output grad_Q [B,H,I,D]
          B, H, I, J, K, D, scale);      // Dimensions and scale
  }


  // TODO: Implement and launch kernels for grad_R, grad_S

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error in backward_cuda: %s\\n", cudaGetErrorString(err));
  }
  
  cudaDeviceSynchronize(); // Wait for kernel completion

  return std::make_tuple(
      grad_Q, grad_R, grad_S,
      grad_Vq_1, grad_Vq_2, 
      grad_Vr_1, grad_Vr_2,
      grad_Vs_1, grad_Vs_2 // Return all computed gradients
  );
}
