#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>       // optional
#include <cuda.h>
#include <cuda_runtime.h>

// Forward declarations for kernels (optional but good practice)
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
  auto grad_Q = torch::zeros_like(Q);
  auto grad_R = torch::zeros_like(R);
  auto grad_S = torch::zeros_like(S);
  auto grad_Vq_1 = torch::zeros_like(Vq_1);
  auto grad_Vq_2 = torch::zeros_like(Vq_2);
  auto grad_Vr_1 = torch::zeros_like(Vr_1);
  auto grad_Vr_2 = torch::zeros_like(Vr_2);
  auto grad_Vs_1 = torch::zeros_like(Vs_1);
  auto grad_Vs_2 = torch::zeros_like(Vs_2);

  // TODO: Implement actual gradient computation kernels
  
  return std::make_tuple(
      grad_Q, grad_R, grad_S,
      grad_Vq_1, grad_Vq_2,
      grad_Vr_1, grad_Vr_2,
      grad_Vs_1, grad_Vs_2
  );
}

// Gather for fixed_dim = 0  (i.e. output shape [B,H,I,D])
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

// Gather for fixed_dim = 1 (i.e. output shape [B,H,J,D])
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

// Gather for fixed_dim = 2 (i.e. output shape [B,H,K,D])
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

    const float scale = 1.0f / std::sqrt((float)D);

    // 3. allocate outputs on GPU
    auto opts = Q.options();
    auto Y_q  = torch::zeros({B,H,I,D}, opts);
    auto Y_r  = torch::zeros({B,H,J,D}, opts);
    auto Y_s  = torch::zeros({B,H,K,D}, opts);
    auto Y_q_ = torch::zeros({B,H,I,D}, opts);
    auto Y_r_ = torch::zeros({B,H,J,D}, opts);
    auto Y_s_ = torch::zeros({B,H,K,D}, opts);

    const int threads = 256;

    // GATHER Calls
    // dim=0: Output Y_q[B,H,I,D], iterate j,k. Inputs: Q,R,S, Vr_1, Vs_1
    {
        const int64_t N = (int64_t)B*H*I*D;
        const dim3 blocks((N + threads - 1) / threads);
        gather_dim0_kernel<<<blocks, threads>>>(
            Q.data_ptr<float>(), R.data_ptr<float>(), S.data_ptr<float>(),
            Vr_1.data_ptr<float>(), Vs_1.data_ptr<float>(),
            Y_q.data_ptr<float>(),
            B, H, I, J, K, D, scale);
    }

    // dim=1: Output Y_r[B,H,J,D], iterate i,k. Inputs: R,Q,S, Vq_1, Vs_1
    {
        const int64_t N = (int64_t)B*H*J*D;
        const dim3 blocks((N + threads - 1) / threads);
        gather_dim1_kernel<<<blocks, threads>>>(
            R.data_ptr<float>(), Q.data_ptr<float>(), S.data_ptr<float>(),
            Vq_1.data_ptr<float>(), Vs_1.data_ptr<float>(),
            Y_r.data_ptr<float>(),
            B, H, I, J, K, D, scale);
    }

    // dim=2: Output Y_s[B,H,K,D], iterate i,j. Inputs: S,Q,R, Vq_1, Vr_1
    {
        const int64_t N = (int64_t)B*H*K*D;
        const dim3 blocks((N + threads - 1) / threads);
        gather_dim2_kernel<<<blocks, threads>>>(
            S.data_ptr<float>(), Q.data_ptr<float>(), R.data_ptr<float>(),
            Vq_1.data_ptr<float>(), Vr_1.data_ptr<float>(),
            Y_s.data_ptr<float>(),
            B, H, I, J, K, D, scale);
    }


    cudaDeviceSynchronize(); // Wait for all gather kernels to complete


    // TODO: Implement SCATTER kernels (Y_q_, Y_r_, Y_s_)

    // TODO: Combine outputs (Placeholder - needs scatter results)
    // Remove the erroneous return statement
    // return Y_q; 
    
    // Return the tuple of all intermediate tensors
    return std::make_tuple(Y_q, Y_r, Y_s, Y_q_, Y_r_, Y_s_);
}
