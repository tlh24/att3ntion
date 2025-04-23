#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>       // optional
#include <cuda.h>
#include <cuda_runtime.h>

// Device helper: computes dot product for Q[i], R[j], S[k] for a specific (b,h)
template <typename scalar_t>
__device__ inline float device_compute_dot_product(
    const scalar_t* __restrict__ q_bh_base,
    const scalar_t* __restrict__ r_bh_base,
    const scalar_t* __restrict__ s_bh_base,
    int i, int j, int k,
    int I_stride, int J_stride, int K_stride, int D)
{
    float dot = 0.0f;
    const scalar_t* q_ptr = q_bh_base + i * I_stride;
    const scalar_t* r_ptr = r_bh_base + j * J_stride;
    const scalar_t* s_ptr = s_bh_base + k * K_stride;
    #pragma unroll
    for (int d = 0; d < D; ++d) {
        dot += (float)q_ptr[d] * (float)r_ptr[d] * (float)s_ptr[d];
    }
    return dot;
}

// Gather kernel for fixed_dim=0 (Y_q = softmax_{j,k}(QRS) * Vr1 * Vs1)
template <typename scalar_t>
__global__ void gather0_kernel(
    const scalar_t* __restrict__ Q, const scalar_t* __restrict__ R, const scalar_t* __restrict__ S,
    const scalar_t* __restrict__ Vr1, const scalar_t* __restrict__ Vs1,
          scalar_t* __restrict__ Yq,
    int B, int H, int I, int J, int K, int D,
    int Q_stride, int R_stride, int S_stride,
    int Vr1_stride, int Vs1_stride, int Yq_stride,
    float scale)
{
    int d = threadIdx.x;
    int i = blockIdx.z;
    int h = blockIdx.y;
    int b = blockIdx.x;

    if (d >= D || i >= I || h >= H || b >= B) return;

    const scalar_t* q_bh = Q + (b * H + h) * I * D;
    const scalar_t* r_bh = R + (b * H + h) * J * D;
    const scalar_t* s_bh = S + (b * H + h) * K * D;
    const scalar_t* vr1_bh = Vr1 + (b * H + h) * J * D;
    const scalar_t* vs1_bh = Vs1 + (b * H + h) * K * D;
    scalar_t* yq_bh = Yq + (b * H + h) * I * D;

    float max_val = -INFINITY;
    for (int j = 0; j < J; ++j) {
        for (int k = 0; k < K; ++k) {
            float dot = device_compute_dot_product(q_bh, r_bh, s_bh, i, j, k, I*D, J*D, K*D, D);
            max_val = max(max_val, dot * scale);
        }
    }

    float sum_exp = 0.0f;
    for (int j = 0; j < J; ++j) {
        for (int k = 0; k < K; ++k) {
            float dot = device_compute_dot_product(q_bh, r_bh, s_bh, i, j, k, I*D, J*D, K*D, D);
            sum_exp += expf(dot * scale - max_val);
        }
    }
    sum_exp = max(sum_exp, 1e-10f);

    float y_val = 0.0f;
    for (int j = 0; j < J; ++j) {
        for (int k = 0; k < K; ++k) {
            float dot = device_compute_dot_product(q_bh, r_bh, s_bh, i, j, k, I*D, J*D, K*D, D);
            float attn = expf(dot * scale - max_val) / sum_exp;
            y_val += attn * (float)vr1_bh[j * D + d] * (float)vs1_bh[k * D + d];
        }
    }

    yq_bh[i * D + d] = (scalar_t)y_val;
}

// Gather kernel for fixed_dim=1 (Y_r = softmax_{i,k}(QRS) * Vq1 * Vs1)
template <typename scalar_t>
__global__ void gather1_kernel(
    const scalar_t* __restrict__ Q, const scalar_t* __restrict__ R, const scalar_t* __restrict__ S,
    const scalar_t* __restrict__ Vq1, const scalar_t* __restrict__ Vs1,
          scalar_t* __restrict__ Yr,
    int B, int H, int I, int J, int K, int D,
    int Q_stride, int R_stride, int S_stride,
    int Vq1_stride, int Vs1_stride, int Yr_stride,
    float scale)
{
    int d = threadIdx.x;
    int j = blockIdx.z;
    int h = blockIdx.y;
    int b = blockIdx.x;

    if (d >= D || j >= J || h >= H || b >= B) return;

    const scalar_t* q_bh = Q + (b * H + h) * I * D;
    const scalar_t* r_bh = R + (b * H + h) * J * D;
    const scalar_t* s_bh = S + (b * H + h) * K * D;
    const scalar_t* vq1_bh = Vq1 + (b * H + h) * I * D;
    const scalar_t* vs1_bh = Vs1 + (b * H + h) * K * D;
    scalar_t* yr_bh = Yr + (b * H + h) * J * D;

    float max_val = -INFINITY;
    for (int i = 0; i < I; ++i) {
        for (int k = 0; k < K; ++k) {
            float dot = device_compute_dot_product(q_bh, r_bh, s_bh, i, j, k, I*D, J*D, K*D, D);
            max_val = max(max_val, dot * scale);
        }
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < I; ++i) {
        for (int k = 0; k < K; ++k) {
            float dot = device_compute_dot_product(q_bh, r_bh, s_bh, i, j, k, I*D, J*D, K*D, D);
            sum_exp += expf(dot * scale - max_val);
        }
    }
    sum_exp = max(sum_exp, 1e-10f);

    float y_val = 0.0f;
    for (int i = 0; i < I; ++i) {
        for (int k = 0; k < K; ++k) {
            float dot = device_compute_dot_product(q_bh, r_bh, s_bh, i, j, k, I*D, J*D, K*D, D);
            float attn = expf(dot * scale - max_val) / sum_exp;
            y_val += attn * (float)vq1_bh[i * D + d] * (float)vs1_bh[k * D + d];
        }
    }

    yr_bh[j * D + d] = (scalar_t)y_val;
}

// Gather kernel for fixed_dim=2 (Y_s = softmax_{i,j}(QRS) * Vq1 * Vr1)
template <typename scalar_t>
__global__ void gather2_kernel(
    const scalar_t* __restrict__ Q, const scalar_t* __restrict__ R, const scalar_t* __restrict__ S,
    const scalar_t* __restrict__ Vq1, const scalar_t* __restrict__ Vr1,
          scalar_t* __restrict__ Ys,
    int B, int H, int I, int J, int K, int D,
    int Q_stride, int R_stride, int S_stride,
    int Vq1_stride, int Vr1_stride, int Ys_stride,
    float scale)
{
    int d = threadIdx.x;
    int k = blockIdx.z;
    int h = blockIdx.y;
    int b = blockIdx.x;

    if (d >= D || k >= K || h >= H || b >= B) return;

    const scalar_t* q_bh = Q + (b * H + h) * I * D;
    const scalar_t* r_bh = R + (b * H + h) * J * D;
    const scalar_t* s_bh = S + (b * H + h) * K * D;
    const scalar_t* vq1_bh = Vq1 + (b * H + h) * I * D;
    const scalar_t* vr1_bh = Vr1 + (b * H + h) * J * D;
    scalar_t* ys_bh = Ys + (b * H + h) * K * D;

    float max_val = -INFINITY;
    for (int i = 0; i < I; ++i) {
        for (int j = 0; j < J; ++j) {
            float dot = device_compute_dot_product(q_bh, r_bh, s_bh, i, j, k, I*D, J*D, K*D, D);
            max_val = max(max_val, dot * scale);
        }
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < I; ++i) {
        for (int j = 0; j < J; ++j) {
            float dot = device_compute_dot_product(q_bh, r_bh, s_bh, i, j, k, I*D, J*D, K*D, D);
            sum_exp += expf(dot * scale - max_val);
        }
    }
    sum_exp = max(sum_exp, 1e-10f);

    float y_val = 0.0f;
    for (int i = 0; i < I; ++i) {
        for (int j = 0; j < J; ++j) {
            float dot = device_compute_dot_product(q_bh, r_bh, s_bh, i, j, k, I*D, J*D, K*D, D);
            float attn = expf(dot * scale - max_val) / sum_exp;
            y_val += attn * (float)vq1_bh[i * D + d] * (float)vr1_bh[j * D + d];
        }
    }

    ys_bh[k * D + d] = (scalar_t)y_val;
}

// the CUDA entry‐point for forward pass
at::Tensor forward_cuda(
    at::Tensor Q, at::Tensor R, at::Tensor S,
    at::Tensor Vq_1, at::Tensor Vq_2,
    at::Tensor Vr_1, at::Tensor Vr_2,
    at::Tensor Vs_1, at::Tensor Vs_2,
    double dropout_rate) // Dropout not used in kernel yet
{
    // Ensure tensors are contiguous and on CUDA
    TORCH_CHECK(Q.is_cuda() && Q.is_contiguous(), "Q must be a contiguous CUDA tensor");
    // Add similar checks for R, S, V* ...

    int B = Q.size(0), H = Q.size(1), I = Q.size(2), D = Q.size(3);
    int J = R.size(2), K = S.size(2);
    auto options = Q.options();

    // Allocate output tensors for gather ops
    auto Y_q = at::zeros({B, H, I, D}, options);
    auto Y_r = at::zeros({B, H, J, D}, options);
    auto Y_s = at::zeros({B, H, K, D}, options);

    // TODO: Allocate output tensors for scatter ops (Y_q_, Y_r_, Y_s_)
    auto Y_q_ = at::zeros({B, H, I, D}, options);
    auto Y_r_ = at::zeros({B, H, J, D}, options);
    auto Y_s_ = at::zeros({B, H, K, D}, options);

    float scale = 1.f / sqrtf(static_cast<float>(D));
    int block_size = std::min(1024, D); // Max threads per block
    dim3 block(block_size);

    // Calculate strides (assuming standard contiguous layout)
    int Q_stride = D, R_stride = D, S_stride = D;
    int Vq1_stride = D, Vs1_stride = D, Yq_stride = D;
    int Vr1_stride = D, Yr_stride = D;
    int Ys_stride = D;
    // Add strides for V2 tensors when implementing scatter

    AT_DISPATCH_FLOATING_TYPES(Q.scalar_type(), "gather_cuda", [&]{
        // Launch gather0 kernel for Y_q
        dim3 grid0(B, H, I);
        gather0_kernel<scalar_t><<<grid0, block>>>(
            Q.data_ptr<scalar_t>(), R.data_ptr<scalar_t>(), S.data_ptr<scalar_t>(),
            Vr_1.data_ptr<scalar_t>(), Vs_1.data_ptr<scalar_t>(),
            Y_q.data_ptr<scalar_t>(),
            B, H, I, J, K, D,
            Q_stride, R_stride, S_stride,
            Vr1_stride, Vs1_stride, Yq_stride,
            scale
        );

        // Launch gather1 kernel for Y_r
        dim3 grid1(B, H, J);
        gather1_kernel<scalar_t><<<grid1, block>>>(
            Q.data_ptr<scalar_t>(), R.data_ptr<scalar_t>(), S.data_ptr<scalar_t>(),
            Vq_1.data_ptr<scalar_t>(), Vs_1.data_ptr<scalar_t>(),
            Y_r.data_ptr<scalar_t>(),
            B, H, I, J, K, D,
            Q_stride, R_stride, S_stride,
            Vq1_stride, Vs1_stride, Yr_stride,
            scale
        );

        // Launch gather2 kernel for Y_s
        dim3 grid2(B, H, K);
        gather2_kernel<scalar_t><<<grid2, block>>>(
            Q.data_ptr<scalar_t>(), R.data_ptr<scalar_t>(), S.data_ptr<scalar_t>(),
            Vq_1.data_ptr<scalar_t>(), Vr_1.data_ptr<scalar_t>(),
            Y_s.data_ptr<scalar_t>(),
            B, H, I, J, K, D,
            Q_stride, R_stride, S_stride,
            Vq1_stride, Vr1_stride, Ys_stride,
            scale
        );

        // TODO: Launch scatter kernels for Y_q_, Y_r_, Y_s_

    });

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    // Return sum of all components (scatter components are zeros for now)
    // We need to pad Y_r and Y_s if J/K < I before summing
    auto Y = Y_q.clone(); // Start with Y_q
    if (J == I) Y += Y_r; else Y.slice(2, 0, J) += Y_r;
    if (K == I) Y += Y_s; else Y.slice(2, 0, K) += Y_s;
    // Add scatter terms when implemented
    // if (I == I) Y += Y_q_;
    // if (J == I) Y += Y_r_; else Y.slice(2, 0, J) += Y_r_;
    // if (K == I) Y += Y_s_; else Y.slice(2, 0, K) += Y_s_;

    return Y;
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

// similarly you'd write backward_cuda(…) and other kernels