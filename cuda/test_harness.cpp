#include <tuple>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../cpp/manual_att3ntion.h"


int main() {
    // Check if CUDA is available
    if (!torch::cuda::is_available()) {
        std::cerr << "CUDA is not available." << std::endl;
        return 1;
    }
    torch::Device device(torch::kCUDA);

    // Define tensor dimensions
    const int B = 1;
    const int H = 2;
    const int I = 16;
    const int J = 16;
    const int K = 16;
    const int D = 32;

    // Create random input tensors on the GPU
    at::Tensor Q = torch::randn({B, H, I, D}, device);
    at::Tensor R = torch::randn({B, H, J, D}, device);
    at::Tensor S = torch::randn({B, H, K, D}, device);
    at::Tensor Vq_1 = torch::randn({B, H, J, D}, device);
    at::Tensor Vq_2 = torch::randn({B, H, K, D}, device);
    at::Tensor Vr_1 = torch::randn({B, H, I, D}, device);
    at::Tensor Vr_2 = torch::randn({B, H, K, D}, device);
    at::Tensor Vs_1 = torch::randn({B, H, I, D}, device);
    at::Tensor Vs_2 = torch::randn({B, H, J, D}, device);

    std::cout << "Input tensors created on CUDA device." << std::endl;

    // Call the forward_cuda function
    auto outputs = forward_cuda(Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2, 0.0);

    // You can now inspect the output tensors
    auto Y_q = std::get<0>(outputs);
    std::cout << "Y_q shape: " << Y_q.sizes() << std::endl;

    std::cout << "Test harness finished." << std::endl;

    return 0;
}
