
#ifndef MANUAL_ATT3NTION_H
#define MANUAL_ATT3NTION_H

#include <torch/extension.h>
#include <tuple> // Required for std::tuple

// Declare CPU functions defined in manual_att3ntion.cpp

// Computes attention tensors A, Aq, Ar, As for a single batch item and head
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
compute_attention_tensors_single(
    const torch::Tensor& Q_slice, // Shape [I, D]
    const torch::Tensor& R_slice, // Shape [J, D]
    const torch::Tensor& S_slice, // Shape [K, D]
    float scale                   // Scaling factor 1/sqrt(D)
);

// Computes grad_A for a single batch item and head
torch::Tensor compute_grad_A_single(
    const torch::Tensor& grad_output_slice, // Shape [N, D]
    const torch::Tensor& Q_slice,           // [I, D]
    const torch::Tensor& R_slice,           // [J, D]
    const torch::Tensor& S_slice,           // [K, D]
    const torch::Tensor& Vq_1_slice,        // [I, D]
    const torch::Tensor& Vq_2_slice,        // [I, D]
    const torch::Tensor& Vr_1_slice,        // [J, D]
    const torch::Tensor& Vr_2_slice,        // [J, D]
    const torch::Tensor& Vs_1_slice,        // [K, D]
    const torch::Tensor& Vs_2_slice,        // [K, D]
    const torch::Tensor& A_slice,           // [I, J, K]
    const torch::Tensor& Aq_slice,          // [I, J, K]
    const torch::Tensor& Ar_slice,          // [I, J, K]
    const torch::Tensor& As_slice,          // [I, J, K]
    int b, int h
);

// Declare CPU helper defined in manual_att3ntion.cu
// Computes grad_A on CPU and copies back to GPU (temporary)
torch::Tensor compute_grad_A_cpu_and_copy(
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
    float scale
);

#endif // MANUAL_ATT3NTION_H 