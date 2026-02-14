/**
 * @file torch_reference.cpp
 * @brief Pure PyTorch C++ reference implementation of hypergraph attention.
 *
 * Uses torch::einsum for clarity. This is the ground truth implementation
 * for testing the CUDA kernels against.
 *
 * Copyright (c) 2026 Springtail AI. MIT License.
 */

#include <torch/extension.h>
#include <cmath>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
hyper_attn_forward(
    torch::Tensor Q,
    torch::Tensor R,
    torch::Tensor S,
    torch::Tensor Vq_1,
    torch::Tensor Vq_2,
    torch::Tensor Vr_1,
    torch::Tensor Vr_2,
    torch::Tensor Vs_1,
    torch::Tensor Vs_2,
    double dropout_rate = 0.0)
{
    int head_dim = Q.size(3);
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    auto dot_product = torch::einsum("bhid,bhjd,bhkd->bhijk", {Q, R, S}) * scale;

    // Aq: softmax over (j,k) for each i
    auto Aq = torch::softmax(dot_product.flatten(3, 4), -1).reshape_as(dot_product);

    // Ar: softmax over (i,k) for each j
    auto dot_product_r = dot_product.permute({0, 1, 3, 2, 4});
    auto Ar = torch::softmax(dot_product_r.flatten(3, 4), -1).reshape_as(dot_product_r);
    Ar = Ar.permute({0, 1, 3, 2, 4});

    // As: softmax over (i,j) for each k
    auto dot_product_s = dot_product.permute({0, 1, 4, 2, 3});
    auto As = torch::softmax(dot_product_s.flatten(3, 4), -1).reshape_as(dot_product_s);
    As = As.permute({0, 1, 3, 4, 2});

    // Gather outputs
    auto Y_q = torch::einsum("bhijk,bhjd,bhkd->bhid", {Aq, Vr_1, Vs_1});
    auto Y_r = torch::einsum("bhijk,bhid,bhkd->bhjd", {Ar, Vq_1, Vs_1});
    auto Y_s = torch::einsum("bhijk,bhid,bhjd->bhkd", {As, Vq_1, Vr_1});

    // Scatter outputs
    auto Y_q_ = torch::einsum("bhijk,bhjd,bhkd->bhid", {Ar * As, Vr_2, Vs_2});
    auto Y_r_ = torch::einsum("bhijk,bhid,bhkd->bhjd", {Aq * As, Vq_2, Vs_2});
    auto Y_s_ = torch::einsum("bhijk,bhid,bhjd->bhkd", {Aq * Ar, Vq_2, Vr_2});

    return std::make_tuple(Y_q, Y_r, Y_s, Y_q_, Y_r_, Y_s_);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &hyper_attn_forward,
          "Hypergraph Attention forward (returns Y_q, Y_r, Y_s, Y_q_, Y_r_, Y_s_)",
          py::arg("Q"),
          py::arg("R"),
          py::arg("S"),
          py::arg("Vq_1"),
          py::arg("Vq_2"),
          py::arg("Vr_1"),
          py::arg("Vr_2"),
          py::arg("Vs_1"),
          py::arg("Vs_2"),
          py::arg("dropout_rate") = 0.0);
}
