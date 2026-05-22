/**
 * @file cuda_bindings.cpp
 * @brief Python bindings for the hypergraph attention CUDA kernels.
 *
 * This file provides the pybind11 interface between Python and the CUDA
 * implementations in cuda/forward.cu and cuda/backward.cu.
 *
 * Copyright (c) 2026 Springtail AI. MIT License.
 */

#include <torch/extension.h>
#include <tuple>
#include <cuda_runtime.h>
#include "cuda_bindings.h"

// =============================================================================
// Python Bindings
// =============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda,
          "Hypergraph Attention forward (returns Y_q, Y_r, Y_s, Y_q_, Y_r_, Y_s_, m_i, l_i, m_j, l_j, m_k, l_k)",
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

    m.def("backward", &backward_cuda,
          "Hypergraph Attention backward (requires pre-computed softmax stats from forward pass)",
          py::arg("grad_Y_q"),
          py::arg("grad_Y_r"),
          py::arg("grad_Y_s"),
          py::arg("grad_Y_q_"),
          py::arg("grad_Y_r_"),
          py::arg("grad_Y_s_"),
          py::arg("Q"),
          py::arg("R"),
          py::arg("S"),
          py::arg("Vq_1"),
          py::arg("Vq_2"),
          py::arg("Vr_1"),
          py::arg("Vr_2"),
          py::arg("Vs_1"),
          py::arg("Vs_2"),
          py::arg("m_i"),
          py::arg("l_i"),
          py::arg("m_j"),
          py::arg("l_j"),
          py::arg("m_k"),
          py::arg("l_k"),
          py::arg("dropout_rate") = 0.0);
}
