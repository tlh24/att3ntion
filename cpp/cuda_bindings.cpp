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

    m.def("backward", &backward_cuda,
          "Hypergraph Attention backward (returns dQ, dR, dS, dVq_1, dVq_2, dVr_1, dVr_2, dVs_1, dVs_2)",
          py::arg("grad_output"),
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
