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
#include <cstdlib>
#include <tuple>
#include <cuda_runtime.h>
#include "cuda_bindings.h"

namespace att3_tc {

State& state() {
    static State s = []() {
        auto enabled = [](const char* name) {
            const char* e = std::getenv(name);
            return !(e && e[0] == '0');
        };
        return State{enabled("ATT3_YQ_TC"), enabled("ATT3_BWD_TC")};
    }();
    return s;
}

}  // namespace att3_tc

// =============================================================================
// Python Bindings
// =============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tc_launches", []() {
        return std::make_pair(att3_tc::state().fwd_launches.load(),
                              att3_tc::state().bwd_launches.load());
    }, "Cumulative (Y_gather_tc, Bwd_gather_tc) launch counts");

    m.def("tc_set_enabled", [](bool forward, bool backward) {
        auto prev = std::make_pair(att3_tc::state().fwd_enabled,
                                   att3_tc::state().bwd_enabled);
        att3_tc::state().fwd_enabled = forward;
        att3_tc::state().bwd_enabled = backward;
        return prev;
    }, "Set both TC gates, returning the previous (forward, backward)",
       py::arg("forward"), py::arg("backward"));

    m.def(
        "forward",
        [](at::Tensor Q, at::Tensor R, at::Tensor S,
           at::Tensor Vq_1, at::Tensor Vq_2,
           at::Tensor Vr_1, at::Tensor Vr_2,
           at::Tensor Vs_1, at::Tensor Vs_2,
           double dropout_rate,
           int64_t I_valid,
           int64_t J_valid,
           int64_t K_valid,
           c10::optional<at::Tensor> mask_opt) {
            at::Tensor mask = mask_opt.has_value() ? *mask_opt : at::Tensor();
            return forward_cuda(
                Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2,
                mask, dropout_rate, I_valid, J_valid, K_valid
            );
        },
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
        py::arg("dropout_rate") = 0.0,
        py::arg("I_valid") = -1,
        py::arg("J_valid") = -1,
        py::arg("K_valid") = -1,
        py::arg("mask") = py::none()
    );

    m.def(
        "backward",
        [](at::Tensor grad_Y_q, at::Tensor grad_Y_r, at::Tensor grad_Y_s,
           at::Tensor grad_Y_q_, at::Tensor grad_Y_r_, at::Tensor grad_Y_s_,
           at::Tensor Q, at::Tensor R, at::Tensor S,
           at::Tensor Vq_1, at::Tensor Vq_2,
           at::Tensor Vr_1, at::Tensor Vr_2,
           at::Tensor Vs_1, at::Tensor Vs_2,
           at::Tensor m_i, at::Tensor l_i,
           at::Tensor m_j, at::Tensor l_j,
           at::Tensor m_k, at::Tensor l_k,
           double dropout_rate,
           c10::optional<at::Tensor> mask_opt,
           c10::optional<at::Tensor> Y_q_opt,
           c10::optional<at::Tensor> Y_r_opt,
           c10::optional<at::Tensor> Y_s_opt) {
            at::Tensor mask = mask_opt.has_value() ? *mask_opt : at::Tensor();
            at::Tensor Y_q = Y_q_opt.has_value() ? *Y_q_opt : at::Tensor();
            at::Tensor Y_r = Y_r_opt.has_value() ? *Y_r_opt : at::Tensor();
            at::Tensor Y_s = Y_s_opt.has_value() ? *Y_s_opt : at::Tensor();
            return backward_cuda(
                grad_Y_q, grad_Y_r, grad_Y_s, grad_Y_q_, grad_Y_r_, grad_Y_s_,
                Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2,
                m_i, l_i, m_j, l_j, m_k, l_k, mask, dropout_rate,
                Y_q, Y_r, Y_s
            );
        },
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
        py::arg("dropout_rate") = 0.0,
        py::arg("mask") = py::none(),
        py::arg("Y_q") = py::none(),
        py::arg("Y_r") = py::none(),
        py::arg("Y_s") = py::none()
    );
}
