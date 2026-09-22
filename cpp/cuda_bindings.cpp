/**
 * @file cuda_bindings.cpp
 * @brief Python bindings for the hypergraph attention CUDA kernels.
 *
 * Copyright (c) 2026 Springtail AI. MIT License.
 */

#include <torch/extension.h>
#include <cstdlib>
#include <tuple>
#include <cuda_runtime.h>
#include "cuda_bindings.h"
#include "../cuda/common.cuh"

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tc_launches", []() {
        return std::make_pair(att3_tc::state().fwd_launches.load(),
                              att3_tc::state().bwd_launches.load());
    }, "Cumulative (Y_gather_tc, Bwd_gather_tc) launch counts");

    m.def("sg_dispatch", []() {
        auto& s = att3_tc::state();
        py::dict d;
        d["fwd_launches"] = s.sg_fwd_launches.load();
        d["bwd_anchor_launches"] = s.sg_bwd_anchor_launches.load();
        d["bwd_rows_launches"] = s.sg_bwd_rows_launches.load();
        d["last_fwd"] = s.sg_last_fwd;
        d["last_bwd"] = s.sg_last_bwd;
        d["shared_fwd_launches"] = s.sg_shared_fwd_launches.load();
        d["shared_bwd_launches"] = s.sg_shared_bwd_launches.load();
        d["last_shared_fwd"] = s.sg_last_shared_fwd;
        d["last_shared_bwd"] = s.sg_last_shared_bwd;
        return d;
    }, "single_gather_* launch counts per specialization and last selected tile shapes");

    m.def("single_gather_forward",
        [](at::Tensor Q, at::Tensor R, at::Tensor S, at::Tensor Vr, at::Tensor Vs,
           c10::optional<at::Tensor> mask, int64_t window) {
            return single_gather_forward_cuda(Q, R, S, Vr, Vs,
                                              mask.has_value() ? *mask : at::Tensor(), window);
        },
        "Single query-gather forward (returns Y, m, l; LSE = m + log l). window > 0 "
        "declares the mask an equal causal window of that width (>= N: causal) so "
        "tiles outside it are skipped; the mask is still applied.",
        py::arg("Q"), py::arg("R"), py::arg("S"), py::arg("Vr"), py::arg("Vs"),
        py::arg("mask") = py::none(), py::arg("window") = 0);

    m.def("single_gather_backward",
        [](at::Tensor dY, at::Tensor Q, at::Tensor R, at::Tensor S, at::Tensor Vr,
           at::Tensor Vs, at::Tensor Y, at::Tensor m, at::Tensor l,
           c10::optional<at::Tensor> mask, int64_t window) {
            return single_gather_backward_cuda(dY, Q, R, S, Vr, Vs, Y, m, l,
                                               mask.has_value() ? *mask : at::Tensor(), window);
        },
        "Single query-gather backward (returns dQ, dR, dS, dVr, dVs); window as in forward",
        py::arg("dY"), py::arg("Q"), py::arg("R"), py::arg("S"), py::arg("Vr"),
        py::arg("Vs"), py::arg("Y"), py::arg("m"), py::arg("l"),
        py::arg("mask") = py::none(), py::arg("window") = 0);

    m.def("single_gather_shared_forward", &single_gather_shared_forward_cuda,
        "Experimental shared-KV forward: KV tensors [B,1,N,128] read in place; window 16, 32, 64 or 128 "
        "(default 128) with its mask; fwd_group 1 (native) / 2 / 4 (heads per CTA sharing the KV tiles)",
        py::arg("Q"), py::arg("R"), py::arg("S"), py::arg("Vr"), py::arg("Vs"), py::arg("mask"),
        py::arg("window") = 128, py::arg("fwd_group") = 1);

    m.def("single_gather_shared_backward", &single_gather_shared_backward_cuda,
        "Experimental shared-KV backward: returns dQ [B,Hq,N,128] and dR/dS/dVr/dVs [B,1,N,128] "
        "(per-head bf16 partials summed in fp32); rs_group as fwd_group for the R/S passes",
        py::arg("dY"), py::arg("Q"), py::arg("R"), py::arg("S"), py::arg("Vr"), py::arg("Vs"),
        py::arg("Y"), py::arg("m"), py::arg("l"), py::arg("mask"), py::arg("window") = 128,
        py::arg("rs_group") = 1);

    m.def("sg_set_config", [](py::dict cfg) {
        auto& c = att3_tc::state().sg_cfg;
        for (auto item : cfg) {
            const std::string k = py::cast<std::string>(item.first);
            const int v = py::cast<int>(item.second);
            if (k == "fwd_warps") c.fwd_warps = v;
            else if (k == "fwd_bk") c.fwd_bk = v;
            else if (k == "bwd_a_warps") c.bwd_a_warps = v;
            else if (k == "bwd_a_bk") c.bwd_a_bk = v;
            else if (k == "bwd_r_warps") c.bwd_r_warps = v;
            else if (k == "bwd_r_bk") c.bwd_r_bk = v;
            else if (k == "bwd_dh") c.bwd_dh = v;
            else throw std::invalid_argument("sg_set_config: unknown key " + k);
        }
    }, "Override single_gather_* tile shapes: fwd_warps/fwd_bk, bwd_a_warps/bwd_a_bk "
       "(Q-owned pass), bwd_r_warps/bwd_r_bk (R/S-owned passes), bwd_dh (64 or D). "
       "0 restores the default. Illegal shapes raise at the next call.",
       py::arg("cfg"));

    m.def("sg_get_config", []() {
        const auto& c = att3_tc::state().sg_cfg;
        py::dict d;
        d["fwd_warps"] = c.fwd_warps; d["fwd_bk"] = c.fwd_bk;
        d["bwd_a_warps"] = c.bwd_a_warps; d["bwd_a_bk"] = c.bwd_a_bk;
        d["bwd_r_warps"] = c.bwd_r_warps; d["bwd_r_bk"] = c.bwd_r_bk;
        d["bwd_dh"] = c.bwd_dh;
        return d;
    }, "Current sg_set_config() overrides (0 = default)");

    m.def("sg_tile_ranges", [](int64_t N, int64_t window, int64_t a, int64_t side,
                              int64_t bj, int64_t bk) {
        // The kernels' own bounds code, for the visit-coverage test:
        // [(j0, k_lo, k_hi), ...] per row block of the CTA anchored at `a`.
        std::vector<std::tuple<int, int, int>> out;
        int j_lo = 0, j_hi = 0;
        sg_row_bounds((int)side, (int)a, (int)N, (int)window, (int)bj, j_lo, j_hi);
        for (int j0 = j_lo; j0 < j_hi; j0 += (int)bj) {
            int k_lo = 0, k_hi = 0;
            sg_col_bounds((int)side, (int)a, j0, (int)bj, (int)N, (int)window, (int)bk, k_lo, k_hi);
            out.emplace_back(j0, k_lo, k_hi);
        }
        return out;
    }, "Row blocks and col-tile ranges a single-gather CTA visits (side 0 = query is the "
       "anchor, 1 = queries are the rows)",
       py::arg("N"), py::arg("window"), py::arg("a"), py::arg("side"), py::arg("bj"), py::arg("bk"));

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
           c10::optional<at::Tensor> mask_opt,
           int64_t gather_mode) {
            at::Tensor mask = mask_opt.has_value() ? *mask_opt : at::Tensor();
            return forward_cuda(
                Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2,
                mask, dropout_rate, I_valid, J_valid, K_valid, gather_mode
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
        py::arg("mask") = py::none(),
        py::arg("gather_mode") = 0
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
