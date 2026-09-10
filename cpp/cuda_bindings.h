/**
 * @file cuda_bindings.h
 * @brief Declarations for the hypergraph attention CUDA kernels.
 *
 * Copyright (c) 2026 Springtail AI. MIT License.
 */

#pragma once

#include <torch/extension.h>
#include <atomic>
#include <string>
#include <tuple>
#include <vector>

// Both tensor-core gates fail open into the scalar path, so tests need to see
// which one ran. Counters are per launch: an engaged pass adds 3, one per role.
namespace att3_tc {

struct State {
    bool fwd_enabled;                       // seeded from ATT3_YQ_TC
    bool bwd_enabled;                       // seeded from ATT3_BWD_TC
    std::atomic<unsigned long long> fwd_launches{0};
    std::atomic<unsigned long long> bwd_launches{0};
    // single_gather_* entry points: launch counts per specialization and the
    // tile shape each last selected ("D=64 warps=8 bk=32 masked=0").
    std::atomic<unsigned long long> sg_fwd_launches{0};
    std::atomic<unsigned long long> sg_bwd_anchor_launches{0};
    std::atomic<unsigned long long> sg_bwd_rows_launches{0};
    std::string sg_last_fwd, sg_last_bwd;
};

State& state();

}  // namespace att3_tc

// Single query-gather: one softmax over (j, k) per query i, Y = sum P V_r V_s.
// All five inputs are contiguous CUDA bf16 [B,H,N,D] with D in {64,128} and
// N % 16 == 0; mask is undefined or contiguous CUDA bool [B,N,N] (true =
// visible). Both run only the tensor-core kernels and raise if the device
// cannot (no scalar fallback).
void single_gather_check(const std::vector<at::Tensor>& xs, const at::Tensor& mask);

// Returns (Y bf16 [B,H,N,D], m fp32 [B,H,N], l fp32 [B,H,N]); LSE = m + log(l).
std::tuple<at::Tensor, at::Tensor, at::Tensor> single_gather_forward_cuda(
    at::Tensor Q, at::Tensor R, at::Tensor S, at::Tensor Vr, at::Tensor Vs,
    at::Tensor mask);

// Returns (dQ, dR, dS, dVr, dVs), bf16.
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
single_gather_backward_cuda(
    at::Tensor dY, at::Tensor Q, at::Tensor R, at::Tensor S, at::Tensor Vr,
    at::Tensor Vs, at::Tensor Y, at::Tensor m, at::Tensor l, at::Tensor mask);

// Forward pass returns: Y_q, Y_r, Y_s, Y_q_, Y_r_, Y_s_, m_i, l_i, m_j, l_j, m_k, l_k
// The softmax stats (m_i, l_i, m_j, l_j, m_k, l_k) are computed during forward and
// must be saved and passed to backward_cuda to avoid redundant computation.
//
// I_valid/J_valid/K_valid are the *original* (pre-pad) sequence lengths. The
// gather kernels mask softmax cells with j_global >= J_valid or k_global >=
// K_valid (etc.) to NEG_INF, so zero-padded slots drop out of the denominator
// and the output matches an unpadded reference. Pass <= 0 (or the padded N) to
// disable masking and get the legacy behavior.
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
           at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
forward_cuda(
    at::Tensor Q, at::Tensor R, at::Tensor S,
    at::Tensor Vq_1, at::Tensor Vq_2,
    at::Tensor Vr_1, at::Tensor Vr_2,
    at::Tensor Vs_1, at::Tensor Vs_2,
    at::Tensor mask,
    double dropout_rate = 0.0,
    int64_t I_valid = -1,
    int64_t J_valid = -1,
    int64_t K_valid = -1,
    int64_t gather_mode = 0);

// Backward pass using pre-computed softmax stats from forward pass.
// This is the only backward API - stats must come from forward pass to ensure
// numerical consistency and avoid redundant O(N²) computation.
// Upstream grads include gather (grad_Y_q/r/s) and scatter (grad_Y_q_/r_/s_)
// branches separately.
//
// Y_q/Y_r/Y_s are the forward's gather outputs; when provided (and the scatter
// cotangents are all zero) the tensor-core fast path computes its Jacobian
// correction sums as rowsum(dY o Y) instead of a full cube pass. Passing
// undefined tensors keeps the scalar path.
std::tuple<at::Tensor, at::Tensor, at::Tensor,
           at::Tensor, at::Tensor, at::Tensor,
           at::Tensor, at::Tensor, at::Tensor>
backward_cuda(
    at::Tensor grad_Y_q,
    at::Tensor grad_Y_r,
    at::Tensor grad_Y_s,
    at::Tensor grad_Y_q_,
    at::Tensor grad_Y_r_,
    at::Tensor grad_Y_s_,
    at::Tensor Q, at::Tensor R, at::Tensor S,
    at::Tensor Vq_1, at::Tensor Vq_2,
    at::Tensor Vr_1, at::Tensor Vr_2,
    at::Tensor Vs_1, at::Tensor Vs_2,
    at::Tensor m_i, at::Tensor l_i,
    at::Tensor m_j, at::Tensor l_j,
    at::Tensor m_k, at::Tensor l_k,
    at::Tensor mask,
    double dropout_rate = 0.0,
    at::Tensor Y_q = at::Tensor(),
    at::Tensor Y_r = at::Tensor(),
    at::Tensor Y_s = at::Tensor());
