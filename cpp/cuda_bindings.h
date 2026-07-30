/**
 * @file cuda_bindings.h
 * @brief Declarations for the hypergraph attention CUDA kernels.
 *
 * Copyright (c) 2026 Springtail AI. MIT License.
 */

#pragma once

#include <torch/extension.h>
#include <tuple>

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
    int64_t K_valid = -1);

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
