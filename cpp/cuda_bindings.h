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
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
           at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
forward_cuda(
    at::Tensor Q, at::Tensor R, at::Tensor S,
    at::Tensor Vq_1, at::Tensor Vq_2,
    at::Tensor Vr_1, at::Tensor Vr_2,
    at::Tensor Vs_1, at::Tensor Vs_2,
    double dropout_rate = 0.0);

// Backward pass using pre-computed softmax stats from forward pass.
// This is the only backward API - stats must come from forward pass to ensure
// numerical consistency and avoid redundant O(N²) computation.
std::tuple<at::Tensor, at::Tensor, at::Tensor,
           at::Tensor, at::Tensor, at::Tensor,
           at::Tensor, at::Tensor, at::Tensor>
backward_cuda(
    at::Tensor grad_output,
    at::Tensor Q, at::Tensor R, at::Tensor S,
    at::Tensor Vq_1, at::Tensor Vq_2,
    at::Tensor Vr_1, at::Tensor Vr_2,
    at::Tensor Vs_1, at::Tensor Vs_2,
    at::Tensor m_i, at::Tensor l_i,
    at::Tensor m_j, at::Tensor l_j,
    at::Tensor m_k, at::Tensor l_k,
    double dropout_rate = 0.0);
