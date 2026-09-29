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

// The tensor-core gates fail open into the scalar path, so tests read these
// counters (one per kernel launch) to see which path ran.
namespace att3_tc {

struct State {
    bool fwd_enabled;                       // seeded from ATT3_YQ_TC
    bool bwd_enabled;                       // seeded from ATT3_BWD_TC
    std::atomic<unsigned long long> fwd_launches{0};
    std::atomic<unsigned long long> bwd_launches{0};
    // single_gather_*: launch counts per pass and the last selected tile shape
    // ("D=64 warps=8 bk=32 masked=0 win=0").
    std::atomic<unsigned long long> sg_fwd_launches{0};
    std::atomic<unsigned long long> sg_bwd_anchor_launches{0};
    std::atomic<unsigned long long> sg_bwd_rows_launches{0};
    std::string sg_last_fwd, sg_last_bwd;
    // single_gather_shared_*.
    std::atomic<unsigned long long> sg_shared_fwd_launches{0};
    std::atomic<unsigned long long> sg_shared_bwd_launches{0};
    std::string sg_last_shared_fwd, sg_last_shared_bwd;
    // sg_set_config tile overrides, 0 = default. Backward is set per role: `a`
    // is the Q-owned pass (query = anchor), `r` the R/S-owned passes (query =
    // rows); bwd_dh is the output slice per pass (64, or D for one pass at D=128).
    struct SgConfig {
        int fwd_warps = 0, fwd_bk = 0;
        int bwd_a_warps = 0, bwd_a_bk = 0;
        int bwd_r_warps = 0, bwd_r_bk = 0;
        int bwd_dh = 0;
    } sg_cfg;
};

State& state();

}  // namespace att3_tc

// Single query-gather: one softmax over (j, k) per query i, Y = sum P V_r V_s.
// All five inputs are contiguous CUDA bf16 [B,H,N,D] with D in {64,128} and
// N % 16 == 0; mask is undefined or contiguous CUDA bool [B,N,N] (true =
// visible). Tensor-core only: raises if the device cannot run it. window > 0
// declares the mask a causal window of that width on both key axes
// (i - window < j <= i; window >= N is causal) so tiles outside it are skipped;
// the mask is still applied to every visited cell, and must be given.
void single_gather_check(const std::vector<at::Tensor>& xs, const at::Tensor& mask);

// Returns (Y bf16 [B,H,N,D], m fp32 [B,H,N], l fp32 [B,H,N]); LSE = m + log(l).
std::tuple<at::Tensor, at::Tensor, at::Tensor> single_gather_forward_cuda(
    at::Tensor Q, at::Tensor R, at::Tensor S, at::Tensor Vr, at::Tensor Vs,
    at::Tensor mask, int64_t window);

// Returns (dQ, dR, dS, dVr, dVs), bf16.
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
single_gather_backward_cuda(
    at::Tensor dY, at::Tensor Q, at::Tensor R, at::Tensor S, at::Tensor Vr,
    at::Tensor Vs, at::Tensor Y, at::Tensor m, at::Tensor l, at::Tensor mask,
    int64_t window);

// Shared-KV variant (docs/KERNEL_HISTORY.md): Q [B,Hq,N,128], the four KV
// tensors [B,1,N,128] read in place, window in {16,32,64,128} with its [B,N,N]
// mask. fwd_group / rs_group 0 picks the schedule automatically; 1 runs the
// per-head kernels; 2/4 has one CTA share the KV tiles across that many query
// heads (forward and the R/S-owned backward passes; the Q pass stays per head).
// Backward returns dR/dS/dVr/dVs reduced to [B,1,N,128].
void single_gather_shared_check(const at::Tensor& Q, const std::vector<at::Tensor>& kv,
                                const at::Tensor& mask, int64_t window, int64_t group);
std::tuple<at::Tensor, at::Tensor, at::Tensor> single_gather_shared_forward_cuda(
    at::Tensor Q, at::Tensor R, at::Tensor S, at::Tensor Vr, at::Tensor Vs,
    at::Tensor mask, int64_t window, int64_t fwd_group);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
single_gather_shared_backward_cuda(
    at::Tensor dY, at::Tensor Q, at::Tensor R, at::Tensor S, at::Tensor Vr,
    at::Tensor Vs, at::Tensor Y, at::Tensor m, at::Tensor l, at::Tensor mask,
    int64_t window, int64_t rs_group);
// Grouped kernels (cuda/shared_kv/single_gather_shared.cu); return false when the device
// cannot run them.
bool launch_Y_gather_shared_grouped(
    const at::Tensor& Q, const at::Tensor& R, const at::Tensor& S, const at::Tensor& Vr,
    const at::Tensor& Vs, at::Tensor& Y, at::Tensor& m, at::Tensor& l, const bool* mask,
    int B, int H, int N, float scale, int max_smem_optin, cudaStream_t stream, int win, int G);
bool launch_bwd_rows_shared_grouped(
    const at::Tensor& Q, const at::Tensor& dY, const at::Tensor& R, const at::Tensor& Vr,
    const at::Tensor& S, const at::Tensor& Vs, const at::Tensor& m, const at::Tensor& l,
    const at::Tensor& delta, at::Tensor& pR, at::Tensor& pVr, at::Tensor& pS, at::Tensor& pVs,
    const bool* mask, int B, int H, int N, float scale, int max_smem_optin, cudaStream_t stream,
    int win, int G);

// Returns Y_q, Y_r, Y_s, Y_q_, Y_r_, Y_s_, m_i, l_i, m_j, l_j, m_k, l_k; the
// softmax stats are the ones backward_cuda consumes.
//
// I_valid/J_valid/K_valid are the pre-pad sequence lengths: the gathers mask
// cells at or past them to NEG_INF, so zero-padded slots drop out of the
// denominator and the output matches an unpadded reference. <= 0 (or the
// padded N) disables this. gather_mode 1 runs only the Q-anchored gather.
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

// Backward from forward_cuda's softmax stats. Upstream grads come separately
// for the gather (grad_Y_q/r/s) and scatter (grad_Y_q_/r_/s_) outputs.
//
// Y_q/Y_r/Y_s are the forward's gather outputs; when given (and the scatter
// cotangents are all zero) the tensor-core path computes its Jacobian
// correction sums as rowsum(dY o Y) instead of a full cube pass. Undefined
// tensors keep the scalar path.
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

// fwd_group 0 schedule (cuda/shared_kv/single_gather_shared.cu); sets `implementation`
// to the kernel that ran, returns false if none can.
bool launch_Y_gather_shared_auto(
    const at::Tensor& Q,const at::Tensor& R,const at::Tensor& S,const at::Tensor& Vr,
    const at::Tensor& Vs,at::Tensor& Y,at::Tensor& m,at::Tensor& l,const bool* mask,
    int B,int H,int N,float scale,int optin,cudaStream_t stream,int win,const char*& implementation);
