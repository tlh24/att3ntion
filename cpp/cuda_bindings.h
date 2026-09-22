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
    // Experimental shared-KV entry points (single_gather_shared_*).
    std::atomic<unsigned long long> sg_shared_fwd_launches{0};
    std::atomic<unsigned long long> sg_shared_bwd_launches{0};
    std::string sg_last_shared_fwd, sg_last_shared_bwd;
    // Tile-shape overrides for the single_gather_* kernels (sg_set_config);
    // 0 keeps the built-in default. Backward is tuned per role: `a` is the
    // Q-owned pass (query = anchor), `r` the R/S-owned passes (query = rows);
    // bwd_dh is the output slice per pass (64, or D for one pass at D=128).
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
// visible). Both run only the tensor-core kernels and raise if the device
// cannot (no scalar fallback). window > 0 is the caller's statement that the
// mask is an equal causal window of that width (i - window < j <= i on both
// key axes; window >= N is causal), which lets the kernels skip tiles outside
// it; the mask is still applied to every visited cell. It requires the mask.
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

// Experimental shared-KV prototype (docs/KERNEL_HISTORY.md):
// Q [B,Hq,N,128], the four KV tensors [B,1,N,128] read in place, window in
// {16,32,64,128} with its [B,N,N] mask. Group0 selects automatic schedules;
// explicit fwd_group / rs_group in {1,2,4}: 1 = native reads with
// the per-head kernels, 2/4 = one CTA shares the KV tiles across that many
// query heads (forward, and the R/S-owned backward passes; the Q pass stays
// per head). Backward returns dR/dS/dVr/dVs already reduced to [B,1,N,128].
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
// Grouped kernels (cuda/single_gather_shared.cu); return false when the device
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

bool launch_Y_gather_shared_auto(
    const at::Tensor& Q,const at::Tensor& R,const at::Tensor& S,const at::Tensor& Vr,
    const at::Tensor& Vs,at::Tensor& Y,at::Tensor& m,at::Tensor& l,const bool* mask,
    int B,int H,int N,float scale,int optin,cudaStream_t stream,int win,const char*& implementation);
