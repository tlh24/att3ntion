import torch
from torch.library import custom_op

import att3ntion._cuda_kernels as _cuda_kernels


@custom_op("att3ntion::hypergraph_forward", mutates_args=())
def hypergraph_forward(
    Q: torch.Tensor, R: torch.Tensor, S: torch.Tensor,
    Vq_1: torch.Tensor, Vq_2: torch.Tensor,
    Vr_1: torch.Tensor, Vr_2: torch.Tensor,
    Vs_1: torch.Tensor, Vs_2: torch.Tensor,
    dropout_rate: float,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
]:
    return _cuda_kernels.forward(Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2, dropout_rate)


@hypergraph_forward.register_fake
def _hypergraph_forward_fake(
    Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2, dropout_rate,
):
    B, H, I, D = Q.shape
    _, _, J, _ = R.shape
    _, _, K, _ = S.shape
    opts = {"dtype": Q.dtype, "device": Q.device}
    stats_opts = {"dtype": torch.float32, "device": Q.device}
    Y_q = torch.empty((B, H, I, D), **opts)
    Y_r = torch.empty((B, H, J, D), **opts)
    Y_s = torch.empty((B, H, K, D), **opts)
    Y_q_ = torch.empty((B, H, I, D), **opts)
    Y_r_ = torch.empty((B, H, J, D), **opts)
    Y_s_ = torch.empty((B, H, K, D), **opts)
    m_i = torch.empty((B, H, I), **stats_opts)
    l_i = torch.empty((B, H, I), **stats_opts)
    m_j = torch.empty((B, H, J), **stats_opts)
    l_j = torch.empty((B, H, J), **stats_opts)
    m_k = torch.empty((B, H, K), **stats_opts)
    l_k = torch.empty((B, H, K), **stats_opts)
    return Y_q, Y_r, Y_s, Y_q_, Y_r_, Y_s_, m_i, l_i, m_j, l_j, m_k, l_k


@custom_op("att3ntion::hypergraph_backward", mutates_args=())
def hypergraph_backward(
    grad_Y_q: torch.Tensor, grad_Y_r: torch.Tensor, grad_Y_s: torch.Tensor,
    Q: torch.Tensor, R: torch.Tensor, S: torch.Tensor,
    Vq_1: torch.Tensor, Vq_2: torch.Tensor,
    Vr_1: torch.Tensor, Vr_2: torch.Tensor,
    Vs_1: torch.Tensor, Vs_2: torch.Tensor,
    m_i: torch.Tensor, l_i: torch.Tensor,
    m_j: torch.Tensor, l_j: torch.Tensor,
    m_k: torch.Tensor, l_k: torch.Tensor,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
]:
    return _cuda_kernels.backward(
        grad_Y_q, grad_Y_r, grad_Y_s, Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2,
        m_i, l_i, m_j, l_j, m_k, l_k,
    )


@hypergraph_backward.register_fake
def _hypergraph_backward_fake(
    grad_Y_q, grad_Y_r, grad_Y_s, Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2,
    m_i, l_i, m_j, l_j, m_k, l_k,
):
    return (
        torch.empty_like(Q), torch.empty_like(R), torch.empty_like(S),
        torch.empty_like(Vq_1), torch.empty_like(Vq_2),
        torch.empty_like(Vr_1), torch.empty_like(Vr_2),
        torch.empty_like(Vs_1), torch.empty_like(Vs_2),
    )


def _setup_context(ctx, inputs, output):
    Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2, _ = inputs
    _, _, _, _, _, _, m_i, l_i, m_j, l_j, m_k, l_k = output
    ctx.save_for_backward(Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2, m_i, l_i, m_j, l_j, m_k, l_k)


def _backward(
    ctx,
    grad_Y_q, grad_Y_r, grad_Y_s, grad_Y_q_, grad_Y_r_, grad_Y_s_,
    grad_m_i, grad_l_i, grad_m_j, grad_l_j, grad_m_k, grad_l_k,
):
    Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2, m_i, l_i, m_j, l_j, m_k, l_k = ctx.saved_tensors
    return (
        *hypergraph_backward(
            grad_Y_q, grad_Y_r, grad_Y_s, Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2,
            m_i, l_i, m_j, l_j, m_k, l_k,
        ),
        None,
    )


hypergraph_forward.register_autograd(_backward, setup_context=_setup_context)
