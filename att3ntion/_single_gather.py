"""Autograd bridge for the single query-gather kernels.

    Y[i] = sum_{j,k} softmax_{j,k}(Q_i . R_j . S_k / sqrt(D)) Vr[j] Vs[k]

One softmax per query, so unlike HypergraphAttention nothing is anchored on R
or S and there is no Vq. The kernels want N % 16 == 0; the bridge pads, and a
padded position is excluded through the mask (as query and as key), so an
unmasked call still pads into a masked one.
"""
import torch
from torch.autograd import Function

import att3ntion._cuda_kernels as _ck

TILE = 16


def _pad_rows(t, n_pad):
    return torch.nn.functional.pad(t, (0, 0, 0, n_pad)) if n_pad else t


class _SingleGatherFn(Function):
    @staticmethod
    def forward(ctx, Q, R, S, Vr, Vs, mask):
        n = Q.size(2)
        n_pad = (-n) % TILE
        xs = [_pad_rows(x.contiguous().to(torch.bfloat16), n_pad) for x in (Q, R, S, Vr, Vs)]
        if mask is None and n_pad:
            mask = torch.ones(Q.size(0), n, n, dtype=torch.bool, device=Q.device)
        if mask is not None:
            mask = mask.to(device=Q.device, dtype=torch.bool)
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            if mask.size(0) == 1 and Q.size(0) > 1:
                mask = mask.expand(Q.size(0), -1, -1)
            mask = torch.nn.functional.pad(mask, (0, n_pad, 0, n_pad), value=False).contiguous()
        Y, m, l = _ck.single_gather_forward(*xs, mask)
        ctx.save_for_backward(*xs, Y, m, l, mask if mask is not None else torch.empty(0))
        ctx.n = n
        ctx.dtype = Q.dtype
        return Y[:, :, :n].to(Q.dtype)

    @staticmethod
    def backward(ctx, dY):
        Q, R, S, Vr, Vs, Y, m, l, mask = ctx.saved_tensors
        mask = mask if mask.numel() else None
        dY = _pad_rows(dY.contiguous().to(torch.bfloat16), Q.size(2) - ctx.n)
        grads = _ck.single_gather_backward(dY, Q, R, S, Vr, Vs, Y, m, l, mask)
        return tuple(g[:, :, :ctx.n].to(ctx.dtype) for g in grads) + (None,)


def single_gather_attention(Q, R, S, Vr, Vs, mask=None):
    """Q, R, S, Vr, Vs: [B,H,N,D], D in {64,128}. mask: [B,N,N] or [N,N] bool,
    True = visible; a query sees the (j, k) pairs whose members are both
    visible to it. A fully masked query returns zeros."""
    return _SingleGatherFn.apply(Q, R, S, Vr, Vs, mask)
