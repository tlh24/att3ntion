"""Autograd bridge for the shared-KV single query-gather kernels.

Same operator as `single_gather_attention`, with R, S, Vr, Vs shared by every
query head ([B,1,N,128]) and read in place rather than replicated to Hq heads.
Contract: D = 128, Hkv = 1, an equal causal window of 16, 32, 64 or 128; an
extra mask may only remove pairs inside that window (it is intersected with it
here). N is padded to a multiple of 16 with padded queries and keys invisible;
gradients come back cropped, KV gradients as [B,1,N,128].
"""
import torch
from torch.autograd import Function

import att3ntion._cuda_kernels as _ck
from att3ntion._single_gather import TILE, _pad_rows, window_mask

WINDOW = 128


class _SharedFn(Function):
    @staticmethod
    def forward(ctx, Q, R, S, Vr, Vs, mask, fwd_group, rs_group, window):
        n = Q.size(2)
        n_pad = (-n) % TILE
        xs = [_pad_rows(x.contiguous().to(torch.bfloat16), n_pad) for x in (Q, R, S, Vr, Vs)]
        win = window_mask(n, window, Q.device)[None].expand(Q.size(0), -1, -1)
        if mask is not None:
            mask = mask.to(device=Q.device, dtype=torch.bool)
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            win = win & mask                       # only removals inside the window
        mask = torch.nn.functional.pad(win, (0, n_pad, 0, n_pad), value=False).contiguous()
        Y, m, l = _ck.single_gather_shared_forward(*xs, mask, window, fwd_group)
        ctx.save_for_backward(*xs, Y, m, l, mask)
        ctx.n, ctx.dtype, ctx.rs_group, ctx.window = n, Q.dtype, rs_group, window
        return Y[:, :, :n].to(Q.dtype)

    @staticmethod
    def backward(ctx, dY):
        Q, R, S, Vr, Vs, Y, m, l, mask = ctx.saved_tensors
        dY = _pad_rows(dY.contiguous().to(torch.bfloat16), Q.size(2) - ctx.n)
        grads = _ck.single_gather_shared_backward(dY, Q, R, S, Vr, Vs, Y, m, l, mask, ctx.window, ctx.rs_group)
        return tuple(g[:, :, :ctx.n].to(ctx.dtype) for g in grads) + (None, None, None, None)


def single_gather_shared_attention(Q, R, S, Vr, Vs, mask=None, fwd_group=0, rs_group=0, window=WINDOW):
    """Q: [B,Hq,N,128]; R, S, Vr, Vs: [B,1,N,128]. Causal `window` (16, 32, 64 or
    128) on both key axes, optionally narrowed by `mask` ([B,N,N] or [N,N] bool).

    fwd_group / rs_group select the forward and R/S-backward schedules. 0 (default)
    chooses per shape: on H100, the sm_90a warpgroup kernels when built with CuTe,
    else the retained-MMA kernels; elsewhere a fixed schedule. 1, 2 or 4 force a
    fixed schedule (4: two heads and both directions per CTA, reusing Q/dY loads)
    and need Hq divisible by the group. In mode 0 on H100 a query with a single
    visible key gets exactly zero score gradients (other modes: zero up to
    rounding); value gradients are unaffected. Schedules differ only in rounding.

    Inputs and dY are cast to BF16, so a wider dtype adds no precision. Every
    visible ordered key pair is evaluated, with BF16 operands and FP32
    accumulation."""
    return _SharedFn.apply(Q, R, S, Vr, Vs, mask, fwd_group, rs_group, window)
