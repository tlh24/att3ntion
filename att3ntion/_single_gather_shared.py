"""Experimental autograd bridge for the shared-KV single query-gather prototype.

Same operator as `single_gather_attention` with the four KV tensors shared by
every query head ([B,1,N,128]), read in place by the kernels: no replication
to Hq heads. Fixed contract (see docs/KERNEL_HISTORY.md):
D = 128, Hkv = 1, an equal causal window of 16, 32, 64 or 128 (an extra mask may only remove
pairs inside that window; it is intersected with it here). The bridge pads N
to a multiple of 16 at the logical KV size and keeps padded queries and keys
invisible; gradients come back cropped, KV gradients in [B,1,N,128].
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
    """Q: [B,Hq,N,128]; R, S, Vr, Vs: [B,1,N,128]; causal window on both key axes,
    optionally narrowed by `mask` ([B,N,N] or [N,N] bool). fwd_group / rs_group in
    {0, 1, 2, 4}. Mode 0 (default) selects the measured H100 schedules and
    gives singleton queries zero score-gradient contributions while preserving
    their value gradients. Explicit modes 1/2/4 retain their legacy
    arithmetic; Hq must be divisible by each explicitly selected group.
    Windows are 16/32/64/128. Automatic mode uses optional Hopper warpgroup
    kernels when built with CuTe, with retained-MMA schedules for other shapes.
    Explicit R/S strategy 4 uses two heads and two independent directions per CTA,
    reusing raw Q/dY loads. Its smaller row tiles can change rounding relative
    to group 1/2; the mathematical operator and numerical tolerances are the same.
    Inputs and dY are converted to BF16 internally; returning a wider input
    dtype does not increase arithmetic precision. All visible ordered key pairs
    are evaluated, with BF16 intermediate operands and FP32 accumulation.
    See docs/KERNEL_HISTORY.md for the
    measured accuracy gates and known numerical stress limits."""
    return _SharedFn.apply(Q, R, S, Vr, Vs, mask, fwd_group, rs_group, window)
