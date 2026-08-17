import os
import torch
import torch.nn as nn
import math
import att3ntion._custom_op  # noqa: F401 — registers torch.ops.att3ntion.hypergraph_{forward,backward}
# import att3ntion._torch_kernels as _torch_kernels
from torch.autograd import Function

_CHECK_GRAD_FINITE = os.getenv("ATT3NTION_CHECK_GRADS", "0").lower() in {"1", "true", "yes", "on"}

def _pad_to_multiple(tensor, multiple, dim=2):
    """Pad tensor along specified dimension to be a multiple of `multiple`."""
    size = tensor.size(dim)
    if size % multiple == 0:
        return tensor, 0
    pad_size = multiple - (size % multiple)
    # Create padding tuple: (last_dim_left, last_dim_right, ..., dim_left, dim_right)
    # For 4D tensor [B, H, N, D] and dim=2, we need to pad N
    ndim = tensor.ndim
    pad = [0] * (2 * ndim)
    # PyTorch pad works from last dim backwards
    pad_idx = 2 * (ndim - 1 - dim)
    pad[pad_idx + 1] = pad_size  # pad on the right side
    return torch.nn.functional.pad(tensor, pad), pad_size


def _normalize_self_attn_mask(mask, batch_size, ntok, device):
    if mask is None:
        return None

    if mask.ndim == 2:
        if mask.shape != (ntok, ntok):
            raise ValueError(
                f"2D mask must have shape {(ntok, ntok)}, got {tuple(mask.shape)}"
            )
        mask = mask.unsqueeze(0)
    elif mask.ndim != 3:
        raise ValueError(f"mask must have ndim 2 or 3, got ndim={mask.ndim}")

    if mask.shape[-2:] != (ntok, ntok):
        raise ValueError(
            f"mask must have trailing shape {(ntok, ntok)}, got {tuple(mask.shape)}"
        )
    if mask.shape[0] not in (1, batch_size):
        raise ValueError(
            f"mask batch dim must be 1 or {batch_size}, got {mask.shape[0]}"
        )
    if mask.shape[0] == 1 and batch_size > 1:
        mask = mask.expand(batch_size, -1, -1)

    return mask.to(device=device, dtype=torch.bool).contiguous()


class _HypergraphAttentionAutograd(Function):
    """
    Bridge between PyTorch autograd and the manual C++ forward/backward passes.
    Saves softmax statistics from forward pass to avoid redundant computation in backward.
    """

    @staticmethod
    def forward(
        ctx,
        Q,
        R,
        S,
        Vq_1,
        Vq_2,
        Vr_1,
        Vr_2,
        Vs_1,
        Vs_2,
        mask,
        dropout_rate=0.0,
        scatter=False,
    ):
        """
        Calls the C++ forward pass and saves necessary tensors for backward.
        Args:
            ctx: Context object to save tensors.
            Q, R, S, V*_*: Input tensors for the attention mechanism.
            dropout_rate: Dropout rate (passed to C++ if needed, but not differentiable).
        Returns:
            Tuple of selected per-axis outputs (Y_q, Y_r, Y_s).
        """
        TILE_SIZE = 16
        orig_seq_len = Q.size(2)
        input_dtype = Q.dtype
        batch_size = Q.size(0)
        if mask is not None and (R.size(2) != orig_seq_len or S.size(2) != orig_seq_len):
            raise ValueError("masked CUDA hypergraph attention requires equal sequence lengths")
        mask = _normalize_self_attn_mask(mask, batch_size, orig_seq_len, Q.device)

        # CUDA forward kernels are BF16 I/O. Cast at the autograd boundary so
        # callers can remain dtype-agnostic.
        Q, pad_q = _pad_to_multiple(Q.contiguous().to(torch.bfloat16), TILE_SIZE, dim=2)
        R, pad_r = _pad_to_multiple(R.contiguous().to(torch.bfloat16), TILE_SIZE, dim=2)
        S, pad_s = _pad_to_multiple(S.contiguous().to(torch.bfloat16), TILE_SIZE, dim=2)
        Vq_1, _ = _pad_to_multiple(Vq_1.contiguous().to(torch.bfloat16), TILE_SIZE, dim=2)
        Vq_2, _ = _pad_to_multiple(Vq_2.contiguous().to(torch.bfloat16), TILE_SIZE, dim=2)
        Vr_1, _ = _pad_to_multiple(Vr_1.contiguous().to(torch.bfloat16), TILE_SIZE, dim=2)
        Vr_2, _ = _pad_to_multiple(Vr_2.contiguous().to(torch.bfloat16), TILE_SIZE, dim=2)
        Vs_1, _ = _pad_to_multiple(Vs_1.contiguous().to(torch.bfloat16), TILE_SIZE, dim=2)
        Vs_2, _ = _pad_to_multiple(Vs_2.contiguous().to(torch.bfloat16), TILE_SIZE, dim=2)
        if mask is None:
            mask_tensor = torch.empty(0, device=Q.device, dtype=torch.bool)
        else:
            pad_n = Q.size(2) - orig_seq_len
            if pad_n > 0:
                mask_tensor = torch.nn.functional.pad(mask, (0, pad_n, 0, pad_n), value=False)
            else:
                mask_tensor = mask
            mask_tensor = mask_tensor.contiguous()

        outputs_tuple = torch.ops.att3ntion.hypergraph_forward(
            Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2, mask_tensor, dropout_rate,
            orig_seq_len, orig_seq_len, orig_seq_len,
        )

        if isinstance(outputs_tuple, tuple) and len(outputs_tuple) == 12:
            Y_q, Y_r, Y_s, Y_q_, Y_r_, Y_s_, m_i, l_i, m_j, l_j, m_k, l_k = outputs_tuple
        else:
            raise TypeError(f"C++ forward expected to return a tuple of 12 Tensors, but got {type(outputs_tuple)} with len {len(outputs_tuple) if isinstance(outputs_tuple, tuple) else 'N/A'}")

        # Padded bf16 gather outputs, kept for the backward's collapsed
        # correction sums (rowsum(dY * Y) on the tensor-core path).
        Y_q_pad, Y_r_pad, Y_s_pad = Y_q, Y_r, Y_s

        if orig_seq_len != Y_q.size(2):
            Y_q  = Y_q [:, :, :orig_seq_len, :]
            Y_r  = Y_r [:, :, :orig_seq_len, :]
            Y_s  = Y_s [:, :, :orig_seq_len, :]
            Y_q_ = Y_q_[:, :, :orig_seq_len, :]
            Y_r_ = Y_r_[:, :, :orig_seq_len, :]
            Y_s_ = Y_s_[:, :, :orig_seq_len, :]

        if Y_q.dtype != input_dtype:
            Y_q  = Y_q .to(input_dtype)
            Y_r  = Y_r .to(input_dtype)
            Y_s  = Y_s .to(input_dtype)
            Y_q_ = Y_q_.to(input_dtype)
            Y_r_ = Y_r_.to(input_dtype)
            Y_s_ = Y_s_.to(input_dtype)

        ctx.save_for_backward(Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2,
                              m_i, l_i, m_j, l_j, m_k, l_k, mask_tensor,
                              Y_q_pad, Y_r_pad, Y_s_pad)
        ctx.orig_seq_len = orig_seq_len
        ctx.input_dtype = input_dtype

        return Y_q, Y_r, Y_s, Y_q_, Y_r_, Y_s_

    @staticmethod
    def backward(ctx, grad_Y_q, grad_Y_r, grad_Y_s, grad_Y_q_, grad_Y_r_, grad_Y_s_):
        """
        Calls the C++ backward pass using saved tensors and pre-computed softmax stats.
        This avoids redundant computation of softmax statistics.
        Args:
            ctx: Context object with saved tensors.
            grad_Y_q/grad_Y_r/grad_Y_s: Upstream gradients for per-axis outputs.
        Returns:
            A tuple of gradients corresponding *exactly* to the inputs of the
            forward function (
              Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2, dropout_rate, scatter
            ).
        """
        Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2, \
            m_i, l_i, m_j, l_j, m_k, l_k, mask_tensor, \
            Y_q_pad, Y_r_pad, Y_s_pad = ctx.saved_tensors
        orig_seq_len = ctx.orig_seq_len

        out_dtype = getattr(ctx, "input_dtype", Q.dtype)

        TILE_SIZE = 16
        # Backward CUDA kernels use BF16 I/O with FP32 accumulation internally.
        # Keep gather/scatter cotangents separate so each backward path receives
        # the exact upstream gradient for its own output branch.
        if grad_Y_q_ is None:
            grad_Y_q_ = torch.zeros_like(grad_Y_q)
        if grad_Y_r_ is None:
            grad_Y_r_ = torch.zeros_like(grad_Y_r)
        if grad_Y_s_ is None:
            grad_Y_s_ = torch.zeros_like(grad_Y_s)
        grad_Y_q, _ = _pad_to_multiple(grad_Y_q.contiguous().to(torch.bfloat16), TILE_SIZE, dim=2)
        grad_Y_r, _ = _pad_to_multiple(grad_Y_r.contiguous().to(torch.bfloat16), TILE_SIZE, dim=2)
        grad_Y_s, _ = _pad_to_multiple(grad_Y_s.contiguous().to(torch.bfloat16), TILE_SIZE, dim=2)
        grad_Y_q_, _ = _pad_to_multiple(grad_Y_q_.contiguous().to(torch.bfloat16), TILE_SIZE, dim=2)
        grad_Y_r_, _ = _pad_to_multiple(grad_Y_r_.contiguous().to(torch.bfloat16), TILE_SIZE, dim=2)
        grad_Y_s_, _ = _pad_to_multiple(grad_Y_s_.contiguous().to(torch.bfloat16), TILE_SIZE, dim=2)

        grad_tuple = torch.ops.att3ntion.hypergraph_backward(
            grad_Y_q, grad_Y_r, grad_Y_s, grad_Y_q_, grad_Y_r_, grad_Y_s_,
            Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2,
            m_i, l_i, m_j, l_j, m_k, l_k, mask_tensor,
            Y_q_pad, Y_r_pad, Y_s_pad
        )

        if not (isinstance(grad_tuple, tuple) and len(grad_tuple) == 9):
            raise ValueError(f"C++ backward expected to return a tuple of 9 gradients, got {len(grad_tuple) if isinstance(grad_tuple, tuple) else type(grad_tuple)}")

        grad_Q, grad_R, grad_S, grad_Vq_1, grad_Vq_2, grad_Vr_1, grad_Vr_2, grad_Vs_1, grad_Vs_2 = grad_tuple

        if orig_seq_len != grad_Q.size(2):
            grad_Q = grad_Q[:, :, :orig_seq_len, :]
            grad_R = grad_R[:, :, :orig_seq_len, :]
            grad_S = grad_S[:, :, :orig_seq_len, :]
            grad_Vq_1 = grad_Vq_1[:, :, :orig_seq_len, :]
            grad_Vq_2 = grad_Vq_2[:, :, :orig_seq_len, :]
            grad_Vr_1 = grad_Vr_1[:, :, :orig_seq_len, :]
            grad_Vr_2 = grad_Vr_2[:, :, :orig_seq_len, :]
            grad_Vs_1 = grad_Vs_1[:, :, :orig_seq_len, :]
            grad_Vs_2 = grad_Vs_2[:, :, :orig_seq_len, :]

        grad_Q = grad_Q.to(out_dtype)
        grad_R = grad_R.to(out_dtype)
        grad_S = grad_S.to(out_dtype)
        grad_Vq_1 = grad_Vq_1.to(out_dtype)
        grad_Vq_2 = grad_Vq_2.to(out_dtype)
        grad_Vr_1 = grad_Vr_1.to(out_dtype)
        grad_Vr_2 = grad_Vr_2.to(out_dtype)
        grad_Vs_1 = grad_Vs_1.to(out_dtype)
        grad_Vs_2 = grad_Vs_2.to(out_dtype)

        if _CHECK_GRAD_FINITE:
            debug_names = [
                "grad_Q", "grad_R", "grad_S",
                "grad_Vq_1", "grad_Vq_2",
                "grad_Vr_1", "grad_Vr_2",
                "grad_Vs_1", "grad_Vs_2",
            ]
            grads = (grad_Q, grad_R, grad_S, grad_Vq_1, grad_Vq_2, grad_Vr_1, grad_Vr_2, grad_Vs_1, grad_Vs_2)
            for name, tensor in zip(debug_names, grads):
                if tensor is not None and tensor.is_floating_point():
                    if not torch.isfinite(tensor).all():
                        raise RuntimeError(f"{name} contains non-finite values")

        return (
            grad_Q,
            grad_R,
            grad_S,
            grad_Vq_1,
            grad_Vq_2,
            grad_Vr_1,
            grad_Vr_2,
            grad_Vs_1,
            grad_Vs_2,
            None,
            None,
            None,
        )

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class HypergraphAttention(nn.Module):
    """
    3-way hypergraph attention layer backed by hand-written CUDA kernels.
    """
    def __init__(self, d_model, n_heads, dropout_rate=0, scatter=False):
        super().__init__()
        
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scatter = scatter
        
        self.Wq = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.Wr = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.Ws = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        
        value_proj_multiplier = 2 if self.scatter else 1
        self.Wv_q = nn.Linear(d_model, n_heads * self.head_dim * value_proj_multiplier, bias=True)
        self.Wv_r = nn.Linear(d_model, n_heads * self.head_dim * value_proj_multiplier, bias=True)
        self.Wv_s = nn.Linear(d_model, n_heads * self.head_dim * value_proj_multiplier, bias=True)
        
        self.Wo = nn.Linear(n_heads * self.head_dim, d_model, bias=True)
       
        self.dropout = nn.Dropout(dropout_rate)
        self.gelu = QuickGELU()

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Backward-compatible loading for Wv_* shapes across gather/scatter configs.
        for proj_name in ("Wv_q", "Wv_r", "Wv_s"):
            module = getattr(self, proj_name)
            for suffix, target in (("weight", module.weight), ("bias", module.bias)):
                key = f"{prefix}{proj_name}.{suffix}"
                if key not in state_dict:
                    continue
                loaded = state_dict[key]
                if loaded.shape == target.shape:
                    continue
                if (
                    loaded.ndim == target.ndim
                    and loaded.shape[0] == target.shape[0] * 2
                    and loaded.shape[1:] == target.shape[1:]
                ):
                    state_dict[key] = loaded[: target.shape[0]].contiguous()
                    continue
                if (
                    loaded.ndim == target.ndim
                    and loaded.shape[0] * 2 == target.shape[0]
                    and loaded.shape[1:] == target.shape[1:]
                ):
                    expanded = torch.zeros_like(target)
                    expanded[: loaded.shape[0]] = loaded
                    state_dict[key] = expanded

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        
    def forward(self, x, rotary_emb=None, mask=None):
        if rotary_emb is not None:
            raise ValueError("CUDA HypergraphAttention does not support rotary embeddings")
        batch_size, ntok, d_model = x.shape
        
        Q = self.Wq(x)
        R = self.Wr(x)
        S = self.Ws(x)
        
        if self.scatter:
            Vq = self.Wv_q(x)
            Vr = self.Wv_r(x)
            Vs = self.Wv_s(x)
            Vq_1, Vq_2 = Vq.reshape(batch_size, ntok, self.n_heads, self.head_dim * 2).permute(0, 2, 1, 3).split(self.head_dim, dim=-1)
            Vr_1, Vr_2 = Vr.reshape(batch_size, ntok, self.n_heads, self.head_dim * 2).permute(0, 2, 1, 3).split(self.head_dim, dim=-1)
            Vs_1, Vs_2 = Vs.reshape(batch_size, ntok, self.n_heads, self.head_dim * 2).permute(0, 2, 1, 3).split(self.head_dim, dim=-1)
        else:
            Vq_1 = self.Wv_q(x).reshape(batch_size, ntok, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            Vr_1 = self.Wv_r(x).reshape(batch_size, ntok, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            Vs_1 = self.Wv_s(x).reshape(batch_size, ntok, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            Vq_2 = torch.zeros_like(Vq_1)
            Vr_2 = torch.zeros_like(Vr_1)
            Vs_2 = torch.zeros_like(Vs_1)
        
        Q = Q.reshape(batch_size, ntok, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        R = R.reshape(batch_size, ntok, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        S = S.reshape(batch_size, ntok, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        Y_q, Y_r, Y_s, Y_q_, Y_r_, Y_s_ = _HypergraphAttentionAutograd.apply(
            Q, R, S,
            Vq_1, Vq_2,
            Vr_1, Vr_2,
            Vs_1, Vs_2,
            mask,
            self.dropout.p,
            self.scatter,
        )

        y = self.gelu(Y_q) + self.gelu(Y_r) + self.gelu(Y_s)
        if self.scatter:
            y = y + self.gelu(Y_q_) + self.gelu(Y_r_) + self.gelu(Y_s_)

        y = y.permute(0, 2, 1, 3).contiguous().view(batch_size, ntok, self.n_heads * self.head_dim)

        y = self.Wo(y)

        return y


class _HypergraphAttentionTorch(nn.Module):
    """
    Torch-based reference implementation (uses PyTorch autograd for backward).
    Intended for correctness testing against the CUDA kernels.
    """
    def __init__(self, d_model, n_heads, dropout_rate=0):
        super().__init__()
        
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.Wq = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.Wr = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.Ws = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        
        self.Wv_q = nn.Linear(d_model, n_heads * self.head_dim * 2, bias=True)
        self.Wv_r = nn.Linear(d_model, n_heads * self.head_dim * 2, bias=True)
        self.Wv_s = nn.Linear(d_model, n_heads * self.head_dim * 2, bias=True)
        
        self.Wo = nn.Linear(n_heads * self.head_dim, d_model, bias=True)
       
        self.dropout = nn.Dropout(dropout_rate)
        self.gelu = QuickGELU()
        
    def forward(self, x):
        batch_size, ntok, d_model = x.shape
        
        Q = self.Wq(x)
        R = self.Wr(x)
        S = self.Ws(x)
        
        Vq = self.Wv_q(x)
        Vr = self.Wv_r(x)
        Vs = self.Wv_s(x)
        
        Q = Q.reshape(batch_size, ntok, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        R = R.reshape(batch_size, ntok, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        S = S.reshape(batch_size, ntok, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        Vq_1, Vq_2 = Vq.reshape(batch_size, ntok, self.n_heads, self.head_dim*2).permute(0, 2, 1, 3).split(self.head_dim, dim=-1)
        Vr_1, Vr_2 = Vr.reshape(batch_size, ntok, self.n_heads, self.head_dim*2).permute(0, 2, 1, 3).split(self.head_dim, dim=-1)
        Vs_1, Vs_2 = Vs.reshape(batch_size, ntok, self.n_heads, self.head_dim*2).permute(0, 2, 1, 3).split(self.head_dim, dim=-1)
        
        outputs_tuple = _torch_kernels.forward(
            Q.contiguous(), R.contiguous(), S.contiguous(),
            Vq_1.contiguous(), Vq_2.contiguous(),
            Vr_1.contiguous(), Vr_2.contiguous(),
            Vs_1.contiguous(), Vs_2.contiguous(),
            self.dropout.p
        )
        
        y = sum(outputs_tuple)
        
        y = y.permute(0, 2, 1, 3).contiguous().view(batch_size, ntok, self.n_heads * self.head_dim)
        
        y = self.Wo(y) 

        return y
