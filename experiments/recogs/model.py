from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Optional
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

_cuda_kernels_available = False
try:
    from att3ntion import HypergraphAttention

    _cuda_kernels_available = True
except (ImportError, OSError):
    HypergraphAttention = None


def _flash_only_sdpa_ctx():
    """
    Prefer the modern SDPA context manager when available, while keeping
    compatibility with older torch releases.
    """
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        return sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION])
    except Exception:
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "sdp_kernel"):
            return torch.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_mem_efficient=False,
                enable_math=False,
            )
    return nullcontext()


def _normalize_self_attn_mask(
    mask: Optional[torch.Tensor],
    batch_size: int,
    ntok: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if mask is None:
        return None

    if mask.ndim == 2:
        if mask.shape != (ntok, ntok):
            raise ValueError(f"2D mask must have shape {(ntok, ntok)}, got {tuple(mask.shape)}")
        mask = mask.unsqueeze(0)
    elif mask.ndim != 3:
        raise ValueError(f"mask must have ndim 2 or 3, got ndim={mask.ndim}")

    if mask.shape[-2:] != (ntok, ntok):
        raise ValueError(f"mask must have trailing shape {(ntok, ntok)}, got {tuple(mask.shape)}")
    if mask.shape[0] not in (1, batch_size):
        raise ValueError(f"mask batch dim must be 1 or {batch_size}, got {mask.shape[0]}")
    if mask.shape[0] == 1 and batch_size > 1:
        mask = mask.expand(batch_size, -1, -1)

    return mask.to(device=device, dtype=torch.bool).contiguous()


class GraphFlashAttention(nn.Module):
    """Pairwise graph attention via PyTorch SDPA (flash on CUDA when available).

    ``post_gelu=True`` (the default, used by the ``graph_flash`` arm) applies a
    GELU after the attention mix and before ``Wo``, mirroring the post-aggregation
    GELUs in the hypergraph CUDA branch so the symmetric comparison is
    apples-to-apples. ``post_gelu=False`` yields a *canonical* pre-norm SDPA
    transformer block (the ``graph_clean`` arm) whose absolute numbers are
    directly comparable to published ReCOGS transformers.
    """

    def __init__(self, d_model: int, n_heads: int, dropout_rate: float = 0.0, post_gelu: bool = True) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout_rate = dropout_rate
        self.post_gelu = post_gelu

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=True)
        self.Wo = nn.Linear(d_model, d_model, bias=True)
        self.gelu = nn.GELU() if post_gelu else None
        self._warned_flash_fallback = False

    def _build_additive_mask(self, mask: Optional[torch.Tensor], batch_size: int, ntok: int, dtype: torch.dtype, device: torch.device) -> Optional[torch.Tensor]:
        mask = _normalize_self_attn_mask(mask, batch_size=batch_size, ntok=ntok, device=device)
        if mask is None:
            return None
        additive = torch.zeros((batch_size, 1, ntok, ntok), device=device, dtype=dtype)
        additive = additive.masked_fill(~mask[:, None, :, :], float("-inf"))
        return additive

    def forward(self, x: torch.Tensor, rotary_emb=None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if rotary_emb is not None:
            raise ValueError("GraphFlashAttention does not support rotary embeddings")

        out_dtype = x.dtype
        batch_size, ntok, _ = x.shape

        q = self.Wq(x).reshape(batch_size, ntok, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        k = self.Wk(x).reshape(batch_size, ntok, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        v = self.Wv(x).reshape(batch_size, ntok, self.n_heads, self.d_head).permute(0, 2, 1, 3)

        additive_mask = self._build_additive_mask(mask, batch_size, ntok, q.dtype, q.device)

        if q.is_cuda:
            try:
                with _flash_only_sdpa_ctx():
                    y = F.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        attn_mask=additive_mask,
                        dropout_p=self.dropout_rate if self.training else 0.0,
                        is_causal=False,
                    )
            except RuntimeError:
                if not self._warned_flash_fallback:
                    warnings.warn(
                        "Flash SDPA backend unavailable for current shape/dtype; "
                        "falling back to default SDPA backend.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    self._warned_flash_fallback = True
                y = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=additive_mask,
                    dropout_p=self.dropout_rate if self.training else 0.0,
                    is_causal=False,
                )
        else:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=additive_mask,
                dropout_p=self.dropout_rate if self.training else 0.0,
                is_causal=False,
            )

        y = y.permute(0, 2, 1, 3).reshape(batch_size, ntok, self.d_model)
        if self.gelu is not None:
            y = self.gelu(y)
        y = self.Wo(y)
        return y.to(out_dtype)

    def calcFlops(self, x: torch.Tensor) -> float:
        bs, ntok, d_model = x.shape
        f = 0.0
        f += 3 * bs * ntok * d_model * d_model * 2
        f += 2 * bs * self.n_heads * ntok * ntok * self.d_head
        f += 2 * bs * self.n_heads * ntok * ntok * self.d_head
        f += bs * ntok * d_model * d_model * 2
        return f


class HypergraphCudaAttentionAdapter(nn.Module):
    """Wraps CUDA hypergraph attention into the common (x, rotary_emb, mask) API."""

    def __init__(self, d_model: int, n_heads: int, scatter: bool = False) -> None:
        super().__init__()
        if not _cuda_kernels_available:
            raise RuntimeError(
                "CUDA HypergraphAttention kernels are not available. "
                "Build/install att3ntion CUDA extension or use attn_impl='graph_flash'."
            )
        # scatter=False -> H-gather (3-way gather only); scatter=True -> H-full
        # (also writes to two tokens at once, doubling the value projections).
        self.inner = HypergraphAttention(d_model, n_heads, scatter=scatter)
        self.d_model = d_model
        self.n_heads = n_heads
        self.scatter = scatter

    def forward(self, x: torch.Tensor, rotary_emb=None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if rotary_emb is not None:
            raise ValueError("HypergraphCudaAttentionAdapter does not support rotary embeddings")
        return self.inner(x, mask=mask)

    def calcFlops(self, x: torch.Tensor) -> float:
        bs, ntok, d_model = x.shape
        f = 0.0
        f += 3 * bs * ntok * d_model**2 * self.n_heads * d_model
        f += 3 * bs * ntok * d_model**2 * self.n_heads * d_model * 2
        f += bs * self.n_heads * ntok**3 * d_model * 2
        f += bs * self.n_heads * ntok**3 * 2 * 3
        f += bs * self.n_heads * ntok**3 * d_model * 3
        f += bs * self.n_heads * ntok**3 * d_model * 3 * 3
        f += bs * self.n_heads * ntok * d_model * (6 + 6)
        f += bs * self.n_heads * ntok * d_model**2
        return f


class RecogsDecoderLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 3,
        attn_impl: str = "hypergraph_cuda",
        max_seq_len: int = 512,
        scatter: bool = False,
        ffn_hidden: int | None = None,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.attn_impl = attn_impl
        self.max_seq_len = max_seq_len
        self.scatter = scatter
        # FFN hidden width. Defaults to 3*d_model; exposed so the graph arm can be
        # tuned to parameter-match the hypergraph arm closely (see run_sweep.py).
        self.ffn_hidden = ffn_hidden if ffn_hidden else 3 * d_model

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.repeated_layers = nn.ModuleList()
        for _ in range(n_layers):
            if attn_impl == "hypergraph_cuda":
                attention_layer = HypergraphCudaAttentionAdapter(
                    d_model=d_model, n_heads=n_heads, scatter=scatter
                )
            elif attn_impl == "graph_flash":
                attention_layer = GraphFlashAttention(d_model=d_model, n_heads=n_heads, post_gelu=True)
            elif attn_impl == "graph_clean":
                # Canonical pre-norm SDPA transformer block (no post-attention GELU).
                attention_layer = GraphFlashAttention(d_model=d_model, n_heads=n_heads, post_gelu=False)
            else:
                raise ValueError(f"Unsupported attn_impl: {attn_impl!r}")

            ffn_layer = nn.Sequential(
                nn.Linear(d_model, self.ffn_hidden),
                nn.ReLU(),
                nn.Linear(self.ffn_hidden, d_model),
            )
            self.repeated_layers.append(
                nn.ModuleDict(
                    {
                        "attention": attention_layer,
                        "norm1": nn.RMSNorm(d_model),
                        "ffn": ffn_layer,
                        "norm2": nn.RMSNorm(d_model),
                    }
                )
            )

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [B, N], got {tuple(input_ids.shape)}")

        batch_size, ntok = input_ids.shape
        if ntok > self.max_seq_len:
            raise ValueError(f"Sequence length {ntok} exceeds max_seq_len {self.max_seq_len}")

        pos = torch.arange(ntok, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_emb(input_ids) + self.pos_emb(pos)

        for layer_block in self.repeated_layers:
            xn = layer_block["norm1"](x)
            attn_output = layer_block["attention"](xn, None, attn_mask)
            x = x + attn_output
            xn = layer_block["norm2"](x)
            ffn_output = layer_block["ffn"](xn)
            x = x + ffn_output

        return self.lm_head(x)

    @torch.no_grad()
    def printParamCount(self) -> None:
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        tag = self.attn_impl
        if self.attn_impl == "hypergraph_cuda":
            tag += "(scatter)" if self.scatter else "(gather)"
        print(
            f"RecogsDecoderLM {tag} L={self.n_layers} ffn_hidden={self.ffn_hidden}: "
            f"number of model parameters: {trainable_params/1e6:.3f}M ({trainable_params})"
        )

    def calcFlops(self, input_ids: torch.Tensor) -> float:
        bs, ntok = input_ids.shape
        x = torch.zeros((bs, ntok, self.d_model), device=input_ids.device, dtype=torch.float32)
        f = 0.0
        for layer_block in self.repeated_layers:
            f += layer_block["attention"].calcFlops(x)
            f += bs * ntok * self.d_model * 10
            f += bs * ntok * self.d_model**2 * 3 * 2
            f += bs * ntok * self.d_model * 10
        f += bs * ntok * self.d_model * self.vocab_size * 2
        return f

