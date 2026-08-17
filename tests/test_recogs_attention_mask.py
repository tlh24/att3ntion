from pathlib import Path
import sys

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RECOGS_DIR = PROJECT_ROOT / "experiments" / "recogs"
if str(RECOGS_DIR) not in sys.path:
    sys.path.insert(0, str(RECOGS_DIR))

from model import GraphFlashAttention, HypergraphCudaAttentionAdapter, RecogsDecoderLM  # noqa: F401
from data import EncodedExample, collate_decoder_only  # noqa: F401


def _assert_no_future_leakage(attn_module, x: torch.Tensor, mask: torch.Tensor, query_idx: int, future_idx: int):
    x1 = x.clone()
    x2 = x.clone()
    x2[:, future_idx, :] = x2[:, future_idx, :] + 10.0 * torch.randn_like(x2[:, future_idx, :])

    y1 = attn_module(x1, None, mask)
    y2 = attn_module(x2, None, mask)
    assert torch.allclose(y1[:, query_idx, :], y2[:, query_idx, :], atol=1e-4, rtol=1e-4)


def _assert_influence(attn_module, x: torch.Tensor, mask: torch.Tensor, query_idx: int, key_idx: int):
    """Inverse of the leakage check: perturbing ``key_idx`` MUST move ``query_idx``."""
    x1 = x.clone()
    x2 = x.clone()
    x2[:, key_idx, :] = x2[:, key_idx, :] + 10.0 * torch.randn_like(x2[:, key_idx, :])

    y1 = attn_module(x1, None, mask)
    y2 = attn_module(x2, None, mask)
    assert not torch.allclose(y1[:, query_idx, :], y2[:, query_idx, :], atol=1e-4, rtol=1e-4)


def _prefix_lm_mask(n: int, prefix_len: int, device, batch: int = 1) -> torch.Tensor:
    """Bidirectional prefix (cols < prefix_len) | causal, matching data.collate_decoder_only."""
    causal = torch.tril(torch.ones((n, n), device=device, dtype=torch.bool))
    causal[:, :prefix_len] = True
    return causal.unsqueeze(0).expand(batch, -1, -1).contiguous()


def test_graph_flash_causal_mask_blocks_future_influence():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, N, D, H = 1, 6, 64, 4
    x = torch.randn(B, N, D, device=device, dtype=torch.float32)
    mask = torch.tril(torch.ones((B, N, N), device=device, dtype=torch.bool))

    mod = GraphFlashAttention(d_model=D, n_heads=H).to(device=device, dtype=torch.float32)
    _assert_no_future_leakage(mod, x=x, mask=mask, query_idx=2, future_idx=5)


def test_graph_flash_bool_mask_polarity_allowed_true():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mod = GraphFlashAttention(d_model=4, n_heads=1).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        mod.Wq.weight.zero_()
        mod.Wk.weight.zero_()
        mod.Wv.weight.copy_(torch.eye(4, device=device))
        mod.Wv.bias.zero_()
        mod.Wo.weight.copy_(torch.eye(4, device=device))
        mod.Wo.bias.zero_()

    x = torch.tensor([[[0.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0]]], device=device)
    mask = torch.tensor([[[True, False], [True, True]]], dtype=torch.bool, device=device)
    y = mod(x, None, mask)
    assert abs(float(y[0, 0, 1].item())) < 1e-6


def test_graph_clean_causal_mask_blocks_future_influence():
    """The clean baseline arm (post_gelu=False) must be causal too."""
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, N, D, H = 1, 6, 64, 4
    x = torch.randn(B, N, D, device=device, dtype=torch.float32)
    mask = torch.tril(torch.ones((B, N, N), device=device, dtype=torch.bool))

    mod = GraphFlashAttention(d_model=D, n_heads=H, post_gelu=False).to(device=device, dtype=torch.float32)
    assert mod.gelu is None
    _assert_no_future_leakage(mod, x=x, mask=mask, query_idx=2, future_idx=5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_hypergraph_cuda_causal_mask_blocks_future_influence():
    torch.manual_seed(1)
    device = torch.device("cuda")
    B, N, D, H = 1, 6, 256, 4  # D/H=64 supported by CUDA kernels
    x = torch.randn(B, N, D, device=device, dtype=torch.float32)
    mask = torch.tril(torch.ones((B, N, N), device=device, dtype=torch.bool))

    mod = HypergraphCudaAttentionAdapter(d_model=D, n_heads=H).to(device=device, dtype=torch.float32)
    _assert_no_future_leakage(mod, x=x, mask=mask, query_idx=2, future_idx=5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_hypergraph_cuda_scatter_causal_mask_blocks_future_influence():
    """H-full (scatter=True) must remain causal: position i may only aggregate
    pairs (j,k) with j,k <= i, and the scatter/write must not leak future tokens."""
    torch.manual_seed(2)
    device = torch.device("cuda")
    B, N, D, H = 1, 6, 256, 4
    x = torch.randn(B, N, D, device=device, dtype=torch.float32)
    mask = torch.tril(torch.ones((B, N, N), device=device, dtype=torch.bool))

    mod = HypergraphCudaAttentionAdapter(d_model=D, n_heads=H, scatter=True).to(device=device, dtype=torch.float32)
    assert mod.scatter is True
    _assert_no_future_leakage(mod, x=x, mask=mask, query_idx=2, future_idx=5)


# --------------------------------------------------------------------------- #
# Prefix-LM mask: bidirectional within [BOS] src [SEP], causal over the LF span.
# --------------------------------------------------------------------------- #


def test_collate_prefix_lm_mask_semantics():
    """The collator's prefix-LM mask must be bidirectional within the prefix,
    causal over the LF span, block prefix->LF, and zero out all pad rows/cols."""
    pad_id = 0
    # ex0: prefix_len=3 ([BOS] a [SEP]) + LF [x y EOS], len 6, no padding.
    ex0 = EncodedExample(
        input_ids=[1, 10, 2, 11, 12, 3],
        labels=[-100, -100, -100, 11, 12, 3],
        category="c", source="a", logical_form="x y", was_truncated=False, prefix_len=3,
    )
    # ex1: prefix_len=2 + LF [y EOS], len 4 -> padded to 6.
    ex1 = EncodedExample(
        input_ids=[1, 2, 12, 3],
        labels=[-100, -100, 12, 3],
        category="c", source="", logical_form="y", was_truncated=False, prefix_len=2,
    )
    out = collate_decoder_only([ex0, ex1], pad_token_id=pad_id, prefix_lm=True)
    m = out["attn_mask"]
    assert m.shape == (2, 6, 6)

    # ex0: every prefix query (rows 0-2) sees all prefix keys (bidirectional)...
    assert m[0, :3, :3].all()
    # ...and no prefix query sees any LF key.
    assert not m[0, :3, 3:].any()
    # LF rows are causal but still see the whole prefix.
    assert m[0, 3].tolist() == [True, True, True, True, False, False]
    assert m[0, 4].tolist() == [True, True, True, True, True, False]
    assert m[0, 5].tolist() == [True, True, True, True, True, True]

    # ex1: pad rows (4,5) and pad cols (4,5) are fully masked.
    assert not m[1, 4:, :].any()
    assert not m[1, :, 4:].any()
    # prefix (cols 0,1) visible to the valid LF query at row 3.
    assert m[1, 3].tolist() == [True, True, True, True, False, False]


def test_collate_causal_unchanged_without_prefix_lm():
    """Default (prefix_lm=False) keeps the strictly-causal mask."""
    ex = EncodedExample(
        input_ids=[1, 10, 2, 11, 3],
        labels=[-100, -100, -100, 11, 3],
        category="c", source="a", logical_form="x", was_truncated=False, prefix_len=3,
    )
    out = collate_decoder_only([ex], pad_token_id=0, prefix_lm=False)
    m = out["attn_mask"][0]
    expected = torch.tril(torch.ones((5, 5), dtype=torch.bool))
    assert torch.equal(m, expected)


def test_graph_flash_prefix_lm_directionality():
    """Prefix query is influenced by a future prefix key (bidirectional); an LF
    query is NOT influenced by a future LF key (causal)."""
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, N, D, H, P = 1, 6, 64, 4, 3
    x = torch.randn(B, N, D, device=device, dtype=torch.float32)
    mask = _prefix_lm_mask(N, P, device, batch=B)

    mod = GraphFlashAttention(d_model=D, n_heads=H).to(device=device, dtype=torch.float32)
    _assert_influence(mod, x=x, mask=mask, query_idx=0, key_idx=1)          # prefix sees future prefix
    _assert_no_future_leakage(mod, x=x, mask=mask, query_idx=3, future_idx=5)  # LF stays causal


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_hypergraph_cuda_prefix_lm_directionality():
    """The CUDA hypergraph kernel must honor a non-causal prefix mask: prefix is
    bidirectional, LF span stays causal. This gates prefix-LM for the hg arm."""
    torch.manual_seed(1)
    device = torch.device("cuda")
    B, N, D, H, P = 1, 6, 256, 4, 3
    x = torch.randn(B, N, D, device=device, dtype=torch.float32)
    mask = _prefix_lm_mask(N, P, device, batch=B)

    mod = HypergraphCudaAttentionAdapter(d_model=D, n_heads=H).to(device=device, dtype=torch.float32)
    _assert_influence(mod, x=x, mask=mask, query_idx=0, key_idx=1)
    _assert_no_future_leakage(mod, x=x, mask=mask, query_idx=3, future_idx=5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required (hypergraph kernel)")
def test_graph_ffn_hidden_param_matches_hypergraph():
    """The graph-4L arm, FFN-widened, lands within <1% of the hypergraph-3L arm.

    Mirrors the closed form used in run_sweep / the plan to pick --graph-ffn-hidden:
    embeds C, then solve graph(H) = C + 4*(attn + 2*RMSNorm + FFN(H)) == P_hg.
    """
    device = torch.device("cuda")
    d, heads, vocab, msl = 256, 4, 893, 512

    hg = RecogsDecoderLM(vocab, d, heads, 3, "hypergraph_cuda", msl, scatter=False).to(device)
    p_hg = sum(p.numel() for p in hg.parameters())

    embeds = vocab * d + msl * d + d * vocab          # token + pos + lm_head
    attn = 4 * d * d + 2 * d                            # Wq,Wk,Wv,Wo (+ 2 biases)
    fixed = embeds + 4 * (attn + 2 * d)                 # + 2 RMSNorm/layer
    h = round(((p_hg - fixed) / 4 - d) / (2 * d + 1))

    graph = RecogsDecoderLM(vocab, d, heads, 4, "graph_flash", msl, ffn_hidden=h).to(device)
    p_graph = sum(p.numel() for p in graph.parameters())
    assert abs(p_graph - p_hg) / p_hg < 0.01, f"h={h} graph={p_graph} hg={p_hg}"

