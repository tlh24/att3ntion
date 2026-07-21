"""Measure the real hg:graph FLOP ratio on the ReCOGS train set and report what
graph config FLOP-matches the 3-layer hypergraph arm (RUN_3 / hg_gather).

Pure analytics: tokenizes train.tsv exactly as training does (prefix-LM packing,
max_seq_len truncation), then evaluates corrected closed-form forward-FLOPs for
both arms over the *actual* length distribution. Also simulates the bucketed
batch sampler so the hg N^3 cost is charged at the real *padded* batch length,
which is what the GPU actually computes.

No GPU, no model forward. mult-add counted as 2 FLOPs throughout.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
RECOGS = HERE.parent
sys.path.insert(0, str(RECOGS))

from data import (  # noqa: E402
    RecogsSequenceDataset,
    build_tokenizer_from_train,
    BucketedBatchSampler,
)

DATA_DIR = RECOGS / "data" / "raw"
MAX_SEQ_LEN = 512          # v3 regime
BATCH_SIZE = 32            # v3 micro-batch (dataloader batch)
BUCKET_MULT = 50           # data.py default
D = 256                    # d_model (both arms, v3)
H = 4                      # heads (both arms, v3)
F_DEFAULT = 3 * D          # 768, both arms' FFN in v3 (graph v3 used 736; immaterial here)


# ---- corrected closed-form forward FLOPs for ONE example at length N ----
# Shared per-layer pieces (both arms): FFN = 4*N*d*F ; 2x RMSNorm ~ 20*N*d (tiny).
def _ffn_flops(N, d, F):
    return 4 * N * d * F + 20 * N * d


def graph_layer_flops(N, d, F, heads):
    # QKV proj 6 N d^2 ; QK^T 2 N^2 d ; A.V 2 N^2 d ; Wo 2 N d^2
    attn = 8 * N * d * d + 4 * N * N * d
    return attn + _ffn_flops(N, d, F)


def hg_layer_flops(N, d, F, heads):
    # Q,R,S proj 6 N d^2 ; Vq,Vr,Vs proj 6 N d^2 ; Wo 2 N d^2  -> 14 N d^2
    # 3-way scores einsum bhid,bhjd,bhkd->bhijk : 3 H N^3 d_head = 3 N^3 d
    # 3 output einsums bhijk,bh.d,bh.d->bh.d     : 3 * (3 N^3 d) = 9 N^3 d
    # 3 softmaxes over N^3 elements              : ~5 * 3 * H * N^3 (small)
    proj = 14 * N * d * d
    three_way = 12 * N * N * N * d
    softmax = 15 * heads * N * N * N
    return proj + three_way + softmax + _ffn_flops(N, d, F)


def model_flops(N, d, F, heads, n_layers, vocab, layer_fn):
    f = n_layers * layer_fn(N, d, F, heads)
    f += 2 * N * d * vocab           # lm_head (shared)
    return f


def total_flops_over_epoch(lengths, d, F, heads, n_layers, vocab, layer_fn, padded_batches=None):
    """If padded_batches given (list of (count, padded_len)), charge each example
    at its batch's padded length. Else ideal (no padding)."""
    if padded_batches is None:
        return sum(model_flops(N, d, F, heads, n_layers, vocab, layer_fn) for N in lengths)
    return sum(cnt * model_flops(Lpad, d, F, heads, n_layers, vocab, layer_fn)
               for cnt, Lpad in padded_batches)


def simulate_bucketed_batches(lengths, batch_size, bucket_mult, seed=0):
    torch.manual_seed(seed)
    sampler = BucketedBatchSampler(lengths=lengths, batch_size=batch_size,
                                   bucket_multiplier=bucket_mult)
    out = []
    for batch in sampler:
        ls = [lengths[i] for i in batch]
        out.append((len(ls), max(ls)))
    return out


def fmt(x):
    return f"{x/1e12:8.2f} TF" if x >= 1e12 else f"{x/1e9:8.2f} GF"


def main():
    tok = build_tokenizer_from_train(DATA_DIR)
    vocab = tok.vocab_size
    ds = RecogsSequenceDataset(DATA_DIR / "train.tsv", tok, max_seq_len=MAX_SEQ_LEN)
    lengths = [len(ex.input_ids) for ex in ds.examples]
    n = len(lengths)
    trunc = sum(ex.was_truncated for ex in ds.examples)

    t = torch.tensor(lengths, dtype=torch.float64)
    pct = lambda p: int(torch.quantile(t, p).item())
    print(f"=== ReCOGS train set ({n} examples, vocab={vocab}, max_seq_len={MAX_SEQ_LEN}) ===")
    print(f"seq length N: min {int(t.min())}  mean {t.mean():.1f}  median {pct(0.5)}  "
          f"p90 {pct(0.9)}  p99 {pct(0.99)}  max {int(t.max())}  truncated={trunc}")
    S1, S2, S3 = float((t).sum()), float((t**2).sum()), float((t**3).sum())
    print(f"SumN={S1:.3e}  SumN^2={S2:.3e}  SumN^3={S3:.3e}")
    print(f"FLOP-weighting note: hg total ∝ SumN^3 (tail-dominated); "
          f"cube-weighted mean N = {(S3/S1)**0.5:.0f}\n")

    batches = simulate_bucketed_batches(lengths, BATCH_SIZE, BUCKET_MULT, seed=0)
    pad_waste = sum(c * lp for c, lp in batches) / S1 - 1
    print(f"bucketed batches: {len(batches)}  padded-token overhead vs ideal = {pad_waste*100:.1f}%\n")

    hg = lambda lay, dd=D, FF=F_DEFAULT, hh=H, pb=batches: total_flops_over_epoch(
        lengths, dd, FF, hh, lay, vocab, hg_layer_flops, pb)
    gr = lambda lay, dd=D, FF=F_DEFAULT, hh=H, pb=batches: total_flops_over_epoch(
        lengths, dd, FF, hh, lay, vocab, graph_layer_flops, pb)

    hg3 = hg(3)
    print(f"=== per-epoch forward FLOPs (padded/bucketed) ===")
    print(f"hg_gather  3L  d={D} F={F_DEFAULT}:  {fmt(hg3)}   <-- target\n")

    print(f"graph_flash, d={D} F={F_DEFAULT}, varying depth:")
    for L in (3, 4, 6, 8, 12, 16, 24, 32):
        g = gr(L)
        print(f"  {L:3d}L : {fmt(g)}   ratio hg/graph = {hg3/g:6.2f}x   "
              f"({'graph<hg' if g < hg3 else 'graph>=hg'})")
    print()

    # depth needed at d=256 to match
    g1 = gr(1)
    L_match = hg3 / g1
    print(f"depth to FLOP-match at d={D}: ~{L_match:.1f} layers "
          f"(graph 1L = {fmt(g1)})\n")

    print(f"graph_flash width scaling (heads=d/64, F=3d), matched depth options:")
    for L in (4, 6):
        print(f"  at {L} layers:")
        for d in (256, 384, 512, 640, 768, 1024, 1280):
            heads = max(1, d // 64)
            g = gr(L, d, 3 * d, heads)
            star = "  <== match" if 0.97 <= g / hg3 <= 1.03 else ""
            print(f"    d={d:5d} heads={heads:2d} F={3*d:5d}: {fmt(g)}  "
                  f"ratio graph/hg = {g/hg3:5.2f}x{star}")
        print()

    # find the width that FLOP-matches at 4L and 6L (fine search)
    print(f"exact FLOP-matched width (graph/hg in [0.99,1.01]):")
    for L in (4, 6):
        for d in range(256, 2049, 64):
            heads = max(1, d // 64)
            r = gr(L, d, 3 * d, heads) / hg3
            if r >= 1.0:
                print(f"  {L}L: d={d} heads={heads} F={3*d}  ratio={r:.3f}  "
                      f"(prev d={d-64} ratio={gr(L, d-64, 3*(d-64), max(1,(d-64)//64))/hg3:.3f})")
                break
    print()

    # the manager's specific proposal
    g6 = gr(6)
    print(f"=== manager's proposal: graph 6L vs hg 3L (both d={D}) ===")
    print(f"  graph 6L = {fmt(g6)} ;  hg 3L = {fmt(hg3)}")
    print(f"  graph 6L reaches {g6/hg3*100:.1f}% of hg FLOPs "
          f"(hg is {hg3/g6:.1f}x the 6L graph)")


if __name__ == "__main__":
    main()
