# ReCOGS: Hypergraph vs Graph Attention — Run 1 (v1 pilot)

**Date:** 2026-06-15 · **Setup:** 4×H100 (actl `recogs` pod), DDP via `torchrun` (global batch 64) · **Seeds:** 1 · **Status:** complete — superseded by [`RUN_2.md`](RUN_2.md)

*Does 3-way **hypergraph** attention generalize better than pairwise **graph** attention on ReCOGS compositional generalization (prefix-LM decoder)? First valid pilot; the v2 plan that reviewed it is folded into [`RUN_2.md`](RUN_2.md).*

## TL;DR
- **At matched params (~3.1M), hypergraph beats graph on OOD gen EM: 15.87% vs 11.20% (+4.7 pt).** Survives param-matching, so not a capacity artifact.
- The edge is **entirely lexical**; on **structural** gen (recursion, novel PP-attachment) every model fails (~0–3%), even an over-parameterized graph.
- **Graph scales better with raw capacity:** a 6-layer graph (+45% params) hits 25.8% gen, but via a volatile, likely seed-sensitive per-category profile.
- In-distribution (dev/test) ~98–99% for all configs — not a discriminating signal.
- ⚠️ **Single seed.** The matched-param result is the trustworthy headline; the 6-layer number needs replication.

## What changed (why this rerun was needed)
The earlier Jun-12 run was **invalid**: it used an *unshifted* cross-entropy loss (`logits[t]` vs `labels[t]` instead of `labels[t+1]`). Since `labels[t] == input_ids[t]` at supervised positions in a decoder-only LM, the objective was solved by copying the current token — fake train loss 0.0000, 0% gen EM. This run uses the corrected shifted loss (dev loss 0.80 → ~0.003, non-zero gen EM).

## Setup
Held fixed across arms: `d_model=256`, `n_heads=4`, FFN 256→768→256, RMSNorm, learned absolute positions, bf16; AdamW lr 1e-4, warmup 400 + cosine, grad-clip 1.0, `max_seq_len=256`, vocab 893; ≤100 epochs, early-stop patience 10 on dev loss; report **dev-selected `best.pt`** (gen never used for selection). Arms differ only in the attention operator (and depth for the matched controls). Infra this round: DDP training + distributed bucketed sampler (global batch 64, ~4× speedup, same optimizer dynamics as 1-GPU batch-64), sharded gen eval, and a fixed variable-bijection SEM scorer (invariant to conjunct order + variable renaming).

| arm | attn | layers | params | note |
|---|---|--:|--:|---|
| hypergraph_cuda | 3-way ~O(N³) | 3 | 3.086M | headline |
| graph_flash | pairwise ~O(N²) | 3 | 2.495M | base |
| graph_flash (param-matched) | pairwise | 4 | **3.152M** (+2.1%) | the controlled contrast |
| graph_flash (capacity stress) | pairwise | 6 | 4.467M (+45%) | "graph with more capacity" |

Param-count is the control; FLOP-matching is infeasible (hypergraph ~N³ vs graph ~N²).

## Results
### Overall exact-match
| | hg 3L | graph 3L | graph 4L (PM) | graph 6L |
|---|--:|--:|--:|--:|
| dev | 98.67 | 97.93 | 98.97 | 99.00 |
| test | 98.67 | 98.17 | 98.67 | 99.37 |
| **gen (OOD)** | **15.87** | 12.07 | **11.20** | **25.80** |

Controlled (~3.1M): **hg 15.87 vs graph-4L 11.20 (+4.7 pt)**. Graph 3L→4L barely moves; depth alone doesn't help graph until 6L.

### Lexical vs structural gen (mean over categories)
| split | hg 3L | graph 3L | graph 4L (PM) | graph 6L |
|---|--:|--:|--:|--:|
| Lexical (18 cats) | **18.4** | 14.0 | 13.0 | 29.6 |
| Structural (3 cats) | 0.8 | 0.6 | 0.4 | 3.0 |

Structural = `pp_recursion`, `cp_recursion`, `obj_pp_to_subj_pp`. hg's lexical edge concentrates in `prim_to_subj_proper` (81.9 vs 0.3 for graph-4L) and the unaccusative-subject / dative-alternation splits; full per-category tables in each run's `final_metrics.json`.

## Analysis
- **Hypergraph's OOD edge is real per-parameter** (15.9 vs 11.2), driven by lexical categories.
- **Structural gen is unsolved by all** (~0–3%) regardless of attention or capacity — the known hard core of COGS/ReCOGS for small from-scratch models.
- **Graph trades capacity for lexical gen:** at 6L it *solves* `prim_to_subj_common` (94.8) and `prim_to_obj_common` (93.8) but *collapses* on `pp_dative_to_do_dative` (3.2 vs hg 57.5) — a spiky, likely seed-sensitive solution basin.
- In-distribution is uninformative (~98–99%); only `gen` discriminates.

## Caveats
- **Single seed.** Matched-param headline is trustworthy; per-category and the 6L numbers need seed averaging.
- **Not FLOP-matched** — controls parameters, not compute (hg ~N³). "Wins per parameter," not per FLOP.
- No per-arm tuning; arms share every hyperparameter except attention and (for controls) depth.

## Reproduction
From `experiments/recogs/` with `PYTHONPATH` at the repo root, on a 4×H100 pod:
```sh
bash scripts/run_ddp.sh                       # headline arms (hg 3L + graph 3L)
torchrun --standalone --nproc_per_node=4 train.py --attn graph_flash --seed 1 \
  --layers 4 --batch-size 16 --epochs 100 --early-stop-patience 10 \
  --eval-dev-every 400 --eval-at-end --log-name pm4   # param-matched (--layers 6 --log-name g6 for the stress test)
```
Each run writes `runs/<log_name>_<attn>_s<seed>/final_metrics.json` (overall + per-category, dev/test/gen). Smoke gate: `SMOKE=1 bash scripts/run_ddp.sh`.
