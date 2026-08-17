# ReCOGS: Hypergraph vs Graph Attention — Run 2 (v2)

**Date:** 2026-06-16 · **Setup:** 2 actl pods (`recogs` 4×H100 seeds 1–2, `att3ntion` 2×H100 seed 3), single-GPU per (arm, seed) via `run_sweep.py` (not DDP) · **Seeds:** 1–3 · **Status:** complete — superseded by [`RUN_3.md`](../v3_prefix_lm/RUN_3.md)

*Same question as [v1](../v1_pilot/RUN_1.md), at a corrected, interpretable regime, as a 3-arm scatter-ablation ladder. The v2 design plan is folded into the [Methodology & roadmap](#methodology--roadmap) section.*

## TL;DR
- **At the 512 regime the v1 result flips: graph ≳ hg-gather** — `graph_flash` 20.2 ± 9.8 vs `hg_gather` 16.7 ± 3.9 gen EM. v1's "+4.7 pt hypergraph" looks like a **low-resource artifact**.
- **The scatter arm (`hg_full`) diverges on every seed** (loss → ~10⁶; all metrics 0.00). Root-caused below — unbounded scatter output → residual blow-up → AdamW corruption. Not a hyperparameter issue.
- **512 removes truncation entirely, but structural gen stays ~0–2%** — clearing the length artifact did not unlock recursion/PP-attachment. A counterpoint to the "it's just sequence length" hypothesis.
- In-distribution solved by all healthy arms (dev/test ~98–99%).
- ⚠️ **Large cross-seed gen variance** at saturated dev (`graph_flash` 8.9 / 25.1 / 26.5) — `best.pt` selected on sub-0.01 dev-loss noise. **No headline is safe yet** without `best.pt`-vs-`last.pt` and ≥5 seeds.

## What changed from v1
1. **Regime fix:** `max_seq_len=512`, `max_new_tokens=512`, effective batch 128 (32 × `--grad-accum-steps 4`), ≤300 epochs, patience 15. Other hyperparameters as v1.
2. **3-arm scatter ablation:** `--scatter` wired through `model.py` → `train.py`. Contrasts: **G → H-gather** isolates 3-way arity; **H-gather → H-full** isolates the scatter/write.
3. **Clean baseline arm** (`graph_clean`, canonical pre-norm SDPA) added for literature comparability (available; not run this round).
4. **Phase-0 truncation gate** + **per-example `(idx, category, correct)` dump** for paired McNemar/bootstrap + **divergence guard** (`--divergence-loss-threshold`, default 1000).

## Setup
Held fixed across arms: `d_model=256`, `n_heads=4` (`d_head=64`), FFN 256→768→256, RMSNorm, learned absolute positions (512), bf16, vocab 893; report dev-selected `best.pt`. CUDA hypergraph requires `d_head ∈ {16,32,64}`, runs bf16 with no rotary, and is ~O(N³) (256→512 ≈ 8× its attention cost vs ~4× pairwise).

**Phase-0 gate (passed):** at 512, truncation (`3 + len(src) + len(lf) > max_seq_len`) drops to **0.00%** on every split including structural categories (gen `cp_recursion` 45.5%@256 → 0%@512). Removes truncation as a confound.

| arm | attn | layers | params | status |
|---|---|--:|--:|---|
| `graph_flash` | pairwise SDPA + post-GELU | 4 | 3.218M | ✅ ×3 |
| `hg_gather` | hypergraph_cuda, scatter=False | 3 | 3.152M | ✅ ×3 |
| `hg_full` | hypergraph_cuda, scatter=True | 3 | 3.744M | ❌ diverged ×3 |

graph_flash vs hg_gather within 2.1% params (clean arity contrast); `hg_full`'s extra params are intrinsic to scatter (doubled value projections). Depth is a reported confound favoring graph.

## Results
### Overall exact-match
| | graph_flash | hg_gather | hg_full |
|---|--:|--:|--:|
| gen s1 / s2 / s3 | 26.47 / 25.07 / 8.91 | 14.83 / 14.13 / 21.14 | diverged |
| **gen mean ± std** | **20.15 ± 9.76** | **16.70 ± 3.86** | — |
| dev / test (mean) | 98.85 / 98.88 | 98.60 / 98.66 | 0.00 |

graph_flash ≳ hg_gather (20.2 vs 16.7) — opposite of v1 — but graph swings 8.9 → 26.5, so not significant on 3 seeds.

### Lexical vs structural gen (mean over categories)
| split | graph_flash | hg_gather |
|---|--:|--:|
| Lexical (18 cats) | 23.40 | 19.26 |
| Structural (3 cats) | 0.67 | 1.37 |

Structural ~0–2% for both even with truncation gone — essentially unchanged from v1@256. Full per-category tables in each `final_metrics.json`.

### vs v1
| | v1 (256, gather-only) | v2 (512) |
|---|---|---|
| graph gen (PM) | 11.20 | **20.15 ± 9.76** |
| hg-gather gen | 15.87 | **16.70 ± 3.86** |
| headline | hg +4.7 pt | graph ≳ hg (n.s.) |

The regime fix ~doubled graph's gen (11 → 20) while hg-gather barely moved (15.9 → 16.7), consistent with v1's hg edge being under-resourcing, not arity.

## Analysis: the `hg_full` divergence
Localized with `diagnose_scatter.py` (per-layer activation + grad norms):
- **Scatter attention output is unbounded** — while loss is still ~2.3, a layer's attn-output RMS spikes O(1) → hundreds–thousands (591 → 791 → 1034 at steps 271/276/281); gather/graph stay ~O(1).
- Pure pre-norm (`x = x + attn(norm1(x))`) has **no norm between attn output and the residual add**, so a spike dumps straight into the residual stream.
- Pre-clip grad norms hit 600–2450 (healthy ~5); grad-clip caps the step but **AdamW's 2nd-moment ingests the huge grad²** and is poisoned until divergence. Lower LR diverged *earlier* — onset is stochastic (when a spike lands, not LR).
- **Implied fix (not applied):** bound the attn sublayer output before the residual add (post-/sandwich RMSNorm on `attn_output`, or a small-init learned residual scale). The scatter contrast can't be drawn until this lands.

## Caveats
- **3 seeds, large variance, saturated-dev selection** — `best.pt` on sub-0.01 dev noise is near-random; need `best.pt`-vs-`last.pt` + ≥5 seeds.
- **`hg_full` has no valid result** — 0.00 is a divergence artifact; the headline scatter mechanism is untested pending the stabilizer.
- **Not FLOP-matched** (hg ~N³; ~8× the gap vs 256). Cross-pod seed 3 (md5-identical code) is selection noise, not environment.
- graph arms run the SDPA **math** backend (flash rejects the `(B,N,N)` mask) → CPU-bound, slower than the GPU-pinned hg arms.

## Methodology & roadmap
*(folded in from the separate v2 design plan)*
- **Two uncertainties — don't conflate:** within-run sampling noise over ~21k gen examples (paired McNemar/bootstrap; ~14σ for the v1 gap, *not* the binding constraint) vs across-seed variance (init + data order + ckpt selection) — the latter is what licenses a headline.
- **Acceptance gate (not cleared):** the clean baseline must hit ~literature lexical gen (>90%) before the comparison is meaningful; at ~19–23% here, treat all comparisons as provisional.
- **Literature (verify before quoting):** test ~>90% (often ~99%); lexical gen mostly >90%; structural ~0 unless length is fixed, then ~`obj_pp_to_subj_pp` 20% / `pp_recursion` 40% / `cp_recursion` 52%; RASP-equivalents ~100%. Refs: Wu, Manning & Potts, "ReCOGS" (TACL 2023); RASP study (arXiv:2504.15349).
- **Roadmap:** Phase 0 ✅, Phase 2 (regime + ladder) partial. Next: `best.pt`-vs-`last.pt` + ≥5 seeds; Phase 3 FLOP/compute-matched frontiers; Phase 4 scaling (≥3 sizes × 3 arms); Phase 5 per-category error analysis + honest (length-conditioned) structural result. The scatter fix gates the H-full contrast.

## Reproduction
From `experiments/recogs/` with `PYTHONPATH` at the repo root, on a multi-GPU pod:
```sh
python truncation_report.py                                           # Phase-0 gate
TAG=run GPUS=0,1,2,3 ARMS=graph_flash,hg_gather SEEDS=1,2,3 bash scripts/run_overnight.sh
python run_sweep.py --arms hg_full --seeds 1,2,3 --gpus 0,1,2,3 \
  --extra-args "--lr 3e-5 --warmup-steps 1000 --grad-clip 0.5"        # diverges pending the attn-output norm fix
python diagnose_scatter.py --seed 2 --lr 3e-5                          # localize divergence
```
Each run writes `final_metrics.json` (overall + per-category, dev/test/gen) and `per_example_<split>.json` (paired-test records).
