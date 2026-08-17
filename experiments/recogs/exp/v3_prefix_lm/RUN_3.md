# ReCOGS: Hypergraph vs Graph Attention — Run 3 (v3)

**Date:** 2026-06-17 · **Setup:** 2 actl pods (`recogs` 4×H100 `hg_gather`, `att3ntion` 2×H100 `graph_flash`), single-GPU per (arm, seed); batched generation · **Seeds:** 1–10 · **Status:** in progress — results pending

*True prefix-LM + tight param-match, 2 arms × 10 seeds. Restarted the stopped v3 attempt; results to be written up against [`RUN_2.md`](../v2_three_arm/RUN_2.md).*

## TL;DR
- Two arms only — **graph_flash (4L)** vs **hg_gather (3L)**; `hg_full` deferred (diverges, see [RUN_2](../v2_three_arm/RUN_2.md)).
- **Tight param-match via FFN width:** graph 4L `--graph-ffn-hidden 736` = 3,152,256 vs hg 3L 3,151,872 (**+0.01%**, was +2.1% at 768).
- **True prefix-LM:** `[BOS] src [SEP]` bidirectional, LF span causal (v2 was only partly prefix).
- **Overnight-fit blocker solved with batched generation** (not a KV cache) — uniform speedup across both arms, no kernel change.
- ⚠️ Results pending — gen EM / significance get written into this doc when all 20 runs finish.

## What changed from v2
1. **Batched generation** (`evaluate.greedy_decode_batch` + rewritten `evaluate_split_generation`): decode `batch_size` examples/forward instead of one. Length-bucketed by prefix, prefix-LM-correct, preserves stable global index + DDP sharding + per-example records. Token-for-token equal to per-example decode (`tests/test_recogs_eval.py`).
2. **True prefix-LM** mask (the v3 regime change; see Setup invariants).
3. **Tight param-match** via `--graph-ffn-hidden 736`.
4. **Min-delta early-stop** (`--early-stop-min-delta 1e-4`) so microscopic dev lows stop resetting patience.
5. **Epoch caps** 50 (hg) / 40 (graph) — past ~epoch-30 convergence, sized to the overnight window.
6. **`MNT=384`** (max real LF = 364 tok), **`timing.json`** projections, **`monitor_v3.py`** live status + loss curve.

Why not a KV cache: a true cache for the 3-way O(N³) kernel needs a new incremental CUDA kernel (no existing API) — too risky for this run; batched generation gives the speedup with no kernel change.

## Setup
| pod | arm | seeds | GPUs | epochs |
|---|---|---|---|--:|
| recogs (4×H100) | `hg_gather` | 1–10 | 0–3 | 50 |
| att3ntion (2×H100) | `graph_flash` | 1–10 | 0,1 | 40 |

Common regime: true prefix-LM, seq 512, batch 32 × accum 4 (eff 128), `EVAL_DEV_EVERY=100`, `PATIENCE=60`, `MIN_DELTA=1e-4`, `MNT=384`, `--graph-ffn-hidden 736`, `--out-dir runs/v3`, `--eval-at-end`. Different epoch caps are intentional (both past convergence; `best.pt` is dev-selected). Measured: hg 272 s/epoch, graph 210 s/epoch.

**Invariants (regime correctness — keep true on resume):**
- **Prefix-LM mask must match train and eval:** `data.py::collate_decoder_only(prefix_lm=True)` builds `allowed = causal | (key < prefix_len)` then `& valid`; `evaluate.py::greedy_decode_*` must mirror it (cols `[:prefix_len] = True`). Drift = generation silently sees a different mask than training. `prefix_len` rides on `EncodedExample` + `config.json`.
- **Verified gates (pytest, CUDA):** kernel honors the non-causal prefix mask (`test_hypergraph_cuda_prefix_lm_directionality`); param-match <1% (`test_graph_ffn_hidden_param_matches_hypergraph`); batched-gen equivalence (`tests/test_recogs_eval.py`).
- **Split roles:** dev = selection (drives `best.pt` + early stop); test = held-out **monitor-only** (the moment a decision keys on it, it's no longer unbiased); gen = OOD headline, final EM only.
- **EOS/PAD:** distinct — EOS (3) supervised; PAD (0) masked from loss/attention/metrics.
- **Param-match is dim-dependent:** recompute `--graph-ffn-hidden` if dims change.

## Results
**Pending — run in progress.** Convergence is healthy (hg s1 dev/test loss 5.0 → 0.008 by ~step 3000 / ~epoch 16). Final gen EM (mean ± std over 10 seeds), lexical/structural breakdown, per-category table, and McNemar/bootstrap significance go here once all 20 runs write `final_metrics.json`.

## Analysis
*Pending results.*

## Caveats
- No final numbers yet; everything above Results is regime/method, not outcome.
- Still not FLOP-matched (hg ~N³); controls parameters, not compute — as in v1/v2.
- `att3ntion` pod has no auto-sync → graph run dirs need a manual pull before aggregation.

## What's next (results pipeline)
1. Pull `att3ntion` graph dirs into recogs `runs/v3/` (separate PVC, no auto-sync).
2. `python aggregate.py --runs-dir runs/v3 --output-json aggregate_summary_v3.json` → per-split + per-category mean±std.
3. `python plots/plot_gen_comparison.py` → `docs/graph_vs_hypergraph_gen_v3.png`; `python scripts/monitor_v3.py --plot` → loss-trajectory figure.
4. McNemar + bootstrap CI on paired `per_example_gen.json` (keyed by stable idx).
5. Fill in the Results and Analysis sections above.

## Reproduction
From `experiments/recogs/`, detached overnight (survives disconnect):
```sh
# recogs / hg:
TAG=v3_hg GPUS=0,1,2,3 ARMS=hg_gather SEEDS=1,2,3,4,5,6,7,8,9,10 \
  EVAL_DEV_EVERY=100 PATIENCE=60 EPOCHS=50 \
  EXTRA="--prefix-lm --graph-ffn-hidden 736 --out-dir runs/v3 --eval-at-end" \
  bash scripts/run_overnight.sh
# att3ntion / graph: ARMS=graph_flash EPOCHS=40, same EXTRA
```
Watch with `python scripts/monitor_v3.py` (`--plot` for the loss-curve PNG); per-run signals in `runs/v3/<run>/eval_curve.tsv` + `timing.json`.
