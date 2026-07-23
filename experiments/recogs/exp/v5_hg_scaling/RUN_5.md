# ReCOGS: Hypergraph Chinchilla-style Scaling — Run 5 (v5)

**Date:** 2026-06-26 · **Setup:** scale the hypergraph (gather) arm across a width ladder
(head_dim fixed at 64), reuse the RUN_3 3L/d256 hg run as the ladder anchor, add 12 new runs
across 3 pods (10×H100). · **Status:** ✅ COMPLETE — all 12 new runs finished (failures=0); results below

> **Run notes:** RUN_3/RUN_4 compared hg vs graph at *single points*; RUN_5 measures the
> **scaling behavior of hypergraph alone** (gather only, no scatter). Reads as a Chinchilla
> **Approach 1 (training-curve envelope)**: train a ladder of hg sizes and use each run's
> already-logged dev/test/train-loss-vs-step trajectory as its loss-vs-compute curve; the
> lower envelope across the ladder is the compute-optimal frontier. Per-category OOD
> generalization is read at convergence from `final_metrics.json` (inline gen) per rung.

## TL;DR
- **🏁 Answer (full ladder @ 10–18 seeds/rung, gen = OOD EM):** hg OOD generalization **scales
  monotonically with width** — d128 **0.71%** → d256 **3.9%** → d384 **6.8%** → d512 **8.3%** →
  d768 **10.9%**. (Updated with Run 5b seeds; means dropped ~1–2 pts vs the earlier small-n
  estimates — see §5b and §Results.)
- **Width ≫ depth per unit compute (headline):** at ~equal FLOPs (~500 TF), wide **3L/d512 = 8.3%**
  beats deep **6L/d256 = 2.4%** by **~3.5×** → wide-shallow is the compute-optimal way to spend
  hg's O(N³·d) budget, as the cost structure predicted.
- **No clear plateau yet:** gen keeps rising through d768 (d512 8.3% → d768 10.9%); the earlier
  "plateau ~10–11% / compute-optimal at d512" call was an artifact of d512's optimistic 4-seed
  mean and does NOT survive the 12-seed rerun. d768 is still only 2 offline-gen seeds — treat the
  top of the ladder as tentative until the 5b d768 seeds (s3–10) are scored.
- **Sanity:** d256 mean is now 3.87% over 18 seeds (v3 s1–10 + v5 s11–18); the original 10-seed
  v3 number was 4.77%. All 3L rungs now carry 10–18 seeds, so their bands are trustworthy; only
  **6L/d256 (2 seeds)** and **d768 (2 offline-gen seeds)** remain thin.
- See **Results** below for the table + figures (`docs/v5_{loss_curves,scaling,gen_vs_test}.png`).

- **Question:** how does hypergraph attention scale — what's the compute-vs-loss frontier, and
  does OOD generalization improve with size? RUN_3/RUN_4 gave single points; RUN_5 gives a curve.
- **Why hg scaling is special:** hg attention is **O(N³·d)** — *linear* in width `d` — so width
  buys parameters cheaply in FLOP terms (d768 = 7.9× params for only 3.2× FLOPs), the opposite
  of a standard transformer (O(N²·d), width is FLOP-expensive). The ladder tests whether
  **compute-optimal hg favors wide-shallow models**.
- **Ladder = pure-width** (3L, head_dim=64, heads=d/64, FFN=3·d) from d128→d512, anchored by the
  reused 3L/d256, **+ one iso-FLOP depth control** (6L/d256 ≈ same FLOPs as 3L/d512, very
  different shape) **+ a d768 range-extender** (train-only overnight, offline gen).
- **Logging already does everything** — no `train.py` change: `eval_curve.tsv` carries
  `step  train_loss  dev_loss  test_loss` every 100 steps (test loss IS tracked during
  training); `final_metrics.json` + `per_example_{dev,test,gen}.json` carry per-category EM and
  per-example correctness at `--eval-at-end`.

## The ladder (head_dim=64 throughout, FFN=3·d, 50 epochs)

Per-epoch fwd FLOPs from `scripts/flop_match.py` (real ReCOGS length dist, bucketed/padded);
per-seed hours ≈ measured anchor 6.1h × TF/252 (4.77h train + 1.34h inline gen at d256):

| rung | layers | d_model | heads | params | TF/epoch | ×FLOP | ~h/seed |
|---|--:|--:|--:|--:|--:|--:|--:|
| R0 | 3 | 128 | 2 | 0.94M | 124 | 0.49× | ~3.0 |
| R1 (reuse v3) | 3 | 256 | 4 | 3.15M | 252 | 1.00× | — (have 10 seeds) |
| R2 | 3 | 384 | 6 | 6.65M | 385 | 1.53× | ~9.3 |
| R3 | 3 | 512 | 8 | 11.4M | 521 | 2.07× | ~12.6 |
| R4 (depth control) | 6 | 256 | 4 | ~6.3M | 503 | 2.00× | ~12.2 |
| R5 (extender) | 3 | 768 | 12 | 24.8M | 807 | 3.20× | ~19.6 (train-only) |

Param range R0→R3 ≈ **12×**, FLOP range ≈ **4.2×**; d768 pushes params to ~26×. Smoke test
confirmed R3 builds at exactly **11.416M params** and fits 80 GB at micro-batch 32 with huge
headroom (peak ~5.3 GB — the flash-style kernel never materializes the N³ score tensor).

## Seed allocation & GPU map (10×H100 across 3 pods)

Cluster only fit **one** more `4x` node (g256) → 10 GPUs total, not the 14 originally scoped.
Seeds: loss curves are low-variance (≥2 ample); OOD gen EM is high-variance (observed 0.5–12.3%
spread), so the well-sampled points are the reused d256 (10) and d512 (4).

| pod | GPUs | invocation | runs | ~makespan |
|---|--:|---|--:|--:|
| **recogs** | 4 | `ARMS=hg_d512 SEEDS=1,2,3,4 GPUS=0,1,2,3` | 4× d512 | ~12.6h |
| **recogs-scale** | 4 | `ARMS=hg_6L256,hg_d384,hg_d128 SEEDS=1,2 GPUS=0,1,2,3` | 2×6L256 + 2×d384 + 2×d128 | ~12.3h |
| **att3ntion** | 2 | `ARMS=hg_d768 SEEDS=1,2 GPUS=0,1 EXTRA=--no-eval-at-end` | 2× d768 (train-only) | ~15.3h |

New runs: **12** (d128×2, d384×2, d512×4, 6L256×2, d768×2) + reused d256×10. Main ladder
(everything but d768) done in **~12.6h** (ready by morning); d768 trains ~15.3h then is scored
offline next day. Total ≈ 130 GPU-hours.

## Results

gen = OOD exact-match (the headline), test = in-distribution. Mean over seeds [min–max].
Per-epoch FLOPs from `flop_match.py`; figures in `docs/v5_{loss_curves,scaling,gen_vs_test}.png`
(generate via `plots/plot_v5_scaling.py`). Per-category OOD heatmap (21 cats × 6 rungs):
`docs/v5_per_category.png` via `plots/plot_v5_per_category.py` — gains concentrate in
verb-argument-structure/lexical categories (e.g. only_seen_as_unacc_subj 6.8%→51.8% d128→d512);
structural-recursion categories stay low across all sizes.

| rung | params | TF/ep | **gen EM** | test EM | seeds |
|---|--:|--:|--:|--:|--:|
| 3L/d128 | 0.94M | 124 | **0.71%** [0.00–2.38] | 83.9% | 10 |
| 3L/d256 (v3+v5) | 3.15M | 252 | **3.87%** [0.48–12.26] | 95.7% | 18 |
| 3L/d384 | 6.65M | 385 | **6.77%** [1.93–14.79] | 97.7% | 10 |
| 3L/d512 | 11.4M | 521 | **8.30%** [3.62–12.36] | 98.6% | 12 |
| 3L/d768 | 24.8M | 807 | **10.89%** [10.20–11.59] | 99.0% | 2 (offline) |
| 6L/d256 (iso-FLOP ctrl) | 6.3M | 503 | **2.39%** [1.25–3.53] | 97.1% | 2 |

> Numbers above updated 2026-07-01 with Run 5b seeds (figures regenerated via
> `plots/plot_v5_scaling.py` + `plots/plot_v5_per_category.py`). d768/6L256 unchanged (not
> extended / offline-gen pending). Earlier small-n values are preserved in the §5b history below.

**Findings** (updated with Run 5b — all 3L rungs now 10–18 seeds):
1. **OOD generalization scales monotonically with width:** 0.71 → 3.87 → 6.77 → 8.30 → 10.89%.
   In-distribution test rises in lockstep and reaches near-ceiling (98–99%) from d384 up;
   d128 underfits even in-distribution (84%).
2. **Width ≫ depth per unit compute (the headline).** At ~equal FLOPs (~500 TF), wide
   **3L/d512 = 8.3%** vs deep **6L/d256 = 2.4%** — a ~3.5× OOD advantage for spending hg's
   O(N³·d) budget on width rather than depth. Direct support for wide-shallow being
   compute-optimal for hypergraph attention.
3. **No plateau yet in this regime:** d512 (8.3%) → d768 (10.9%) still rises with +13.4M params.
   The earlier "plateau ~10–11% / compute-optimal at d512" claim was driven by d512's optimistic
   4-seed mean (10.03%) and does **not** survive the 12-seed rerun (8.30%). d768 is still only 2
   offline-gen seeds, so the top of the ladder stays tentative pending the 5b d768 scoring.
4. **Adding seeds lowered the thin rungs by ~1–2 pts** (regression from optimistic small-n):
   d256 4.77→3.87% (10→18), d384 8.36→6.77% (2→10), d512 10.03→8.30% (4→12), d128 0.74→0.71%
   (2→10). The monotonic width trend and the width≫depth headline are unchanged in direction.
5. **Variance now well-sampled** on every 3L rung (10–18 seeds; d384 still spans 1.93–14.79,
   d512 3.62–12.36). Only **6L/d256** and **d768** remain 2-seed and should be quoted as ranges.

## Common regime (matches RUN_3/RUN_4 / v3)
True prefix-LM, `--max-seq-len 512`, batch 32 × accum 4 (eff 128), `--eval-dev-every 100`,
`--eval-max-new-tokens 384` (pinned — launcher default is 512), `--out-dir runs/v5`,
`--epochs 50`, **`--early-stop-patience 0`** (early stop OFF → full fixed-epoch trajectories for
the envelope). Main ladder uses inline `--eval-at-end`; d768 uses `--no-eval-at-end` (train-only).

## Prerequisites (done)
1. **`run_sweep.py` arms added:** `hg_d128/hg_d384/hg_d512/hg_6L256/hg_d768` (head_dim=64,
   FFN=3·d). `--no-eval-at-end` toggle added (makes the inline `--eval-at-end` conditional).
2. **Driver `scripts/run_v5.sh`** — generic per-pod, detached, 4× crash-retry, `--skip-existing`.
3. **Smoke test** (R3/d512 on recogs): builds, 11.4M params, memory fine. ✅
4. **recogs-scale pod** (3rd, 4×H100): repo synced (devspace), CUDA extension built.
5. **Reuse R1:** 10 `hg_gather_hypergraph_cuda_s{1..10}` dirs in `runs/v3` (local + recogs pod).

## Invariants (keep true)
- **head_dim=64 on every rung** — the only per-head size the CUDA kernels support {16,32,64};
  width scales via head count (heads=d/64), never head_dim.
- **Identical regime to v3** (prefix-LM, seq 512, eff-batch 128, MNT 384, FFN=3·d) — the only
  intended differences across rungs are depth/width.
- **Envelope tolerates heterogeneous training lengths** but every kept rung here runs full 50
  epochs (no epoch caps) so OOD EM is a converged-model property and comparable.
- **`--skip-existing` keys on `final_metrics.json`** → d768 (train-only, no final_metrics) is
  NOT auto-resumable mid-training; `best.pt`/`last.pt` are the rescue artifacts.
- **Split roles unchanged:** dev = selection + early stop (off here); test = monitor-only;
  gen = OOD headline EM. Paired McNemar/bootstrap via shared-idx `per_example_gen.json`.

## Aggregation & plotting
- **Do NOT use `aggregate.py`** — it groups by `attn` substring only, so all hg rungs collapse
  into one average. Use `scripts/gen_summary.py <base> '<per-arm glob>'` + the
  `plots/recogs_plot_common.py` loaders (`eval_curves`, `mean_band`, `per_category`, `overall`),
  keyed on the run-dir glob (the RUN_4 path).
- **New plot scripts to add** (local, post-hoc, non-blocking): train+dev+test loss vs step per
  rung (no existing script plots all three); **gen-EM/loss vs FLOPs on a log-x compute axis**
  with the lower envelope (no compute-axis plot exists); per-category OOD across the rung ladder
  (extend `plot_gen_test_ladder.py`).

## Caveats to state in results
- **Fixed tiny dataset (~27k ex):** extra compute = more epochs = *repeated data*, not fresh
  tokens. The loss-vs-compute envelope is valid; the params-vs-tokens Chinchilla interpretation
  is weaker. Say so plainly.
- **OOD gen EM high-variance**; R0/R2/R4/R5 at 2 seeds are ranges, not point estimates. Report
  median + min/max, not mean±sd.
- **6L/d256 vs 3L/d512 (iso-FLOP):** if their dev-loss envelopes coincide → loss is
  compute-determined (Chinchilla holds); if they diverge → flag shape-confounding.
- No auto-sync on pods → results pulled manually (`runs/v5/*`).

## Reproduction
```sh
# recogs (4 GPU), main heavy rung:
ARMS=hg_d512 SEEDS=1,2,3,4 GPUS=0,1,2,3 setsid nohup bash scripts/run_v5.sh ... &
# recogs-scale (4 GPU):
ARMS=hg_6L256,hg_d384,hg_d128 SEEDS=1,2 GPUS=0,1,2,3 setsid nohup bash scripts/run_v5.sh ... &
# att3ntion (2 GPU), d768 train-only:
ARMS=hg_d768 SEEDS=1,2 GPUS=0,1 EXTRA=--no-eval-at-end setsid nohup bash scripts/run_v5.sh ... &
```
Progress: `runs/v5/<run>/eval_curve.tsv` + `sweep_logs/v5_*_driver.log`. d256 anchor reused from
`runs/v3`. d768 offline gen next day on `best.pt` via `evaluate.py` (MNT 384).

---

## Seed Extension (Run 5b) — 8 additional seeds per 3L rung

**Date:** 2026-06-29 · **Setup:** single 4×H100 pod (recogs-devspace), 4 sequential phases via
`scripts/run_v5b.sh`. · **Status:** 🔄 IN PROGRESS (as of 2026-07-01, pid 37939) — Phases 1–3
✅ done (d128/d384/d256/d512 all at target seeds, failures=0); Phase 4 (d768 train-only, s3–10)
running. Updated 3L-rung gen EM filled into §Results above + the table below.

**Motivation:** d384 and 6L256 had only 2 seeds (high variance; d384 swung 1.9↔14.8%). Even d512
(4 seeds) and d128 (2 seeds) benefit from more. Target: 10 seeds on every 3L rung for comparable
uncertainty across the ladder. 6L256 (depth control) is excluded — the 2-seed range already
brackets the 4× gap to d512; the headline finding is robust without it.

### New seed allocation

| rung | arm | existing seeds | **new seeds** | **total** | out-dir |
|---|---|--:|---|--:|---|
| 3L/d128 | `hg_d128` | s1–2 (v5) | **s3–s10** | 10 | `runs/v5` |
| 3L/d256 | `hg_gather` | s1–s10 (v3) | **s11–s18** | 18 | `runs/v5` (new) + `runs/v3` (old) |
| 3L/d384 | `hg_d384` | s1–2 (v5) | **s3–s10** | 10 | `runs/v5` |
| 3L/d512 | `hg_d512` | s1–4 (v5) | **s5–s12** | 12 | `runs/v5` |
| 3L/d768 | `hg_d768` | s1–2 (v5) | **s3–s10** | 10 | `runs/v5` (train-only → offline gen) |

d256 new seeds land in `runs/v5/hg_gather_hypergraph_cuda_s{11..18}/`; combine with `runs/v3`
seeds when aggregating. d768 uses `--no-eval-at-end` (same as original); score offline via
`evaluate.py --eval-max-new-tokens 384` on `best.pt` after Phase 4 completes.

### Phase schedule (single pod, 4×H100, sequential)

| phase | arms | seeds | jobs | ~GPU-h | ~wall |
|---|---|---|--:|--:|--:|
| 1 | `hg_d128,hg_d384` | 3–10 | 16 | ~98 | ~25h |
| 2 | `hg_gather` | 11–18 | 8 | ~49 | ~12h |
| 3 | `hg_d512` | 5–12 | 8 | ~101 | ~25h |
| 4 | `hg_d768` (train-only) | 3–10 | 8 | ~122 | ~31h |
| **total** | | | **40** | **~370** | **~93h (~4 days)** |

Phases are sequential (all share GPUS=0,1,2,3); driver launched 2026-06-29:

```sh
cd /home/dev/workspace/experiments/recogs
setsid nohup bash scripts/run_v5b.sh >/dev/null 2>&1 </dev/null & echo "pid $!"
# pid 37938
```

Progress: `sweep_logs/v5b_master.log` (phase transitions) + `sweep_logs/v5b_p{1..4}_driver.log`
(per-phase run_sweep output) + `sweep_logs/hg_d{128,384,512,768}_s{N}.log` (per-run).

### Updated results (pending — fill in after phases complete)

Same table format as §Results above. Run `scripts/gen_summary.py` across the combined seed sets.
d256 aggregation: glob both `runs/v3/hg_gather_hypergraph_cuda_s*` and
`runs/v5/hg_gather_hypergraph_cuda_s*`. d768 gen EM: score `best.pt` offline after Phase 4.

| rung | params | seeds (total) | **gen EM** [min–max] | test EM | status |
|---|--:|--:|--:|--:|---|
| 3L/d128 | 0.94M | 10 | **0.71%** [0.00–2.38] | 83.9% | ✅ done |
| 3L/d256 | 3.15M | 18 | **3.87%** [0.48–12.26] | 95.7% | ✅ done (v3 s1–10 + v5 s11–18) |
| 3L/d384 | 6.65M | 10 | **6.77%** [1.93–14.79] | 97.7% | ✅ done |
| 3L/d512 | 11.4M | 12 | **8.30%** [3.62–12.36] | 98.6% | ✅ done |
| 3L/d768 | 24.8M | 2 → 10 | **10.89%** [10.20–11.59] (2 seeds) | 99.0% | 🔄 Phase 4 training s3–10; offline-gen of s3–10 pending |

**Note on shifts vs the original §Results estimates:** every 3L rung's mean came *down* as seeds
grew (regression from optimistic small-n) — d256 4.77→3.87%, d384 8.36→6.77%, d512 10.03→8.30%,
d128 0.74→0.71%. Direction of the scaling trend and the width≫depth headline are unchanged; the
"plateau at d512" framing is retracted (see §Results finding 3). Figures regenerated 2026-07-01.
d768's 8 new seeds still need offline `evaluate.py --eval-max-new-tokens 384` scoring on `best.pt`
after Phase 4 finishes (~2026-07-02), after which re-pull `runs/v5` and re-run the plot scripts.
