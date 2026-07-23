# ReCOGS: Hypergraph vs Graph Attention — Run 4 (v4): FLOP-Matched Graph Scaling

**Date:** 2026-06-22 (results 2026-06-23) · **Setup:** reuse the existing RUN_3 hg + graph runs as curve anchors; add 2 new scaled graph arms × 10 seeds · **Status:** ✅ **COMPLETE** — all 20 v4 runs finished (failures=0), results below

> **Run notes (recogs-only):** att3ntion left idle for the launch — recogs had all 4 GPUs free and already holds hg v3, so no cross-pod gather was needed. Both new arms (`graph_flop6L`, `graph_6L256`) ran on recogs's 4 GPUs via `scripts/run_v4_recogs.sh` (detached; calls `run_sweep.py` directly with 4× crash-retry + `--skip-existing`; auto-plots on completion). Smoke test confirmed `graph_flop6L` = **73.58M params**. Sweep ran ~22:36→15:32 UTC (~17 h; the graph prefix-LM mask disables the flash SDPA kernel → standard SDPA, as in v3). The param-matched graph (RUN_3) lives on the **att3ntion** pod; its per-category data was pulled as a compact summary (`scripts/gen_summary.py`) to build the combined charts.

*The compute-side counterpart to [`RUN_3.md`](../v3_prefix_lm/RUN_3.md). RUN_3 held **parameters** fixed; RUN_4 holds **compute (FLOPs)** fixed, and reads the two together as the two faces of one comparison. No hg re-run — the slow `hg_gather` 10-seed sweep from v3 is reused verbatim.*

## TL;DR
- **The question:** does graph attention match hypergraph when given **equal compute** (not equal size)? RUN_3 answered the equal-size version; this answers the equal-compute version.
- **You cannot do both at once.** The two mechanisms have different scaling laws — graph attention is O(N²·d), hg is O(N³·d) — so a config can match params *or* FLOPs, never both. RUN_3 (param-matched) + RUN_4 (FLOP-matched) **bracket** the comparison.
- **Measured FLOP gap** (`scripts/flop_match.py`, real train-set lengths, bucketed/padded): `hg_gather` 3L = **252 TF/epoch**. The param-matched graph (4L, d256) is only **0.04×** that. Doubling depth to 6L reaches just **0.06×** — *not* FLOP parity.
- **FLOP parity needs width, not depth:** **6 layers @ d≈1088** (heads 17, FFN 3·d) ≈ 0.97× hg's FLOPs — but ~**24× hg's parameters** (~74M vs 3.15M).
- **Design = a graph compute-scaling ladder** (param-matched → 6L/d256 → FLOP-matched) with hg's single point overlaid. Answers "how much compute does graph need to catch hg, if ever?" not just a single matched point.
- **🏁 Result — the winner flips with the control.** At **equal size** hypergraph beats graph (gen EM **4.8% vs 2.1%**) — the 3-way mechanism helps per-parameter. At **equal compute** the wide graph wins (**16.6% vs 4.8%**) — but with ~24× the parameters. Adding depth alone (manager's 6L/d256) lifts graph to **6.5%**, enough to overtake hg cheaply; width is the bigger lever. Honest summary: **hypergraph is the better mechanism per-parameter; a dense graph is the better way to spend compute.**
- ⚠️ The hg gen EM here (4.8%) is well below the pre-prefix-LM ~15.9% — regime change, flagged for sanity-check (see Caveats).

## What this run answers (and why, vs RUN_3)
RUN_3 fixed **model size** (~3.15M params each) and asked *"best use of a parameter budget?"* RUN_4 fixes **compute** and asks *"best use of a FLOP budget?"* — the budget that actually costs GPU-hours. This is the standard **IsoFLOP** framing (Chinchilla, [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)); the params-vs-FLOPs decoupling is the same one the MoE literature leans on (Switch Transformers [arXiv:2101.03961](https://arxiv.org/abs/2101.03961); "Parameters vs FLOPs" [arXiv:2501.12370](https://arxiv.org/abs/2501.12370)).

Read jointly:
- **hg wins/ties in *both* regimes** → strongest result: hg matches graph whether you equalize size *or* compute, even when FLOP-matching hands graph ~24× the parameters.
- **graph only wins under FLOP-match** → it's a compute story, not a mechanism story.

## The FLOP measurement (settled, reproducible)
`scripts/flop_match.py` tokenizes `train.tsv` exactly as training does (prefix-LM packing, `max_seq_len=512`), then evaluates corrected closed-form forward FLOPs over the **actual** bucketed/padded batch lengths (mult-add = 2 FLOPs). No GPU, no model forward.

Train set: 27,227 ex, **mean N=57.7, median 49, p99=205, max 339, 0 truncated**. Because hg cost ∝ ΣN³ (tail-dominated), the cube-weighted mean length ≈ 100. Bucketed padding overhead = +5.8%.

| arm (per-epoch fwd FLOPs) | layers | d_model | heads | FFN | FLOPs | vs hg | params | vs hg |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| **hg_gather** (target) | 3 | 256 | 4 | 768 | **252 TF** | 1.00× | 3.15M | 1.0× |
| graph_flash (RUN_3) | 4 | 256 | 4 | 736 | 10.2 TF | 0.04× | 3.22M | 1.0× |
| graph 6L/d256 (manager's "2× depth") | 6 | 256 | 4 | 768 | 14.9 TF | **0.06×** | ~4.5M | 1.4× |
| **graph FLOP-matched** | 6 | **1088** | 17 | 3264 | **~244 TF** | **0.97×** | ~74M | **~24×** |

Depth alone would need **~81 graph layers** to reach parity — impractical. Width is the efficient lever (FLOPs ∝ d²). The FLOP-matched arm keeps the manager's **6 layers** but corrects the width (d≈1088, not 256).

### How FLOPs are counted (exact)
Closed form in `scripts/flop_match.py`, **forward pass, one multiply-add = 2 FLOPs**. Per layer, per example, with `d`=d_model, `F`=FFN width, `H`=heads, `N`=sequence length (the in-repo `calcFlops` methods are **not** used — `HypergraphCudaAttentionAdapter.calcFlops` has dimensionally-wrong projection terms):

- **shared per layer:** FFN = `4·N·d·F` (two linears d→F→d) `+ 20·N·d` (two RMSNorms, negligible).
- **graph layer** = `8·N·d²` (Q,K,V,O projections: 4 matmuls × 2·N·d²) `+ 4·N²·d` (QKᵀ and A·V: 2 × 2·N²·d, using H·d_head = d) ` + FFN`.
- **hg layer** = `14·N·d²` (Q,R,S + Vq,Vr,Vs + O = 7 projections) `+ 12·N³·d` (the 3-way score einsum `bhid,bhjd,bhkd→bhijk` ≈ 3·N³·d, plus 3 output einsums ≈ 9·N³·d) `+ 15·H·N³` (3 softmaxes over N³, negligible) ` + FFN`.
- **whole model** = `n_layers × layer + 2·N·d·V` (lm_head, `V`=vocab 893). Embedding lookups are ~0 FLOPs.

The single structural difference is the attention core: graph **O(N²·d)** vs hg **O(N³·d)** — everything else (projections, FFN, lm_head) is O(N·d²) and identical in form.

**Per-epoch total** = the quantity actually matched. Each example's FLOPs depend on `N`, so we **don't** evaluate at a single length: we tokenize `train.tsv` exactly as training does, **simulate the length-bucketed batch sampler** (batch 32, bucket_multiplier 50), and charge every example at its **batch's padded length** (`Σ_batch count × model_flops(N_pad)`), since hg pays the padded-length³ cost. That's why the match is *aggregate over the real length distribution* (+5.8% padding overhead), not pointwise. Backward ≈ 2× forward for both arms (matmul/einsum-dominated), so matching forward matches training compute.

### How params are counted (exact)
`sum(p.numel() for p in model.parameters())` — the literal weight count, verified on-pod by `test_graph_ffn_hidden_param_matches_hypergraph`. Closed form (d=256, H=4, V=893, msl=512):
- **embeddings** = `V·d` (token) `+ msl·d` (positional) `+ d·V` (lm_head, no bias).
- **graph attn / layer** = `4·d² + 2·d` (Wq,Wk,Wv,Wo + 2 biases); **+ 2 RMSNorm** = `2·d`; **+ FFN** = `2·d·F + F + d`.
- **hg attn / layer** is larger (3-way: extra value projections) — this is why, at equal depth, hg has *more attention params*, and RUN_3 widened the graph FFN (`--graph-ffn-hidden 736`) to compensate: graph 4L = **3,152,256** vs hg 3L = **3,151,872** (+0.01%).

**Why you can't do both:** matching params holds `Σ numel` equal; matching FLOPs holds per-epoch FLOPs equal. Because the hg attention core is N³ and graph's is N², equalizing one forces the other apart — the FLOP-matched graph (6L/d1088) lands at ~74M params (**~24× hg**). RUN_3 fixes params (graph stays ~3.2M, far fewer FLOPs); RUN_4 fixes FLOPs (graph balloons in params). Hence the bracket.

## Arms & ladder
RUN_4 introduces **two new graph arms** and **reuses** RUN_3 for the curve's two anchors (no re-run):

| rung | arm | source | role |
|---|---|---|---|
| 1 (low / param-matched) | `graph_flash` 4L d256 | **reuse `runs/v3`** | param-matched anchor (= RUN_3) |
| 2 (intermediate) | `graph_6L256` | **new, `runs/v4`** | manager's depth-doubling point |
| 3 (FLOP-matched) | `graph_flop6L` (6L, d1088) | **new, `runs/v4`** | equal-compute anchor |
| hg reference | `hg_gather` 3L | **reuse `runs/v3`** | the single hg point, overlaid |

Only rungs 2 and 3 are run here → **20 new runs** (2 arms × 10 seeds). The slow hg sweep and the param-matched graph are reused as-is.

## Common regime (matches RUN_3 / v3 exactly)
True prefix-LM, `--max-seq-len 512`, batch 32 × accum 4 (eff 128), `EVAL_DEV_EVERY=100`, `PATIENCE=60`, `MIN_DELTA=1e-4`, `MNT=384`, `--out-dir runs/v4`, `--eval-at-end`. **Epoch cap 50 for both new arms** (= hg's cap, so total training FLOPs ≈ per-epoch × 50 line up; both converge well before — `best.pt` is dev-selected). FFN = 3·d_model on both new arms (no param-match — that's deliberately abandoned here).

## Prerequisites (code changes — do before launching)
1. **`run_sweep.py` — per-arm width.** The `ARMS` dict (lines 37–42) only carries `attn/scatter/layers`; `d_model`/`heads` fall back to train.py defaults (256/4). Extend the spec and `build_command` (lines 104–136) to emit `--d-model`, `--heads`, `--ffn-hidden` when present:
   ```python
   ARMS["graph_6L256"]  = {"attn": "graph_flash", "scatter": False, "layers": 6}
   ARMS["graph_flop6L"] = {"attn": "graph_flash", "scatter": False, "layers": 6,
                           "d_model": 1088, "heads": 17, "ffn_hidden": 3264}
   # in build_command, after --layers:
   for k, flag in (("d_model","--d-model"), ("heads","--heads"), ("ffn_hidden","--ffn-hidden")):
       if k in spec: cmd += [flag, str(spec[k])]
   ```
   (train.py already accepts `--d-model/--heads/--ffn-hidden`, lines 133–135 + the `--ffn-hidden` arg.)
2. **`aggregate.py` — group by arm, not attn.** Lines 34–44 bucket by the `attn_impl` substring, so `graph_6L256` and `graph_flop6L` (both `graph_flash`) would collapse into one average. Change the group key to the **run-dir arm prefix** (the `{arm}` in `{arm}_{attn}_s{seed}`, available as `run_name.rsplit("_" + attn + "_s", 1)[0]` or via the saved `config.json`/`--log-name`). Verify three distinct graph groups + hg appear.
3. **Confirm reusable runs are local:** all 10 `hg_gather_hypergraph_cuda_s{1..10}` and 10 `graph_flash_graph_flash_s{1..10}` dirs in `runs/v3` with `final_metrics.json` + `per_example_gen.json`. Pull any missing graph dirs from the att3ntion pod (no auto-sync — see [RUN_3](../v3_prefix_lm/RUN_3.md) caveats).
4. **Re-pin the matched width if any dim changes:** rerun `scripts/flop_match.py`; keep `graph_flop6L` within ±5% of hg's per-epoch FLOPs.

## Invariants (keep true)
- **Identical regime to v3** (prefix-LM mask, seq 512, eff-batch 128, MNT 384) — the only intended differences from RUN_3's graph are depth/width and the dropped param-match. Any other drift contaminates the comparison.
- **FLOP-match is aggregate, not pointwise.** Because hg ∝ N³ and graph ∝ N², a config matched on the total is slightly over-FLOPped on short sequences and under on the long tail. The match holds on **total per-epoch FLOPs over the real length distribution**, not at every length.
- **FLOP-match = forward FLOPs.** Backward ≈ 2× forward for both arms (matmul/einsum-dominated), so matching forward matches training compute. Report actual per-seed totals (epochs × per-epoch) as a check.
- **Split roles unchanged:** dev = selection + early stop; test = monitor-only; gen = OOD headline EM only.
- **Paired comparison:** gen examples + stable indices are shared across all arms → use paired McNemar / bootstrap on `per_example_gen.json` keyed by idx.

## Results
All 10 seeds per arm completed (failures=0). Generalization (gen) is the OOD split; test is in-distribution. EM = ReCOGS semantic exact-match.

| rung | arm | params | FLOPs vs hg | **gen EM** | in-dist test EM |
|---|---|--:|--:|--:|--:|
| baseline | hypergraph `hg_gather` 3L/d256 | 3.15M | 1.00× | **4.8%** (±4.1) | 95.8% |
| equal size | graph param-matched 4L/d256 | 3.2M | 0.04× | **2.1%** | 94.6% |
| +depth (manager's) | graph `graph_6L256` 6L/d256 | ~4.5M | 0.06× | **6.5%** | 97.2% |
| equal compute | graph `graph_flop6L` 6L/d1088 | ~74M | 0.97× | **16.6%** (±2.8) | 99.6% |

Gen EM is the macro-average over the 21 ReCOGS categories (≈ the micro-average here; hg micro = 4.77±4.08, FLOP-matched graph = 16.62±2.75). hg has high seed variance (gen range 0.5–12.3%); the graph arms are tighter.

**The comparison flips with the control:**
- **Equal size (param-matched):** hypergraph **4.8%** > graph **2.1%** → the 3-way attention mechanism generalizes better *per parameter*.
- **Equal compute (FLOP-matched):** graph **16.6%** > hypergraph **4.8%** → but the graph is spending those FLOPs on ~24× more parameters, not a better mechanism.
- **+depth only (manager's 6L/d256, same width):** **6.5%** — overtakes hypergraph at ~6% of its compute and ~1.4× its params; depth is a cheap win, width is the larger one.
- **In-distribution test is near-ceiling for all arms (94.6–99.6%)** — every arm learns the task; the entire story is OOD generalization.

**Figures** (`docs/`):
- `graph_vs_hg_training_v4.png` — training loss curves (dev/test, mean±std) + final gen EM, FLOP-matched graph vs hg. Graph reaches ~10× lower teacher-forced loss (≈6e-4 vs hg ≈8e-3).
- `graph_vs_hypergraph_gen_test_v4.png` — per-category (v3-style), FLOP-matched graph vs hg.
- `graph_vs_hypergraph_gen_test_v3v4.png` — per-category **bracket**: hypergraph (shared) + param-matched + FLOP-matched graph.
- `graph_vs_hypergraph_gen_test_ladder.png` — per-category **4-arm ladder**: hypergraph + param-matched + +depth + FLOP-matched.

## Analysis
- **Mechanism vs compute, disentangled.** RUN_3+RUN_4 together show the two questions have *opposite* answers: hypergraph is the stronger mechanism at fixed size, while a plain dense graph is the more effective use of a fixed FLOP budget (because the N³ kernel buys parameters expensively). A single comparison would have implied the reverse of the other — which is exactly why both controls were needed.
- **Graph scaling within d256.** 4L→6L at the same width moves graph 2.1%→6.5% (past hg) for almost no extra compute; the jump to 16.6% only comes with width (d256→d1088, ~24× params). So depth is necessary but not sufficient to make graph competitive — the large gains are width/parameter-driven.
- **What this does *not* settle.** The FLOP-matched win comes with a 24× parameter advantage; it is not evidence that graph attention is a better *mechanism*. And see the hg-magnitude caveat below before treating 4.8% as hg's definitive number.

## Caveats
- **hg gen EM (4.8%) is much lower than the pre-prefix-LM ~15.9%** (the unshifted-loss-bug-fixed 4×H100 run). This v3 regime (true prefix-LM) is the first time RUN_3's hg gen numbers were read end-to-end. A drop this large should be sanity-checked — is prefix-LM genuinely hurting hg's OOD generalization, or is there a v3 hg eval/config issue? Resolve before treating 4.8% as hg's definitive number; it anchors every comparison here.
- **~24× parameter imbalance** at the FLOP-matched point is intrinsic, not a flaw — it is the price of equalizing compute across different scaling laws. State it plainly alongside any graph win.
- **No paired significance yet.** Means±std only; McNemar/bootstrap on the shared-index `per_example_gen.json` is still to do (see What's next).
- One `hg_gather` seed's `eval_curve.tsv` had a malformed (non-finite) row; the loss-curve plotter filters it. Final gen EM (from `final_metrics.json` / `per_example_gen.json`) is unaffected.
- Matched on aggregate per-epoch FLOPs over the empirical length distribution, not pointwise (see Invariants).
- The reused 4L anchor used FFN 736 (RUN_3 param-match) vs 768 default; <1% FLOP difference — negligible on the curve.
- att3ntion pod has no auto-sync → any graph dirs run there need a manual pull before aggregation.

## What's next (pipeline)
Done: code changes applied, smoke test, 20-run sweep (failures=0), training/EM + per-category + bracket + ladder figures. Remaining:
1. **Resolve the hg-magnitude question** (top caveat) — spot-check a few hg gen predictions and confirm the prefix-LM eval path; this gates how much weight the comparison carries.
2. **Paired significance:** McNemar + bootstrap CI on `per_example_gen.json` (shared `idx`): `graph_flop6L` vs `hg_gather`, and `graph_6L256` vs `hg_gather`.
3. (Optional) explicit **gen-EM-vs-FLOPs** scaling curve (`plots/`), the four rungs on a log-FLOP x-axis with hg's point overlaid.

## Reproduction
What actually ran: the detached driver `scripts/run_v4_recogs.sh` on the **recogs pod (4×H100)**. It calls `run_sweep.py` directly (with its own 4× crash-retry + `--skip-existing`) rather than `run_overnight.sh` — on the pod `run_overnight.sh` lives at the repo top level and its `cd "$(dirname $0)"` breaks when invoked as `scripts/...`. On completion it auto-runs `plots/plot_v4_graph_vs_hg.py`.
```sh
# on recogs, from /home/dev/workspace/experiments/recogs, detached:
setsid nohup bash scripts/run_v4_recogs.sh >/dev/null 2>&1 </dev/null &
# driver effectively runs:
python -u run_sweep.py --arms graph_flop6L,graph_6L256 \
  --seeds 1,2,3,4,5,6,7,8,9,10 --gpus 0,1,2,3 --skip-existing --out-dir runs/v4 \
  --max-seq-len 512 --batch-size 32 --grad-accum-steps 4 \
  --epochs 50 --early-stop-patience 60 --early-stop-min-delta 1e-4 \
  --eval-dev-every 100 --eval-batch-size 32 --eval-max-new-tokens 384 --prefix-lm
```
Per-category / combined charts (run locally; laptop has matplotlib): `scripts/gen_summary.py <base> '<glob>'` → compact JSON per arm (param-matched graph pulled from att3ntion this way), then `plots/plot_gen_test_v4.py`, `plots/plot_gen_test_v3v4_combined.py`, `plots/plot_gen_test_ladder.py`. Progress: `runs/v4/<run>/eval_curve.tsv` + `sweep_logs/v4_driver.log`.
