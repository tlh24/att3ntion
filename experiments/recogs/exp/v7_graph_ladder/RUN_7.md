# ReCOGS: Graph width-ladder BASELINE — Run 7 (v7)

**Date:** 2026-07-03 (launched ~19:30 PT, overnight) · **Setup:** graph_flash twins of the
RUN_5 hypergraph ladder — identical shapes and identical regime, only the attention mechanism
differs. 60 runs (6 configs × 10 seeds) across 3 pods / 16 GPUs. · **Status:** ✅ COMPLETE —
all 60 runs finished (failures=0; drivers rc=0 2026-07-04/05); results pulled + analyzed below.

## Synthesis — what the RUN_5→RUN_7 program established (2026-07-07)

The five findings worth carrying forward (evidence: ~130 runs, 10–18 seeds/cell, identical
prefix-LM regime throughout; figures `v7_graph_vs_hg_scaling.png`,
`v7_graph_vs_hg_per_category{_d768,_parammatch}.png`, `v7_depth_at_d256.png`,
`v5_*` family):

1. **Hypergraph's OOD advantage is mechanism-level.** It survives shape-matching (+5.7 pts
   at d768), param-matching (+4.0 pts at 24.7M: hg 12.41 vs graph_d870 8.38), and is invisible
   in-distribution (both ~99% test). hg converts params into OOD gen at **~2.7× graph's
   slope**. Standing caveat: per-FLOP graph leads until its slope flattens — hg's advantage
   costs O(N³·d) compute.
2. **Neither mechanism does recursion, ever.** CP/PP recursion ≤1.5% across 26× params,
   3L/6L, both attentions. Structural generalization is not more-scale-away; it needs a
   different kind of solution (data/curriculum/architecture). Don't spend compute here
   expecting scale to fix it.
3. **The mechanisms generalize *differently*, not just unequally.** Shape: depth helps graph
   (+2.9 at d256) and hurts hg through d384 (−0.6, −1.8), ~neutral at d512 (+0.9, n=2).
   Categories: hg owns one-shot **common-noun** role assignment + argument alternations;
   **graph is reliably better at proper nouns** (13.3 vs 2.6 at param parity, growing with
   size). Likely two different binding strategies — the most interesting mechanistic thread.
4. **Methodology: small-n gen EM misleads; in-dist can't rescue it.** Every rung's mean moved
   as n grew; seed CV shrinks with width (1.21 → 0.21); r(test, gen) flips negative at the
   top of the ladder → no model selection for OOD from dev/test. Quote ranges for n≤4.
5. **The matching axis changes the conclusion.** Shape-matched read "graph plateaus";
   param-matched corrected it to "shallower slope." FLOP-, param-, and shape-matching are
   three different experiments — say which one a claim comes from.

Nulls for the record: FFN pre-ReLU L2 at d256 = no OOD effect (RUN_6, +0.29, p=0.46).
Open threads: d1024 hg rung (does the 2.7× slope hold?); proper-vs-common asymmetry probe;
6L384/6L512 seeds 3–10; graph_6L512 (shape convergence); recursion interventions.

## Full config capture (all identical-regime runs, RUN_3→RUN_7 program)

Regime for every row unless noted: ReCOGS (~27k train, fixed), true prefix-LM, seq 512,
batch 32×4 (eff 128), 50 epochs, early-stop OFF, MNT 384, head_dim 64, FFN=3·d, 1 GPU/run.
"Width" = d_model with heads=d/64 (head_dim fixed). gen = OOD EM over the 21k gen split.

**A. 3L width ladder — hg vs graph twins side-by-side** (read down Δ: hg's lead grows with width)

| d (3L) | hg gen | graph gen | **Δ gen** | hg test | graph test | hg/graph params | n (hg/gr) |
|--:|--:|--:|--:|--:|--:|---|---|
| 128 | 0.71 | 0.00 | **+0.71** | 83.9 | 70.0 | 0.94M / 0.79M | 10/10 |
| 256 | 3.87 | 3.55 | **+0.32** | 95.7 | 95.6 | 3.15M / 2.56M | 18/10 |
| 384 | 6.77 | 5.76 | **+1.01** | 97.7 | 97.3 | 6.65M / 5.32M | 10/10 |
| 512 | 8.30 | 6.55 | **+1.75** | 98.5 | 98.2 | 11.4M / 9.05M | 12/10 |
| 768 | **12.41** | 6.75 | **+5.67** | 99.0 | 98.7 | 24.8M / 19.5M | 10/10 |

gen ranges (seed min–max): hg 128 [0.0–2.4] · 256 [0.5–12.3] · 384 [1.9–14.8] · 512 [3.6–12.4]
· 768 [9.9–16.7]; graph 128 [0.0–0.03] · 256 [1.0–6.5] · 384 [1.5–10.6] · 512 [2.3–10.2] ·
768 [3.4–11.6]. hg_d256 = v3 s1–10 + v5 s11–18 (v3 early-stop armed, never triggered);
hg_d768 s3–10 scored offline via the identical eval path.

**B. Depth: 6L vs 3L per mechanism** (same move, opposite signs at d256; hg 6L always wins
in-dist while losing OOD through d384 — the memorization dissociation)

| d | hg 3L → 6L gen | **Δ hg** | graph 3L → 6L gen | **Δ graph** | n (hg 6L) | hg test 6L vs 3L |
|--:|---|--:|---|--:|--:|---|
| 128 | 0.71 → 0.63 | −0.08 | — | — | 8 | 87.3 > 83.9 |
| 256 | 3.87 → 3.25 | **−0.62** | 3.55 → 6.46 | **+2.91** | 10 | 97.1 > 95.7 |
| 384 | 6.77 → 4.96 | **−1.81** | — | — | 2 | 98.3 > 97.7 |
| 512 | 8.30 → 9.24 | **+0.94** ⚠ | — | — | 2 | 99.2 > 98.5 |

6L gen ranges: hg 128 [0.1–2.5] · 256 [1.3–6.6] · 384 [4.3–5.6] · 512 [5.0–13.5];
graph 6L256 [2.3–11.5]. 6L params: hg 1.58/5.71/12.42/21.63M; graph_6L256 4.53M.
⚠ d512 reversal is n=2 with a 5.0–13.5 spread — hint, not claim.

**C. Special comparisons**

| comparison | A | B | verdict |
|---|--:|--:|---|
| **Param-match @ ~24.7M** | hg_d768 **12.41** (n=10) | graph_d870 (3L/h15/hd58) **8.38** [5.5–12.7] (n=10) | hg **+4.03** at equal params |
| FFN-L2 @ d256 (RUN_6) | +L2 4.16 [1.9–7.6] (n=8) | baseline 3.87 (n=18) | null (p=0.46) |
| Legacy v3 graph (4L/d256, 40 ep, early-stop ON, ffn 736) | gen 2.15, test 94.6 (n=10) | — | pre-ladder regime; not comparable |

Not in this table (different variants/regimes, see their docs): hg_full scatter runs (RUN_hgfull),
v4 graph_flop6L (FLOP-matched 6L/d1088; data on pods only), pre-prefix-LM pilots (RUN_1–2).
Total in-regime: **169 runs** across 18 configs.

**A pattern worth staring at (visible in the table): for hg, 6L beats 3L in-distribution at
EVERY width (87.3>83.9, 97.1>95.7, 98.3>97.7, 99.2>98.5) while OOD gen is lower through d384.**
Depth buys hg in-dist fit but not generalization — a memorization/shortcut signature, not a
capacity failure. Graph shows no such dissociation (its 6L helps both).

## Results (2026-07-04) — hypergraph keeps scaling, graph plateaus

gen = OOD EM mean [min–max] over seeds; identical shapes per rung. Figure:
`docs/v7_graph_vs_hg_scaling.png` (via `plots/plot_v7_graph_vs_hg_scaling.py`).

| rung | hg gen | graph gen | Δ (hg−graph) | hg test | graph test |
|---|--:|--:|--:|--:|--:|
| d128 | 0.71 (n=10) | **0.00** [0.00–0.03] (n=10) | +0.71 | 83.9 | 70.0 |
| d256 | 3.87 (n=18) | 3.55 [1.04–6.50] (n=10) | +0.32 | 95.7 | 95.6 |
| d384 | 6.77 (n=10) | 5.76 [1.50–10.61] (n=10) | +1.01 | 97.7 | 97.3 |
| d512 | 8.30 (n=12) | 6.55 [2.34–10.23] (n=10) | +1.75 | 98.5 | 98.2 |
| d768 | **12.41** (n=10) | **6.75** [3.42–11.63] (n=10) | **+5.67** | 99.0 | 98.7 |
| 6L256 (depth ctrl) | 2.39 (n=2) | **6.46** [2.34–11.50] (n=10) | −4.07 | 97.1 | 97.2 |

**Param-match extension (2026-07-06/07, `graph_d870`):** graph attn is ~4d²/layer vs hg's
~7d², so same shape ⇒ hg carries ~1.27× params (graph_d768 = 19.48M vs hg_d768 = 24.79M).
`graph_d870` (3L, h15, ffn 2610, **24.73M = hg_d768 −0.3%**, 10 seeds):
**gen 8.38%** med 7.99 [5.52–12.74], test 98.9%.

**Findings:**
1. **Hypergraph scales ~2.7× steeper in params; param parity does NOT close the gap.**
   At true param counts the curves overlap up to ~6M, then diverge: graph 6.55 (9.1M) →
   8.38 (24.7M) vs hg 8.30 (11.4M) → 12.41 (24.8M). At matched 24.7M params:
   **hg 12.41 vs graph 8.38 (+4.0 pts, 1.48×)**. The earlier "graph plateaus" read
   (6.75 at d768) was partly a param-axis artifact — graph resumes climbing with more
   params, but on a much shallower slope; the remaining gap at equal params is mechanism.
2. **The width≫depth law is hg-specific.** For graph, 6L/d256 (6.46) ≈ 3L/d512 (6.55) —
   depth and width are interchangeable. For hg, depth was catastrophic (2.39 vs 8.30).

## Depth-at-width extension (hg 6L ladder, 2026-07-07)

6L twins of the hg width rungs (identical regime, runs/v5). gen EM, Δ vs the 3L twin:

| width | 3L | 6L | Δ(6L−3L) |
|---|--:|--:|--:|
| d128 | 0.71 (n=10) | 0.63 [0.1–2.5] (n=8) | −0.08 |
| d256 | 3.87 (n=18) | 3.25 [1.3–6.6] (n=10) | −0.62 |
| d384 | 6.77 (n=10) | 4.96 [4.3–5.6] (n=2) | −1.81 |
| d512 | 8.30 (n=12) | **9.24** [5.0–13.5] (n=2) | **+0.94** |

- **hg's depth aversion is not universal**: no effect at the d128 floor, hurts at d256
  (n=10, −0.62; milder than the old n=2 estimate −1.5) and d384, and **disappears — possibly
  reverses — at d512** (n=2, wide range: a hint, not a claim).
- Contrast graph at d256: **+2.91** from depth. The mechanisms still want different shapes
  in the small/mid regime; at d512 they may converge.
- **Width remains the better per-param spend for hg everywhere tested**: 6L/d512 (~21.6M
  params) = 9.24 still loses to 3L/d768 (24.8M) = 12.41.
- Follow-ups if pursued: seeds 3–10 for 6L384/6L512 (both n=2), and graph_6L at d512 to
  test shape-convergence on the graph side.
3. **Tiny scale: graph fails outright.** graph_d128 = 0.00% OOD and only 70% in-dist
   (hg_d128: 0.71% / 83.9%) — hg is substantially more parameter-efficient at the bottom too.
4. **In-distribution is a wash from d256 up** (Δtest ≤ 0.4 pts) — the entire architectural
   difference lives in OOD generalization, invisible to test EM.
5. **Caveat:** at identical shapes graph uses far fewer FLOPs (O(N²·d) vs O(N³·d)); per-FLOP
   graph wins until its plateau — but the plateau means extra graph compute stops buying OOD,
   while hg's curve is still rising. Report both framings.

> **Purpose:** RUN_5 gave the hg scaling curve; v3's graph arm is NOT a usable baseline for it
> (4L, ffn 736 param-matched, 40 epochs, early-stop ON). RUN_7 reruns graph at every hg rung
> so tomorrow's graph-vs-hg comparison is apples-to-apples at each size.

## Arms (all attn=graph_flash, prefix-LM, FFN=3·d, heads=d/64 on 3L rungs)

| arm | layers | d_model | heads | ffn | twin of |
|---|--:|--:|--:|--:|---|
| graph_d128 | 3 | 128 | 2 | 384 | hg_d128 |
| graph_d256 | 3 | 256 | 4 | 768 | hg_gather (d256) |
| graph_d384 | 3 | 384 | 6 | 1152 | hg_d384 |
| graph_d512 | 3 | 512 | 8 | 1536 | hg_d512 |
| graph_d768 | 3 | 768 | 12 | 2304 | hg_d768 |
| graph_6L256 | 6 | 256 | 4 | 768 | hg_6L256 (depth ctrl) |

Regime identical to v5: 50 epochs, `--early-stop-patience 0`, seq 512, batch 32×4 (eff 128),
`--eval-dev-every 100`, MNT 384, inline `--eval-at-end` everywhere (graph gen eval is fast,
~2 min — no offline-gen dance needed, unlike hg d768).

## Launch (2026-07-03, driver `scripts/run_v7.sh`, out-dir `runs/v7`)

| pod | ns | GPUs | ARMS | seeds |
|---|---|--:|---|---|
| recogs-8x | obelisk | 8 | graph_d768,graph_d512 | 1–10 |
| recogs | strange-loop | 4 | graph_d384,graph_d256 | 1–10 |
| recogs-scale | strange-loop | 4 | graph_6L256,graph_d128 | 1–10 |

Rough per-seed estimates (3L, 50 ep, from v3 graph 4L/d256 = 242 s/ep, overhead-dominated so
width scaling is flatter than FLOPs): d128 ~1.5h, d256 ~2.5h, d384 ~3.5h, d512 ~4.5h,
d768 ~7h, 6L256 ~4.5h → ~230 GPU-h total, ~15h/pod wall. Expect completion mid-day 2026-07-04.

Progress: `sweep_logs/v7_*_driver.log` + `sweep_logs/graph_*_s*.log` + `runs/v7/*/eval_curve.tsv`.
Drivers are `--skip-existing` + 4× crash-retry — safe to re-run the same invocation to resume.

## Ops notes
- **train.py config.json bug fixed pre-launch:** a v6 edit had nested the config.json write
  inside `if args.ffn_l2_reg:` — every non-L2 run would have been missing config.json. Fixed
  (dedented to `if is_main:`) and pushed to all 3 pods before launch; smoke-verified.
- Code pushed via `kubectl cp` + sha256 verify (sync unreliable — see PLAYBOOK §1).
- Smoke test: graph_d128 s99, 1 epoch on recogs — correct config (graph_flash/3L/d128/h2/
  ffn384/prefix_lm), full eval pipeline, then deleted.
- runs/ is `.actlignore`'d → pull runs/v7 manually tomorrow (tar-over-kubectl-exec,
  exclude checkpoints; the small files are all the comparison needs).

## Tomorrow (analysis)
1. Pull `runs/v7` small files from all 3 pods.
2. Extend `plots/plot_gen_test_v5_ladder.py` / `plot_v5_scaling.py` with the graph arms —
   graph-vs-hg overlay per rung (params axis identical, FLOP axis very different).
3. Key questions: does graph scale OOD with width like hg? Same lexical-vs-structural split?
   Where does hg's advantage live (which categories), and does it grow or shrink with size?
4. Caveat for FLOP framing: at equal shape, graph uses ~6–16× fewer FLOPs — report both
   per-params and per-FLOP comparisons.
