# ReCOGS: 300-epoch d768 hg-vs-graph A/B — Run 8 (v8)

**Date:** launched 2026-07-07 ~20:10 PT · completed 2026-07-10 (drivers rc=0) · **Setup:**
d768/3L hypergraph_cuda vs graph_flash, **300 epochs** (6× the RUN_3→RUN_7 50-epoch regime),
4 seeds/arm, DDP-4 per run across 3 pods / 16 GPUs. · **Status:** ✅ COMPLETE — all 8 runs
trained the full 300 epochs (train_step 63600). 7/8 have final metrics; hg s2's inline gen eval
crashed (SIGABRT in DDP teardown, training intact) and is being re-scored offline.

## Motivation

Does hg's OOD advantage survive SOTA-length training, or was it a training-length artifact?
Published context: Wu/Manning/Potts ReCOGS_pos trained-transformer baseline = **88.55% ±1.87
semantic EM** on gen (enc-dec 2+2L/d300/~4M, word-level, 300 epochs from scratch). Our 50-ep
decoder-only runs (RUN_5/7) sit at 12.41% (hg) / 6.75% (graph) — a heavily under-trained
regime. This run holds everything fixed except epochs (50 → 300) to see whether the gap closes,
widens, or is a mirage.

Regime = identical to the RUN_5→RUN_7 ladder except epochs: eff batch 128 (per-rank 32 × ws4 ×
accum 1), true prefix-LM, seq 512, early-stop OFF, MNT 384, head_dim 64, FFN=3·d. Out-dir
`runs/v8/{hg,graph}_d768_300ep_<attn>_s{1..4}`. Driver `scripts/run_v8.sh` chained
hg → graph per seed (skip-if-final_metrics).

**Figures:** `docs/v8_gen_summary.png` (per-seed gen EM by arm at 300 ep + the 50→300 ep
param-matched reversal; via `plots/plot_v8_gen_summary.py`) · `docs/v8_per_category.png`
(hg_d768 vs param-matched graph_d870 per-category at 300 ep; via `plots/plot_v8_per_category.py`).

## Results — gen (OOD) exact-match, 21k gen split

gen = OOD string EM over the 21k gen split; test = IID EM. 50-ep column = RUN_7 d768 means
(n=10) for the same shapes/regime.

| seed | pod | hg gen | graph gen | hg test | graph test |
|--:|---|--:|--:|--:|--:|
| s1 | recogs | **25.10** | 13.23 | 99.43 | 98.97 |
| s2 | recogs-scale | **27.68** | 13.11 | 99.47 | 99.17 |
| s3 | recogs-8x | **23.69** | **36.24** ⚠ | 99.53 | 99.63 |
| s4 | recogs-8x | **23.58** | 17.54 | 99.47 | 99.37 |
| **mean** | | **25.01** (n=4) | 20.03 (n=4) | 99.47 | 99.28 |
| _50-ep ref_ | | _12.41_ | _6.75_ | _99.0_ | _98.7_ |

hg s2 gen re-scored offline (single-GPU, 21k gen in 10884s ≈ 3.0h) after its inline DDP eval
crashed; dev/test from that pass 99.60/99.67. hg arm now complete at n=4, tight (23.6–27.7).
⚠ graph_d768 s3 (36.24) is a high outlier; graph_d768 excl. s3 = **14.63** (n=3), median 15.4.
NOTE graph_d768 is shape- not param-matched (19.5M vs hg 24.8M) — see the param-match arm below,
which changes the conclusion.

## Findings

1. **hg beats graph at 300 ep ONLY when graph is under-parametrized (shape-matched).** Both
   arms ~double from 50 ep: hg 12.41 → **25.01** (n=4); graph_d768 6.75 → 20.03 (n=4, inflated
   by the s3=36 outlier; ~14.6 without it). So vs the shape-matched graph_d768, hg leads. BUT
   graph_d768 has 21% fewer params — and once graph is given equal params (graph_d870, 24.7M),
   it reaches **28.78** (n=8) and the hg lead *reverses* (see §Param-match arm). The "advantage
   survives long training" story is a param-axis artifact; at equal capacity it does not.
2. **hg is far more seed-stable than graph at 300 ep.** hg spans 23.6–25.1 (CV ~0.04); graph
   spans 13.1–36.2 (CV ~0.5), driven by the s3 outlier. Long training amplifies graph's
   run-to-run OOD variance rather than damping it — consistent with RUN_7 finding #4 (quote
   ranges, not point estimates, for graph).
3. **IID is fully saturated for both (test ≈ 99.3–99.5%).** The entire hg/graph gap is OOD
   generalization; neither arm has any headroom left in-distribution. So the ~24% ceiling is
   not under-fitting — it's a generalization / eval-regime limit.
4. **Still ~64 pts below the 88.55% published baseline even with perfect IID fit.** The gap is
   architecture + eval regime, not epochs: this is decoder-only prefix-LM with strict string
   EM, vs the published enc-dec / semantic-EM setup. 300 epochs does not walk a decoder-only
   prefix-LM toward the enc-dec baseline.
5. **Recursion stays dead; but the RUN_7 clean common/proper dissociation does NOT survive
   param-matching at 300 ep.** cp/pp recursion ≤~6% every seed, both arms (finding #2 holds).
   Per-category (`docs/v8_per_category.png`, hg_d768 vs param-matched graph_d870), the split is
   **mixed, not clean**: graph wins some common-noun *and* some proper-noun categories
   (e.g. proper isolated→subject 58 vs 11) while hg wins others (proper isolated→object 40 vs 17,
   object-omitted→transitive, both datives). The "hg=common / graph=proper" story was a
   shape-/param-axis artifact of the 50-ep regime — at equal params and long training the two
   mechanisms still generalize to *different* categories, but not along the noun-type axis.

## Ops notes

- **hg s2 gen eval crash.** `sweep_logs/v8_hg_d768_300ep_s2.log`: rank 1 SIGABRT (signal 6)
  during the DDP-sharded inline gen eval; dev/test scored fine (both ~99%), `best.pt` +
  `per_example_{dev,test}.json` intact. Driver still marked complete and chained graph s2
  (which finished clean). Re-scoring gen offline via `scripts/offline_gen.py` (single-GPU,
  world_size 1) on recogs-scale → `sweep_logs/v8_offline_gen_hg_s2.log`. Single-GPU over 21k
  gen examples is slow for hg (~2.5–3h; cf. DDP-4 gen_sec≈2934s during training).
- **Cost asymmetry.** hg mean_epoch ≈575–610s vs graph ≈7s (DDP-4, seq 512) — ~80× per epoch;
  hg train ≈48h/seed vs graph ≈35min/seed. hg's O(N³·d) attention dominates wall-clock at
  seq 512, in both training and eval. This is the standing per-FLOP caveat made concrete.

## Param-match arm (v8b, COMPLETE 2026-07-10) — the headline reverses

The v8 A/B above is **shape-matched, not param-matched**: graph attn ≈4d²/layer vs hg ≈7d², so
graph_d768 = **19.5M** vs hg_d768 = **24.8M** (graph −21%). Ran `graph_d870_300ep` (3L, d870,
h15, ffn 2610, d_head 58 = **24.73M ≈ hg_d768 −0.3%**, the RUN_7 param-match shape), byte-identical
v8 regime (DDP-4, eff batch 128, prefix-LM, seq 512, 300 ep, eval-at-end). **8 seeds** on
recogs-8x (2 concurrent DDP-4 chains), driver `scripts/run_v8b.sh`, out
`runs/v8/graph_d870_300ep_*`. Smoke gate passed before launch.

| graph_d870 seed | gen | test | dev |
|--:|--:|--:|--:|
| s1 | 25.72 | 99.60 | 99.70 |
| s2 | 31.15 | 99.43 | 99.53 |
| s3 | 33.55 | 99.60 | 99.80 |
| s4 | 31.07 | 99.63 | 99.60 |
| s5 | 10.94 ⚠low | 99.33 | 99.27 |
| s6 | 43.67 ⚠high | 99.63 | 99.60 |
| s7 | 24.95 | 99.60 | 99.50 |
| s8 | 29.22 | 99.53 | 99.57 |
| **mean** | **28.78** | 99.55 | — |

median 30.15, min 10.94, max 43.67, sd 8.64 (n=8).

**At equal params and 300 epochs, graph OVERTAKES hg — reversing the 50-ep result.**

| regime | hg_d768 (24.8M) | graph_d870 (24.7M) | Δ (hg − graph) |
|---|--:|--:|--:|
| 50 ep (RUN_7) | 12.41 (n=10) | 8.38 (n=10) | **+4.03** (hg ahead) |
| 300 ep (v8/v8b) | 25.01 (n=4) | **28.78** (n=8) | **−3.77** (graph ahead) |

Reading: with 6× the epochs, graph_d870 gains +20.4 (8.38 → 28.78, 3.4×) while hg gains +12.6
(12.41 → 25.01, 2.0×). **hg's param-matched OOD advantage is a short-training phenomenon; graph
closes and passes it given enough training at equal capacity.** RUN_7 finding #1 ("param parity
does NOT close the gap") holds only at 50 ep — it does close, and reverse, by 300 ep.

**Heavy caveats — do NOT over-read the point estimates:**
- **Unequal, small n** (hg n=4 vs graph n=8) and **graph is wildly variable** (10.94–43.67,
  sd 8.64). hg is tight (23.6–27.7, sd ~1.6). The graph mean rides on high seeds (s6 43.7,
  s3 33.6); its median (30.15) is still > hg mean, and even graph's *trimmed* mean beats hg,
  so the reversal is not solely the s6 outlier — but the margin (~3.8) is within plausible
  sampling noise at these n. **Treat as "parity/slight-graph-edge," not a clean win.**
- hg to n=8 costs ~2 days/seed (O(N³d), ~48h/seed) vs graph's ~20 min/seed — firming up hg is
  expensive, which is itself part of the story (graph gets its seeds ~150× cheaper).
- IID saturated for both (test ~99.5%) — entire signal is OOD, as before.
- Same standing caveats: decoder-only prefix-LM, strict string EM ≠ enc-dec/semantic-EM 88.55%.

## Caveats (carry forward)

- Decoder-only prefix-LM, strict string EM — not directly comparable to the enc-dec /
  semantic-EM 88.55% baseline.
- v8 graph_d768 is shape-matched (19.5M), NOT param-matched — use the v8b graph_d870 arm
  (24.7M) for the fair-params comparison against hg_d768.
- graph n=4 with one outlier; hg n=3 until s2 offline gen lands. Treat means as ranges.

Related: RUN_7 (`docs/RUN_7.md` §Synthesis, canonical), RUN_5 (`docs/RUN_5.md`).
