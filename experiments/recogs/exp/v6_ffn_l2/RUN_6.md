# ReCOGS: FFN Pre-Activation L2 Regularizer — Run 6 (v6)

**Date:** 2026-07-01 · **Setup:** take the RUN_5 d256 anchor (identical regime to `hg_gather`)
and add Tim Hanson's **FFN pre-activation L2 regularizer** with a *variable* lambda; run 8 seeds
on the idle **recogs-scale** pod (4×H100) overnight and compare OOD generalization against the
18-seed d256 baseline. · **Status:** ✅ COMPLETE — 8/8 seeds finished 2026-07-02 11:54 UTC
(failures=0, ~14h). **Headline: no improvement in OOD gen at d256** (treatment 4.16% vs baseline
3.87%, p=0.46; in-distribution unchanged). Only hint: ~halved seed variance, but not significant
(Levene p=0.23). See §Results.

> **Interference guard:** the `recogs` pod (g264) is running RUN_5b Phase 4 (d768 s3–10) and must
> not be disturbed. This run is on the **separate** `recogs-scale` pod (g256). Autosync was off
> (no live sync session), so code edits only reached `recogs-scale` (sync brought up for it alone);
> the `recogs` pod's on-disk code stays frozen and its d768 s7–10 launch from that frozen snapshot.
> As defense-in-depth every code change is gated behind `--ffn-l2-reg` (default OFF).

## The technique (from Tim Hanson)

Add an **L2 penalty on the FFN activations *before* the ReLU** (the output of the first FFN
Linear, i.e. the pre-nonlinearity "pre-activations"), summed/averaged across layers, added to the
cross-entropy loss. Claimed to **preserve representation rank during training** and improve the
representation-learning quality of the hypergraph (HG) attention arm.

**Variable lambda (the key detail):** lambda is not fixed. Track EMAs of the CE loss and the raw
L2 term; set `lambda = ratio · EMA(CE) / EMA(L2)` so the penalty's contribution stays pinned at
`ratio` of the CE magnitude. Tim's suggestion: **ratio = 1/20 = 0.05** (L2 term ≈ 5% of CE).

**NOT weight decay** — this penalizes *activations* (the values flowing through the net this
batch), specifically pre-nonlinearity, not the weights.

## Implementation (flag-gated, default OFF)

- **`model.py`** — `forward(..., collect_ffn_preact=False)`. When True, runs the FFN sublayers
  explicitly (`ffn[0]` Linear → capture pre-ReLU tensor → `ffn[1]` ReLU → `ffn[2]` Linear) and
  stashes per-layer pre-activations in `self._ffn_preacts`. Parameter names are unchanged (still
  `ffn.0/1/2`), so existing d768/baseline checkpoints still load with this code.
- **`train.py`** — new flags:
  - `--ffn-l2-reg` (store_true, default off)
  - `--ffn-l2-ratio` (default **0.05** = 1/20)
  - `--ffn-l2-ema-decay` (default **0.99**)
  Per micro-batch: `l2_raw = mean_over_layers( mean(pre_act²) )` (computed in fp32 under bf16
  autocast); update EMAs of CE and L2; `lambda = ratio·EMA(CE)/EMA(L2)`; `loss = CE + lambda·l2_raw`.
  Logged train loss stays **CE-only** (comparable to baseline). Lambda diagnostics written every
  100 steps to `ffn_l2_curve.tsv` (`step  ema_ce  ema_l2  lambda  l2_raw`).
- **`run_sweep.py`** — new arm `hg_d256_l2` (= `hg_gather` d256 spec + `"ffn_l2": True` → appends
  `--ffn-l2-reg`). Self-contained: `ARMS=hg_d256_l2` needs no extra flags.

## Config & comparison

| | treatment (this run) | baseline (RUN_5 §Results) |
|---|---|---|
| arm | `hg_d256_l2` | `hg_gather` d256 |
| attn / layers / d_model / heads / FFN | hypergraph_cuda(gather) / 3 / 256 / 4 / 768 | identical |
| regime | prefix-LM, seq 512, batch 32×accum4 (eff 128), 50 epochs, early-stop 0, MNT 384, inline `--eval-at-end` | identical |
| **only difference** | **+ FFN pre-act L2 (ratio 0.05)** | — |
| seeds | 8 (s1–s8) → `runs/v6/hg_d256_l2_hypergraph_cuda_s{1..8}` | 18 (v3 s1–10 + v5 s11–18) |
| **baseline gen EM** | (to fill) | **3.87%** [0.48–12.26], test 95.7% |

d256 chosen for the strongest statistical A/B: cheapest rung (~6.1h/seed → 8 seeds in ~12h on 4
GPUs) and the best-characterized baseline (18 seeds).

## GPU map (recogs-scale, 4×H100)

| pod | GPUs | invocation | runs | ~makespan |
|---|--:|---|--:|--:|
| **recogs-scale** | 4 | `ARMS=hg_d256_l2 SEEDS=1,2,3,4,5,6,7,8 GPUS=0,1,2,3` | 8× d256+L2 | ~12h (two waves) |

## Reproduction

```sh
# recogs-scale (4 GPU), overnight — dedicated scripts/run_v6.sh driver (writes runs/v6):
cd /home/dev/workspace/experiments/recogs && \
ARMS=hg_d256_l2 SEEDS=1,2,3,4,5,6,7,8 GPUS=0,1,2,3 \
  setsid nohup bash scripts/run_v6.sh >/dev/null 2>&1 </dev/null & echo pid $!
```
Progress: `runs/v6/hg_d256_l2_hypergraph_cuda_s*/eval_curve.tsv` (train/dev/test loss) +
`ffn_l2_curve.tsv` (lambda over training) + `sweep_logs/*`. Final OOD gen EM in each run's
`final_metrics.json` (inline `--eval-at-end`).

## Monitoring & pulling results (runs/ is NOT auto-synced)

A `.actlignore` now excludes `myenv/` and `experiments/recogs/runs/` from sync (they were
clogging the upstream push). So `runs/v6` lives only on the pod — pull it manually when done:

```sh
# progress (any time):
actl pod exec -n strange-loop recogs-scale -- bash -c \
  'cd /home/dev/workspace/experiments/recogs && tail -2 sweep_logs/hg_d256_l2_s*.log'
# pull results once the driver logs "all runs complete":
actl pod exec -n strange-loop recogs-scale -- bash -c \
  'cd /home/dev/workspace/experiments/recogs && tar cz runs/v6' > /tmp/v6.tgz && \
  tar xzf /tmp/v6.tgz -C experiments/recogs/
```

## Aggregation (post-hoc, local)

- `scripts/gen_summary.py runs/v6 'hg_d256_l2_hypergraph_cuda_s*'` for the treatment.
- Baseline: glob both `runs/v3/hg_gather_hypergraph_cuda_s*` and `runs/v5/hg_gather_hypergraph_cuda_s*`.
- Report median + [min–max] (gen EM is high-variance); paired-ish comparison of the two arms.
- **Sanity check** `ffn_l2_curve.tsv`: after warmup, `lambda·l2_raw / ce ≈ 0.05` (the ratio holds).

## Results

gen = OOD exact-match (headline), test = in-distribution. Per-seed gen EM below.

| arm | seeds | gen EM: mean / median [min–max] | sd | test EM |
|---|--:|--:|--:|--:|
| `hg_gather` d256 (baseline) | 18 | 3.87% / 3.29% [0.48–12.26] | 3.45 | 95.7% |
| `hg_d256_l2` (ratio 0.05) | 8 | **4.16% / 3.83% [1.90–7.61]** | 1.78 | 95.6% |

Per-seed treatment gen EM: s1 6.07, s2 1.90, s3 3.39, s4 7.61, s5 3.10, s6 4.43, s7 2.47, s8 4.27.

**Significance (treatment vs baseline):**
- gen EM difference **+0.29 pt** — **not significant**: Mann-Whitney U p=0.46, Welch t p=0.79.
- test (in-distribution) EM essentially identical (95.6% vs 95.7%).
- variance: treatment sd 1.78 vs baseline 3.45 (≈halved; both tails compressed — no seed below
  1.9% or above 7.6%). But **not significant** given n: Levene p=0.23, Bartlett p=0.084 (borderline).

**Findings:**
1. **No mean OOD-gen improvement at d256.** The FFN pre-activation L2 regularizer (variable λ,
   1/20 of CE) does not move the headline metric at this scale (4.16% vs 3.87%, p=0.46).
2. **In-distribution unaffected** — the penalty doesn't hurt fitting (test 95.6% ≈ 95.7%).
3. **Possible variance reduction** (sd halved, both tails clipped) — directionally consistent with
   Tim's "stabilizes / preserves rank" claim, but n=8 vs 18 makes it suggestive, not conclusive.
4. The regularizer worked as specified throughout (λ·l2/CE held ≈0.045–0.05 across all seeds).

**Caveats / next steps:**
- **Scale:** Tim's claim was for "task 4" and may surface at larger width, where representation
  collapse/rank is more of a bottleneck. RUN_5 showed hg gen scales strongly with width
  (d256 3.9% → d768 10.9%); the regularizer might help more at d512/d768. A follow-up at d512 is
  the natural next test if the variance-reduction hint is worth chasing.
- **Ratio:** only tested 1/20. A sweep over `--ffn-l2-ratio` (e.g. 0.02 / 0.1 / 0.2) is cheap and
  untried; a stronger penalty might show an effect (or hurt fitting).
- **Variance:** to confirm the variance-reduction hint, run treatment out to ~18 seeds for an
  apples-to-apples n before trusting the sd comparison.
