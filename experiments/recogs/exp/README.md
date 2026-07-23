# Experiments index

One folder per experiment. Each bundles everything that belongs to that run:

```
exp/vN_title/
├── RUN_N.md      # the write-up: question, design, results, caveats
├── run_*.sh      # driver script(s) — run from experiments/recogs/, e.g.
│                 #   bash exp/vN_title/run_vN.sh
├── plots/        # plot scripts (read run dirs across exp/, write to ../figures/)
├── figures/      # rendered PNGs
└── runs/         # raw artifacts (GITIGNORED): checkpoints, final_metrics.json,
                  # eval_curve.tsv, losslog.txt, driver_logs/
```

Shared pipeline code (`train.py`, `model.py`, `data.py`, `evaluate.py`,
`metrics.py`, `run_sweep.py`, `plots_common.py`) lives one level up and is the
same for every experiment; each run's `config.json` records how it was invoked.
Cross-experiment tools live in `../analysis/`.

| folder | write-up | question |
|---|---|---|
| `v1_pilot` | [RUN_1](v1_pilot/RUN_1.md) | pilot: does hypergraph beat graph on ReCOGS gen? (1 seed, DDP) |
| `v2_three_arm` | [RUN_2](v2_three_arm/RUN_2.md) | 3-arm ladder ×3 seeds @512; scatter arm diverges |
| `v3_prefix_lm` | [RUN_3](v3_prefix_lm/RUN_3.md) | true prefix-LM + param-match, 10 seeds ×2 arms |
| `v4_flop_match` | [RUN_4](v4_flop_match/RUN_4.md) | FLOP-matched graph scaling (IsoFLOP bracket of RUN_3) |
| `v5_hg_scaling` | [RUN_5](v5_hg_scaling/RUN_5.md) | hypergraph width ladder d128–d768 (Chinchilla-style) |
| `v6_ffn_l2` | [RUN_6](v6_ffn_l2/RUN_6.md) | FFN pre-ReLU L2 regularizer at d256 (no effect) |
| `v7_graph_ladder` | [RUN_7](v7_graph_ladder/RUN_7.md) | graph twins of the v5 ladder + d870 param-match; §Synthesis is canonical |
| `v8_300ep` | [RUN_8](v8_300ep/RUN_8.md) | 300-epoch d768 A/B vs published SOTA regime |
| `_legacy_logs` | — | pre-reorg flat `losslogs/` + `sweep_logs/` (superseded: new runs log inside their run dir) |

## Starting a new experiment (v9+)

1. `mkdir -p exp/v9_title/{plots,figures}` and write `exp/v9_title/RUN_9.md`
   **before launching** — goal, arms, seeds, pods.
2. Copy the nearest `run_vN.sh` as `exp/v9_title/run_v9.sh`; point
   `--out-dir` at `exp/v9_title/runs`. Loss logs and driver logs land inside
   the run dirs automatically.
3. Every figure gets a committed script in `exp/v9_title/plots/` that writes
   to `../figures/`.
4. Anything created on a pod (docs, scripts, edits) gets pulled back to the
   laptop the same day.
