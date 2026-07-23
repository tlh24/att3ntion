#!/usr/bin/env python3
"""Train + test loss curves for the ReCOGS v3 sweep (both arms, all seeds).

Reads every exp/v3_prefix_lm/runs/<run>/eval_curve.tsv (cols step/train/dev/test), plots the
TRAIN and TEST teacher-forced loss only (dev omitted for clarity). Per-seed lines
are faint; the per-arm mean (over seeds, aligned by step) is bold. Log-y.
"""
import glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))  # experiments/recogs
RUNS_DIR = os.environ.get("RUNS_DIR", "exp/v3_prefix_lm/runs")
COLORS = {"hypergraph": "#d62728", "graph_flash": "#1f77b4"}


def arm_of(name):
    return "hypergraph" if "hg_gather" in name else "graph_flash" if "graph_flash" in name else None


def read_rows(f):
    """[(step, train, test)] from eval_curve.tsv, NUL-safe, skip malformed."""
    out = []
    for ln in open(f, errors="replace").read().replace("\x00", "").splitlines():
        if not ln or ln.startswith("step"):
            continue
        p = ln.split("\t")
        if len(p) != 4:
            continue
        try:
            out.append((int(p[0]), float(p[1]), float(p[3])))
        except ValueError:
            continue
    return out


# arm -> {"train": [(steps, vals)], "test": [...]}
data = {a: {"train": [], "test": []} for a in COLORS}
for f in sorted(glob.glob(os.path.join(ROOT, RUNS_DIR, "*", "eval_curve.tsv"))):
    arm = arm_of(os.path.basename(os.path.dirname(f)))
    rows = read_rows(f)
    if not arm or not rows:
        continue
    steps = np.array([r[0] for r in rows])
    data[arm]["train"].append((steps, np.array([r[1] for r in rows])))
    data[arm]["test"].append((steps, np.array([r[2] for r in rows])))


def mean_by_step(seeds):
    """Mean across seeds at each step value where >=1 seed has data."""
    acc = {}
    for steps, vals in seeds:
        for s, v in zip(steps, vals):
            acc.setdefault(int(s), []).append(v)
    xs = sorted(acc)
    return np.array(xs), np.array([np.mean(acc[s]) for s in xs])


fig, ax = plt.subplots(figsize=(11, 6.5))
styles = {"train": dict(ls=":", label_suffix="train"), "test": dict(ls="-", label_suffix="test")}
n_seeds = {}
for arm, c in COLORS.items():
    n_seeds[arm] = len(data[arm]["test"])
    for metric in ("train", "test"):
        for steps, vals in data[arm][metric]:
            ax.plot(steps, vals, styles[metric]["ls"], color=c, alpha=0.18, lw=0.9, zorder=2)
        if data[arm][metric]:
            mx, mv = mean_by_step(data[arm][metric])
            ax.plot(mx, mv, styles[metric]["ls"], color=c, lw=2.4, zorder=4,
                    label=f"{arm} {styles[metric]['label_suffix']} (mean)")

ax.set_yscale("log")
ax.set_xlabel("optimizer step")
ax.set_ylabel("teacher-forced loss")
ax.set_title("ReCOGS v3 train & test loss "
             f"(graph_flash: {n_seeds['graph_flash']} seeds, hypergraph: {n_seeds['hypergraph']} seeds; "
             "faint = per seed, bold = mean)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
out = os.path.join(HERE, "..", "figures", "v3_train_test_loss.png")
fig.savefig(out, dpi=150)
print("wrote", out)
