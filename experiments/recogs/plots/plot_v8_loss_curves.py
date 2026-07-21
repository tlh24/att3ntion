#!/usr/bin/env python3
"""RUN_8: overall loss curves over training, hg_d768 vs graph_d870 at 300 ep.

The ONLY time series we currently log is eval_curve.tsv (one row per dev-eval
step): overall, teacher-forced cross-entropy for train / dev / test. These are
aggregate (not per-category) and teacher-forced (not generation). Generation EM
is scored once, at the end (see plot_v8_per_category.py).

Two panels share an x-axis (optimizer step):
  left  : train loss (per-step logged value sampled at each eval step)
  right : dev vs test loss (both in-distribution held-out)
Each arm = mean over seeds, band = ±1 SD. Saves docs/v8_loss_curves.png.
"""
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from recogs_plot_common import COLORS  # noqa: E402

HG, GR = COLORS["hypergraph"], COLORS["graph"]
ARMS = [
    ("hg d768", "runs/v8/hg_d768_300ep_*", HG),
    ("graph d870", "runs/v8/graph_d870_300ep_*", GR),
]
# Which eval_curve.tsv columns to plot in each panel.
PANELS = [("train_loss", "train loss"), ("dev_loss", "dev / test loss")]


def load_curves(run_glob):
    """[(step, {col: values}), ...] per seed; non-finite / ≤0 rows dropped."""
    out = []
    for p in sorted(glob.glob(os.path.join(ROOT, run_glob, "eval_curve.tsv"))):
        try:
            a = np.genfromtxt(p, delimiter="\t", names=True)
            if a.size == 0:
                continue
            step = np.atleast_1d(a["step"])
            cols = {c: np.atleast_1d(a[c]) for c in ("train_loss", "dev_loss", "test_loss")}
            ok = np.isfinite(step)
            for v in cols.values():
                ok &= np.isfinite(v) & (v > 0)
            if ok.sum() >= 2:
                out.append((step[ok], {c: v[ok] for c, v in cols.items()}))
        except Exception:
            continue
    return out


def grid_mean(curves, col):
    """(grid, per_seed_matrix, mean) over seeds on a shared step grid.

    Teacher-forced train/dev loss is spiky on a log axis, so a ±SD band paints
    full-height columns wherever a single seed spikes (m-SD goes ≤0). We instead
    return the per-seed values so callers draw thin per-seed lines + a mean line.
    """
    if not curves:
        return None
    lo = min(float(c[0].min()) for c in curves)
    hi = max(float(c[0].max()) for c in curves)
    grid = np.linspace(lo, hi, 300)
    M = np.vstack([np.interp(grid, c[0], c[1][col]) for c in curves])
    return grid, M, M.mean(0)


fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

for name, run_glob, col_c in ARMS:
    curves = load_curves(run_glob)
    n = len(curves)
    # Panel 1: train loss — per-seed faint lines + bold mean.
    r = grid_mean(curves, "train_loss")
    if r:
        g, M, m = r
        for row in M:
            axes[0].plot(g, row, color=col_c, lw=0.5, alpha=0.20)
        axes[0].plot(g, m, color=col_c, lw=1.8, label=f"{name} (n={n})")
    # Panel 2: dev (solid) + test (dashed) — mean lines + faint per-seed dev.
    for col, ls, tag, show_seeds in [("dev_loss", "-", "dev", True),
                                     ("test_loss", "--", "test", False)]:
        r = grid_mean(curves, col)
        if r:
            g, M, m = r
            if show_seeds:
                for row in M:
                    axes[1].plot(g, row, color=col_c, lw=0.5, alpha=0.18)
            axes[1].plot(g, m, color=col_c, lw=1.8, ls=ls, label=f"{name} {tag}")

for ax, (_, title) in zip(axes, PANELS):
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("cross-entropy loss (teacher-forced)")
    ax.set_yscale("log")
    ax.set_title(title, fontsize=12)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)

fig.suptitle(
    "ReCOGS loss over training @ 300 ep (overall, teacher-forced) — hg d768 vs graph d870",
    fontsize=13,
)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = os.path.join(ROOT, "docs", "v8_loss_curves.png")
fig.savefig(out, dpi=150)
print("wrote", out)
