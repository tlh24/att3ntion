#!/usr/bin/env python3
"""RUN_8/8b: per-category gen (OOD) EM, hg_d768 vs graph_d870 at 300 ep.

Horizontal bars (mean over seeds, whisker = ±1 SD across seeds) per ReCOGS gen
category, sorted by hg mean. SD is n-independent, so the hg (4 seeds) and graph
(8 seeds) spreads stay comparable. A separate row at the top shows
in-distribution (test-split) EM. Saves figures/v8_per_category.png.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))  # experiments/recogs
sys.path.insert(0, ROOT)
from plots_common import per_category, overall, run_dirs, label, STRUCTURAL  # noqa: E402

HG, GR = "#d62728", "#1f77b4"  # hg red, graph blue
HG_DIRS = run_dirs(ROOT, "exp/v8_300ep/runs/hg_d768_300ep_*")
GR_DIRS = run_dirs(ROOT, "exp/v8_300ep/runs/graph_d870_300ep_*")

hg = per_category(HG_DIRS, "gen")
gr = per_category(GR_DIRS, "gen")
cats = sorted(set(hg) & set(gr), key=lambda c: np.mean(hg[c]))  # sort by hg mean

# In-distribution (test split) overall EM per seed, one bar per arm.
hg_id = overall(HG_DIRS, "test")
gr_id = overall(GR_DIRS, "test")

GAP = 1.6                         # gap between the OOD block and the in-dist row
y = np.arange(len(cats))          # OOD category rows
y_id = len(cats) - 1 + GAP        # in-distribution row, sits above the OOD block
h = 0.38
fig, ax = plt.subplots(figsize=(10.5, 9.4))

ERRKW = dict(ecolor="#333", elinewidth=0.9, capsize=2.5, alpha=0.8, zorder=3)

for arm, data, idvals, col, off, name in [
        ("hg", hg, hg_id, HG, +h / 2, "hg d768"),
        ("gr", gr, gr_id, GR, -h / 2, "graph d870")]:
    m = np.array([np.mean(data[c]) for c in cats])
    sd = np.array([np.std(data[c], ddof=1) if len(data[c]) > 1 else 0.0 for c in cats])
    ax.barh(y + off, m, height=h, color=col, alpha=0.85, zorder=2, label=name)
    ax.errorbar(m, y + off, xerr=sd, fmt="none", **ERRKW)
    # in-distribution bar + ±1 SD for this arm
    id_sd = np.std(idvals, ddof=1) if len(idvals) > 1 else 0.0
    ax.barh(y_id + off, np.mean(idvals), height=h, color=col, alpha=0.85, zorder=2)
    ax.errorbar(np.mean(idvals), y_id + off, xerr=id_sd, fmt="none", **ERRKW)

# separator between OOD categories and the in-distribution row
ax.axhline(len(cats) - 1 + GAP / 2, color="#999", lw=0.8, ls="--", alpha=0.7)

yticks = list(y) + [y_id]
lbls = [("* " if c in STRUCTURAL else "") + label(c) for c in cats]
lbls += ["in-distribution (test)"]
ax.set_yticks(yticks)
ax.set_yticklabels(lbls, fontsize=8)
ax.get_yticklabels()[-1].set_fontweight("bold")
ax.set_ylim(-0.6, y_id + 0.9)
ax.set_xlabel("exact-match %  (bar = mean over seeds, whisker = ±1 SD)")
ax.set_xlim(0, 100)
ax.set_title("ReCOGS OOD generalization at 300 epochs: hg d768 vs graph d870", fontsize=12)
ax.grid(axis="x", alpha=0.3)
ax.legend(fontsize=9, loc="lower right")
fig.tight_layout()
out = os.path.join(HERE, "..", "figures", "v8_per_category.png")
fig.savefig(out, dpi=150)
print("wrote", out)
print(f"in-dist test EM: hg {np.mean(hg_id):.2f}%  graph {np.mean(gr_id):.2f}%")
