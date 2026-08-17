"""Seed x category gen exact-match matrices, side by side per arm.

Visual companion to analysis/seed_category_correlation.py (colleague's test:
if seed sensitivity = "did SGD find the structural solution", a good seed
should be good across many categories at once -> coherent dark/light columns).

One panel per arm, same category rows everywhere, seeds sorted best->worst,
one shared color scale so brightness is comparable across panels. The n=4
hg 300-epoch arm is shown as raw data only — too few seeds for across-seed
correlation stats (those live in analysis/seed_category_correlation.py for
the well-seeded arms).
"""
import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from plots_common import LABELS, STRUCTURAL

BASE = os.path.join(os.path.dirname(__file__), "..", "exp")
OUT = os.path.join(os.path.dirname(__file__), "figures", "seed_category_matrix.png")

ARMS = [
    ("graph d870\n300 epochs (8 seeds)", f"{BASE}/v8_300ep/runs/graph_d870_300ep_graph_flash_s*", "Blues"),
    ("hypergraph d768\n300 epochs (4 seeds)", f"{BASE}/v8_300ep/runs/hg_d768_300ep_hypergraph_cuda_s*", "Reds"),
    ("hypergraph d768\n50 epochs (10 seeds)", f"{BASE}/v5_hg_scaling/runs/hg_d768_hypergraph_cuda_s*", "Reds"),
]


def load(globpat):
    """{seed: {cat: em}} from final_metrics.json."""
    rows = {}
    for d in sorted(glob.glob(globpat)):
        p = os.path.join(d, "final_metrics.json")
        if not os.path.exists(p):
            continue
        m = json.load(open(p))
        seed = int(d.rsplit("_s", 1)[1])
        rows[seed] = {c: v["exact_match"] for c, v in m["gen"]["per_category"].items()}
    return rows


arms = [(title, load(pat), cmap) for title, pat, cmap in ARMS]

# One fixed category order for every panel: lexical sorted by the first arm's
# mean (desc), the 3 structural categories pinned at the bottom.
ref = arms[0][1]
cats_all = sorted(ref[next(iter(ref))])
ref_mean = {c: np.mean([ref[s][c] for s in ref]) for c in cats_all}
lex = sorted([c for c in cats_all if c not in STRUCTURAL], key=lambda c: -ref_mean[c])
struct = sorted(STRUCTURAL, key=lambda c: -ref_mean[c])
cats = lex + struct

mats = []
for _, data, _ in arms:
    seeds = sorted(data)
    M = np.array([[data[s][c] for c in cats] for s in seeds])
    order = np.argsort(-M.mean(axis=1))
    mats.append((M[order], [seeds[i] for i in order]))

vmax = max(M.max() for M, _ in mats)  # shared scale across panels

fig, axes = plt.subplots(
    1, len(arms), figsize=(13.5, 8.5), sharey=True,
    gridspec_kw={"width_ratios": [len(s) for _, s in mats], "wspace": 0.06},
)

ims = []
for ax, (title, _, cmap_name), (M, seeds) in zip(axes, arms, mats):
    ims.append(ax.imshow(M.T, aspect="auto", cmap=cmap_name, vmin=0.0, vmax=vmax,
                         interpolation="nearest"))
    ax.set_xticks(range(len(seeds)), [f"s{s}" for s in seeds], fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

# One shared x-axis note (per-panel xlabels collide at these panel widths).
axes[1].set_xlabel("seed (columns ordered by mean EM over categories, best → worst)",
                   fontsize=9)

# One colorbar per colormap: Blues for the graph panel, a single Reds bar
# spanning both hypergraph panels (identical vmin/vmax everywhere).
cb = fig.colorbar(ims[0], ax=axes[0], location="bottom", fraction=0.035, pad=0.09)
cb.ax.tick_params(labelsize=8, length=0)
cb.outline.set_visible(False)
cb = fig.colorbar(ims[1], ax=list(axes[1:]), location="bottom", fraction=0.035, pad=0.09)
cb.ax.tick_params(labelsize=8, length=0)
cb.outline.set_visible(False)
cb.set_label("gen exact match (greedy decoding, semantic EM)", fontsize=9)

axes[0].set_yticks(range(len(cats)),
                   [LABELS.get(c, c) + (" †" if c in STRUCTURAL else "") for c in cats],
                   fontsize=8.5)
for i, c in enumerate(cats):
    if c in STRUCTURAL:
        axes[0].get_yticklabels()[i].set_fontweight("bold")

fig.suptitle(
    "Gen exact match per seed × category  († bold = structural categories)",
    fontsize=12, y=0.97,
)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("wrote", OUT)
