"""RUN_5 per-category OOD generalization across the hg scaling ladder.

Heatmap: rows = the 21 ReCOGS gen categories (grouped lexical | structural), columns =
rungs ordered by size, cells = mean gen EM % over seeds. Shows WHERE width-scaling helps.
Writes figures/v5_per_category.png. Run with a matplotlib python (e.g. ~/venv/bin/python).
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]  # experiments/recogs
DOCS = Path(__file__).resolve().parents[1] / "figures"
STRUCTURAL = {"obj_pp_to_subj_pp", "pp_recursion", "cp_recursion"}

# rung label -> glob(s) (size order). d256 seeds span exp/v3_prefix_lm/runs (s1-10) + exp/v5_hg_scaling/runs (s11-18).
RUNGS = [
    ("d128", "exp/v5_hg_scaling/runs/hg_d128_*"),
    ("d256", ["exp/v3_prefix_lm/runs/hg_gather_*", "exp/v5_hg_scaling/runs/hg_gather_hypergraph_cuda_*"]),
    ("d384", "exp/v5_hg_scaling/runs/hg_d384_*"),
    ("d512", "exp/v5_hg_scaling/runs/hg_d512_*"),
    ("d768", "exp/v5_hg_scaling/runs/hg_d768_*"),
    ("6L/d256", "exp/v5_hg_scaling/runs/hg_6L256_*"),  # iso-FLOP control, kept last/separate
]


def per_cat_mean(run_glob):
    """{category: mean EM% over seeds}."""
    per_seed = defaultdict(list)  # cat -> [em% per seed]
    globs = [run_glob] if isinstance(run_glob, str) else run_glob
    dirs = []
    for g in globs:
        for d in sorted(glob.glob(str(ROOT / g))):
            if d not in dirs:
                dirs.append(d)
    for d in sorted(dirs):
        f = Path(d) / "per_example_gen.json"
        if not f.exists():
            continue
        c = defaultdict(lambda: [0, 0])
        for r in json.loads(f.read_text()):
            c[r["category"]][0] += r["correct"]
            c[r["category"]][1] += 1
        for cat, (ok, n) in c.items():
            per_seed[cat].append(100 * ok / n)
    return {cat: float(np.mean(v)) for cat, v in per_seed.items()}


data = {lab: per_cat_mean(g) for lab, g in RUNGS}
cats = sorted({c for d in data.values() for c in d}, key=lambda c: (c in STRUCTURAL, c))
labels = [lab for lab, _ in RUNGS]
M = np.array([[data[lab].get(c, np.nan) for lab in labels] for c in cats])

fig, ax = plt.subplots(figsize=(9, 10))
im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=max(1, np.nanmax(M)))
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=0)
ax.set_yticks(range(len(cats)))
ax.set_yticklabels([("* " + c if c in STRUCTURAL else c) for c in cats], fontsize=8)
for i in range(len(cats)):
    for j in range(len(labels)):
        v = M[i, j]
        if np.isfinite(v):
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                    color="white" if v < 0.6 * np.nanmax(M) else "black")
ax.set_title("RUN_5 — per-category OOD EM % across the hg ladder\n(* = structural; last col = 6L/d256 iso-FLOP control)")
fig.colorbar(im, ax=ax, label="gen EM %", fraction=0.04)
fig.tight_layout(); fig.savefig(DOCS / "v5_per_category.png", dpi=130); plt.close(fig)
print("wrote v5_per_category.png |", len(cats), "categories ×", len(labels), "rungs")
