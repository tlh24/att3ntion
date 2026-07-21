#!/usr/bin/env python3
"""hg_full (scatter) vs hg_gather (no-scatter), 3L, same config — RUN_4 follow-up.

All 10 scatter seeds DIVERGE (train loss > 1000 guard). This plots the train-loss
trajectories: hg_gather converging vs hg_full climbing, with each scatter seed's
divergence point (from the abort message) marked with an ✕.

Reads eval_curve.tsv files pulled to a local dir (default: arg 1).
"""
import glob, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = sys.argv[2] if len(sys.argv) > 2 else "hg_full_scatter_divergence_v4.png"

# (seed -> (divergence step, train loss at abort)) from the RuntimeError messages.
DIVERGE = {1: (558, 1240.9), 2: (231, 3398.4), 3: (1058, 1525.9), 4: (1601, 1390.9),
           5: (725, 1507.3), 6: (1055, 2271.9), 7: (642, 1910.7), 8: (279, 1189.2),
           9: (1155, 6025.9), 10: (593, 2035.9)}
THRESH = 1000.0


def load(pat):
    out = {}
    for p in sorted(glob.glob(os.path.join(D, pat))):
        s = int(os.path.basename(p).split("_s")[-1].split(".")[0])
        a = np.atleast_1d(np.genfromtxt(p, delimiter="\t", names=True))
        if a.size and np.isfinite(a["step"]).any():
            out[s] = (np.atleast_1d(a["step"]), np.atleast_1d(a["train_loss"]))
    return out

gather = load("hg_gather_s*.tsv")
full = load("hg_full_s*.tsv")

fig, ax = plt.subplots(figsize=(11, 6.5))

for i, (s, (step, tr)) in enumerate(sorted(gather.items())):
    ax.plot(step, np.clip(tr, 1e-5, None), color="#2ca02c", alpha=0.55, lw=1.2,
            label="hg_gather (no scatter) — converges" if i == 0 else None)

for i, (s, (step, tr)) in enumerate(sorted(full.items())):
    ds, dl = DIVERGE[s]
    xs = np.append(step, ds)
    ys = np.append(np.clip(tr, 1e-5, None), dl)        # extend to the abort point
    ax.plot(xs, ys, color="#d62728", alpha=0.6, lw=1.2,
            label="hg_full (scatter) — diverges" if i == 0 else None)
    ax.scatter([ds], [dl], marker="x", color="#d62728", s=55, zorder=5,
               label="divergence (loss > 1000)" if i == 0 else None)

ax.axhline(THRESH, ls="--", color="#888", lw=1, label="divergence guard (1000)")
ax.set_yscale("log")
ax.set_xlabel("training step")
ax.set_ylabel("train loss (log)")
ax.set_xlim(0, 1800)
ax.set_title("ReCOGS hg_full (scatter) vs hg_gather (no scatter), 3L, identical config\n"
             "all 10 scatter seeds diverge at step 231–1601 (~epoch 1–8); no-scatter converges to ~1e-3",
             fontsize=11)
ax.legend(loc="center right", fontsize=9)
ax.grid(True, which="both", alpha=0.25)
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print("wrote", OUT)
print(f"hg_gather seeds: {len(gather)}  hg_full seeds: {len(full)} (all diverged)")
