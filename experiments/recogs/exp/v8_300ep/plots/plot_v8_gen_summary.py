#!/usr/bin/env python3
"""RUN_8/8b: 300-epoch hg-vs-graph summary (headline figure).

Panel A: per-seed gen (OOD) EM by arm at 300 ep — hg_d768 (24.8M),
graph_d768 (19.5M, shape-matched), graph_d870 (24.7M, param-matched). Scatter of
every seed + mean (bar) + median (tick), so the reader sees graph's huge seed
variance vs hg's tight cluster, not just the means.

Panel B: the reversal. Same param-matched pair (hg_d768 vs graph_d870) at 50 ep
(RUN_7) and 300 ep (this run). Lines + seed min-max bands; the crossover is the
result — hg leads at 50 ep (+4.0) and graph passes it by 300 ep (-3.8).

Saves figures/v8_gen_summary.png.
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))  # experiments/recogs

HG, GR_SHAPE, GR_PARAM = "#d62728", "#1f77b4", "#ff7f0e"  # red / blue / orange (recogs house colors)


def gens(pattern):
    dirs = sorted(glob.glob(os.path.join(ROOT, pattern)))
    return [100 * json.load(open(os.path.join(d, "final_metrics.json")))["gen"]["exact_match"]
            for d in dirs if os.path.exists(os.path.join(d, "final_metrics.json"))]


fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.6))

# ---------------- Panel A: per-seed distribution by arm (300 ep) ----------------
ARMS = [
    ("hg d768\n24.8M", "exp/v8_300ep/runs/hg_d768_300ep_*", HG),
    ("graph d768\n19.5M (shape-match)", "exp/v8_300ep/runs/graph_d768_300ep_*", GR_SHAPE),
    ("graph d870\n24.7M (param-match)", "exp/v8_300ep/runs/graph_d870_300ep_*", GR_PARAM),
]
rng = np.random.default_rng(0)
for i, (name, pat, col) in enumerate(ARMS):
    v = np.array(gens(pat))
    ax_mean, ax_med = v.mean(), np.median(v)
    axA.bar(i, ax_mean, width=0.62, color=col, alpha=0.22, zorder=1)
    jit = (rng.random(len(v)) - 0.5) * 0.30
    axA.scatter(i + jit, v, s=46, color=col, edgecolor="white", linewidth=0.8, zorder=3)
    axA.hlines(ax_mean, i - 0.31, i + 0.31, color=col, lw=2.4, zorder=4)
    axA.hlines(ax_med, i - 0.31, i + 0.31, color=col, lw=1.3, ls=":", zorder=4)
    axA.annotate(f"mean {ax_mean:.1f}\n(n={len(v)})", (i, ax_mean), xytext=(0, 8),
                 textcoords="offset points", ha="center", va="bottom", fontsize=9,
                 color=col, fontweight="bold")
axA.set_xticks(range(len(ARMS)))
axA.set_xticklabels([a[0] for a in ARMS], fontsize=9)
axA.set_ylabel("gen (OOD) exact-match %")
axA.set_ylim(0, 48)
axA.set_title("A · Per-seed gen EM by arm at 300 ep\n"
              "bar = mean, dotted = median, dots = seeds", fontsize=10.5)
axA.grid(axis="y", alpha=0.3)

# ---------------- Panel B: the param-matched reversal (50 -> 300 ep) ----------------
EPOCHS = [50, 300]
SERIES = [
    ("hg d768 (24.8M)", HG, ["exp/v5_hg_scaling/runs/hg_d768_*", "exp/v8_300ep/runs/hg_d768_300ep_*"]),
    ("graph d870 (24.7M, param-match)", GR_PARAM, ["exp/v7_graph_ladder/runs/graph_d870_*", "exp/v8_300ep/runs/graph_d870_300ep_*"]),
]
means = {}
for name, col, pats in SERIES:
    m, lo, hi = [], [], []
    for p in pats:
        v = np.array(gens(p))
        m.append(v.mean()); lo.append(v.min()); hi.append(v.max())
    means[name] = m
    axB.fill_between(EPOCHS, lo, hi, color=col, alpha=0.13, zorder=1)
    axB.plot(EPOCHS, m, "o-", color=col, lw=2.4, ms=8, zorder=3, label=name)
# value labels: put the higher series' label above and the lower one below at each epoch
hg_m = means["hg d768 (24.8M)"]
gr_m = means["graph d870 (24.7M, param-match)"]
for j, x in enumerate(EPOCHS):
    hi_above = hg_m[j] >= gr_m[j]
    axB.annotate(f"{hg_m[j]:.1f}", (x, hg_m[j]), xytext=(0, 10 if hi_above else -17),
                 textcoords="offset points", ha="center", fontsize=9, color=HG, fontweight="bold")
    axB.annotate(f"{gr_m[j]:.1f}", (x, gr_m[j]), xytext=(0, -17 if hi_above else 10),
                 textcoords="offset points", ha="center", fontsize=9, color=GR_PARAM, fontweight="bold")
# crossover deltas, parked in open space away from the points
d50 = hg_m[0] - gr_m[0]
d300 = hg_m[1] - gr_m[1]
axB.annotate(f"50 ep: hg leads +{d50:.1f}", (105, 8.0), ha="left", fontsize=9, color="#555")
axB.annotate(f"300 ep: graph leads +{-d300:.1f}", (295, 22.0), ha="right", fontsize=9, color="#555")
axB.set_xticks(EPOCHS); axB.set_xticklabels(["50 ep\n(RUN_7)", "300 ep\n(RUN_8)"])
axB.set_xlim(20, 330)
axB.set_ylabel("gen (OOD) exact-match %")
axB.set_title("B · At equal params, the hg lead reverses with training\n"
              "band = seed min–max", fontsize=10.5)
axB.grid(alpha=0.3)
axB.legend(fontsize=9, loc="upper left")

fig.suptitle("ReCOGS 300-epoch A/B: hg's param-matched OOD edge is a short-training effect "
             "(graph overtakes by 300 ep; high graph seed variance)", fontsize=12, y=1.02)
fig.tight_layout()
out = os.path.join(HERE, "..", "figures", "v8_gen_summary.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
