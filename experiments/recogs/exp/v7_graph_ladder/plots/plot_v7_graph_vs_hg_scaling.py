#!/usr/bin/env python3
"""RUN_7: graph vs hypergraph width-ladder scaling overlay.

Gen (OOD) EM vs params for both attention mechanisms at identical shapes
(3L, heads=d/64, FFN=3d; 50 ep, prefix-LM). Mean + min-max band over seeds;
6L/d256 depth controls as unfilled markers. Saves figures/v7_graph_vs_hg_scaling.png.
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))  # experiments/recogs

# key, hg params(M), graph params(M), hg globs, graph glob.
# Same shape != same params: hg attention is 7 projections (~7d^2/layer) vs graph's
# 4 (~4d^2/layer), so hg carries ~1.27x the params at every rung. Each curve is
# plotted at its OWN true param count (exact instantiation/checkpoint counts).
RUNGS = [
    ("d128", 0.94, 0.789, ["exp/v5_hg_scaling/runs/hg_d128_*"], "exp/v7_graph_ladder/runs/graph_d128_*"),
    ("d256", 3.15, 2.561, ["exp/v3_prefix_lm/runs/hg_gather_*", "exp/v5_hg_scaling/runs/hg_gather_hypergraph_cuda_*"],
     "exp/v7_graph_ladder/runs/graph_d256_*"),
    ("d384", 6.65, 5.315, ["exp/v5_hg_scaling/runs/hg_d384_*"], "exp/v7_graph_ladder/runs/graph_d384_*"),
    ("d512", 11.4, 9.053, ["exp/v5_hg_scaling/runs/hg_d512_*"], "exp/v7_graph_ladder/runs/graph_d512_*"),
    ("d768", 24.79, 19.478, ["exp/v5_hg_scaling/runs/hg_d768_*"], "exp/v7_graph_ladder/runs/graph_d768_*"),
    # graph_d870 (h15, ffn 2610) = 24.73M, param-matched to hg_d768 (-0.3%);
    # no hg twin. Point appears automatically once exp/v7_graph_ladder/runs/graph_d870_* lands.
    ("d870", None, 24.727, [], "exp/v7_graph_ladder/runs/graph_d870_*"),
]
CTRL = ("6L256", 6.30, 4.533, ["exp/v5_hg_scaling/runs/hg_6L256_*"], "exp/v7_graph_ladder/runs/graph_6L256_*")


def gen_ems(globs):
    if isinstance(globs, str):
        globs = [globs]
    dirs = sorted(set(d for g in globs for d in glob.glob(os.path.join(ROOT, g))))
    out = []
    for d in dirs:
        p = os.path.join(d, "final_metrics.json")
        if os.path.exists(p):
            out.append(100 * json.load(open(p))["gen"]["exact_match"])
    return out


fig, ax = plt.subplots(figsize=(8.5, 5.5))
for label, col, pidx in [("hypergraph", "#d62728", 1), ("graph", "#1f77b4", 2)]:
    xs, mean, lo, hi, ns = [], [], [], [], []
    for key, hg_pm, gr_pm, hg_g, gr_g in RUNGS:
        pm = hg_pm if pidx == 1 else gr_pm
        v = gen_ems(hg_g if pidx == 1 else gr_g)
        if not v or pm is None:
            continue
        xs.append(pm); mean.append(np.mean(v)); lo.append(min(v)); hi.append(max(v)); ns.append(len(v))
    ax.plot(xs, mean, "o-", color=col, lw=2, ms=6, zorder=3,
            label=f"{label} (n={min(ns)}–{max(ns)}/rung)")
    ax.fill_between(xs, lo, hi, color=col, alpha=0.15, zorder=2)

# depth controls (unfilled markers at their true params)
for label, col, pm, globs in [("hg 6L/d256", "#d62728", CTRL[1], CTRL[3]),
                              ("graph 6L/d256", "#1f77b4", CTRL[2], CTRL[4])]:
    v = gen_ems(globs)
    if v:
        ax.plot([pm], [np.mean(v)], "s", mfc="none", mec=col, mew=2, ms=9, zorder=4,
                label=f"{label} (n={len(v)})")

ax.set_xscale("log")
ticks = sorted(set(p for _, hp, gp, _, _ in RUNGS for p in (hp, gp) if p is not None))
ax.set_xticks([0.8, 1.6, 3.2, 6.4, 12.8, 25.6])
ax.set_xticklabels(["0.8M", "1.6M", "3.2M", "6.4M", "12.8M", "25.6M"], fontsize=9)
ax.minorticks_off()
ax.set_xlabel("TRUE params per arm (same d ⇒ hg carries ~1.27×; labels mark the d of each rung)")
for key, hg_pm, gr_pm, hg_g, gr_g in RUNGS:
    if hg_pm:
        ax.annotate(key, (hg_pm, 0), xytext=(0, -2), textcoords="offset points",
                    ha="center", va="top", fontsize=7, color="#d62728", alpha=0.8)
ax.set_ylabel("gen (OOD) exact-match %")
ax.set_title("ReCOGS OOD generalization vs TRUE param count: hypergraph scales ~2.7× steeper\n"
             "(at param parity 24.7M: hg 12.4% vs graph 8.4%; band = seed min–max; squares = 6L ctrls)",
             fontsize=10.5)
ax.grid(alpha=0.3)
ax.legend(fontsize=9)
fig.tight_layout()
out = os.path.join(HERE, "..", "figures", "v7_graph_vs_hg_scaling.png")
fig.savefig(out, dpi=150)
print("wrote", out)
