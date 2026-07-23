#!/usr/bin/env python3
"""RUN_7: depth effect at fixed width (d256) — 3L vs 6L for graph and hypergraph.

Slope chart: x = layers (3, 6), gen (OOD) EM on y. One line per mechanism
(mean over seeds), individual seeds as jittered dots — important because the
four cells have very different n (hg 3L n=18, hg 6L n=2, graph n=10 each).
Everything else identical: d256, h4, ffn 768, prefix-LM, 50 ep, seq 512.

Saves figures/v7_depth_at_d256.png.
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))  # experiments/recogs

CELLS = {  # (mechanism, layers) -> globs
    ("hypergraph", 3): ["exp/v3_prefix_lm/runs/hg_gather_*", "exp/v5_hg_scaling/runs/hg_gather_hypergraph_cuda_*"],
    ("hypergraph", 6): ["exp/v5_hg_scaling/runs/hg_6L256_*"],
    ("graph", 3): ["exp/v7_graph_ladder/runs/graph_d256_*"],
    ("graph", 6): ["exp/v7_graph_ladder/runs/graph_6L256_*"],
}
COLOR = {"hypergraph": "#d62728", "graph": "#1f77b4"}


def gen_ems(globs):
    dirs = sorted(set(d for g in globs for d in glob.glob(os.path.join(ROOT, g))))
    out = []
    for d in dirs:
        p = os.path.join(d, "final_metrics.json")
        if os.path.exists(p):
            out.append(100 * json.load(open(p))["gen"]["exact_match"])
    return out


rng = np.random.default_rng(0)  # fixed jitter so the figure is reproducible
fig, ax = plt.subplots(figsize=(7, 5.5))
for mech in ("hypergraph", "graph"):
    xs, means = [], []
    for L in (3, 6):
        v = gen_ems(CELLS[(mech, L)])
        xs.append(L); means.append(np.mean(v))
        jitter = rng.uniform(-0.18, 0.18, len(v)) + (L - 0.35 if mech == "hypergraph" else L + 0.35)
        ax.plot(jitter, v, "o", color=COLOR[mech], alpha=0.45, ms=5, zorder=2)
        ax.annotate(f"n={len(v)}", (jitter.mean(), max(v) + 0.5), color=COLOR[mech],
                    ha="center", fontsize=8)
    ax.plot([2.65, 5.65] if mech == "hypergraph" else [3.35, 6.35], means, "-",
            color=COLOR[mech], lw=2.5, zorder=3)
    ax.plot([2.65, 5.65] if mech == "hypergraph" else [3.35, 6.35], means, "o",
            color=COLOR[mech], ms=9, zorder=4, label=f"{mech} (mean)")
    for x, m in zip([2.65, 5.65] if mech == "hypergraph" else [3.35, 6.35], means):
        ax.annotate(f"{m:.1f}", (x, m), textcoords="offset points",
                    xytext=(-14 if mech == "hypergraph" else 14, 0),
                    ha="right" if mech == "hypergraph" else "left",
                    color=COLOR[mech], fontweight="bold", fontsize=10)

ax.set_xticks([3, 6])
ax.set_xticklabels(["3 layers", "6 layers"])
ax.set_xlim(1.8, 7.5)
ax.set_ylabel("gen (OOD) exact-match %")
ax.set_xlabel("depth at fixed width (d256, h4, ffn 768 — everything else identical)")
ax.set_title("Depth helps graph, hurts hypergraph (ReCOGS OOD, d256)\n"
             "dots = individual seeds; hg 6L is n=2 — treat as a range", fontsize=11)
ax.grid(axis="y", alpha=0.3)
ax.legend(fontsize=10)
fig.tight_layout()
out = os.path.join(HERE, "..", "figures", "v7_depth_at_d256.png")
fig.savefig(out, dpi=150)
print("wrote", out)
