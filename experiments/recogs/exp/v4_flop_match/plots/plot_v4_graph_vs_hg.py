#!/usr/bin/env python3
"""RUN_4 (v4): graph (FLOP-matched) vs hypergraph training comparison.

Left panel  : dev + test loss trajectory over training (mean +/- std band over
              seeds), graph_flop6L (exp/v4_flop_match/runs) vs hg_gather (exp/v3_prefix_lm/runs).
Right panel : final generalization (gen) exact-match %, mean +/- std over seeds.

Robust to runs still in progress / missing files (skips them). Re-runnable any
time. Saves figures/graph_vs_hg_training_v4.png.
"""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))  # experiments/recogs

# arm label -> (glob of run dirs, color)
ARMS = {
    "hypergraph (hg_gather 3L)": (
        os.path.join(ROOT, "exp/v3_prefix_lm/runs", "hg_gather_*"), "#c1121f"),
    "graph (FLOP-matched 6L/d1088)": (
        os.path.join(ROOT, "exp/v4_flop_match/runs", "graph_flop6L_*"), "#1d4e89"),
}


def load_curves(run_glob):
    """Return list of (steps, dev, test) arrays, one per seed with a curve."""
    out = []
    for d in sorted(glob.glob(run_glob)):
        p = os.path.join(d, "eval_curve.tsv")
        if not os.path.isfile(p):
            continue
        try:
            arr = np.genfromtxt(p, delimiter="\t", names=True)
            if arr.size == 0:
                continue
            step = np.atleast_1d(arr["step"])
            dev = np.atleast_1d(arr["dev_loss"])
            test = np.atleast_1d(arr["test_loss"])
            # Drop any non-finite rows (a malformed/partial line in one seed's
            # tsv otherwise poisons the whole arm's min/max -> empty band).
            ok = np.isfinite(step) & np.isfinite(dev) & np.isfinite(test) & (dev > 0) & (test > 0)
            if ok.sum() < 2:
                continue
            out.append((step[ok], dev[ok], test[ok]))
        except Exception:
            continue
    return out


def mean_band(curves, which):
    """Mean/std loss over seeds on a shared grid spanning each arm's FULL range.

    Seeds early-stop at different steps, so we span [min start, max end] and at
    each grid point average only the seeds whose own range covers it (NaN-masked)
    — avoids the intersection collapsing an arm's curve to invisibility.
    """
    idx = 1 if which == "dev" else 2
    if not curves:
        return None
    lo = min(float(c[0].min()) for c in curves)
    hi = max(float(c[0].max()) for c in curves)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    grid = np.linspace(lo, hi, 300)
    # np.interp clamps outside each seed's range (early-stopped seeds hold their
    # final loss flat), so every column has data — no empty/NaN band.
    M = np.vstack([np.interp(grid, c[0], c[idx]) for c in curves])
    return grid, M.mean(0), M.std(0)


def final_gen(run_glob):
    """Return list of final gen exact-match % per seed."""
    out = []
    for d in sorted(glob.glob(run_glob)):
        p = os.path.join(d, "final_metrics.json")
        if not os.path.isfile(p):
            continue
        try:
            m = json.load(open(p))
            if "gen" in m and "exact_match" in m["gen"]:
                out.append(100.0 * float(m["gen"]["exact_match"]))
        except Exception:
            continue
    return out


def main():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2),
                                   gridspec_kw={"width_ratios": [2.1, 1]})

    n_seeds = {}
    plotted = 0
    for label, (g, color) in ARMS.items():
        curves = load_curves(g)
        n_seeds[label] = len(curves)
        for which, ls, alpha in (("dev", "-", 0.18), ("test", "--", 0.12)):
            band = mean_band(curves, which)
            if band is None:
                continue
            grid, mu, sd = band
            axL.plot(grid, mu, ls, color=color, lw=2,
                     label=f"{label} — {which}")
            axL.fill_between(grid, mu - sd, mu + sd, color=color, alpha=alpha, lw=0)
            plotted += 1

    axL.set_xlabel("training step")
    axL.set_ylabel("loss")
    if plotted:
        axL.set_yscale("log")  # only safe once we have positive loss data
    axL.set_title("Training loss (mean ± std over seeds)\nsolid = dev, dashed = test")
    axL.legend(fontsize=8, loc="upper right")
    axL.grid(True, which="both", alpha=0.25)

    # right: final gen EM
    labels, means, stds, colors = [], [], [], []
    for label, (g, color) in ARMS.items():
        vals = final_gen(g)
        short = "hypergraph" if "hypergraph" in label else "graph\n(FLOP-matched)"
        labels.append(f"{short}\n(n={len(vals)})")
        means.append(np.mean(vals) if vals else 0.0)
        stds.append(np.std(vals) if vals else 0.0)
        colors.append(color)
    x = np.arange(len(labels))
    axR.bar(x, means, yerr=stds, color=colors, alpha=0.85, capsize=6, width=0.6)
    for xi, m in zip(x, means):
        axR.text(xi, m, f"{m:.1f}%", ha="center", va="bottom", fontsize=10)
    axR.set_xticks(x)
    axR.set_xticklabels(labels, fontsize=9)
    axR.set_ylabel("generalization exact-match (%)")
    axR.set_title("Final gen EM\n(mean ± std over seeds)")
    axR.grid(True, axis="y", alpha=0.25)

    fig.suptitle("ReCOGS RUN_4: graph (FLOP-matched) vs hypergraph", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(HERE, "..", "figures", "graph_vs_hg_training_v4.png")
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")
    print("seeds with curves:", n_seeds)


if __name__ == "__main__":
    main()
