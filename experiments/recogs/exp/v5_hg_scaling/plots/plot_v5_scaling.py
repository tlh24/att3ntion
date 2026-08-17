"""RUN_5 hypergraph scaling figures (Chinchilla-style).

Produces three panels into figures/:
  v5_loss_curves.png   - train+dev+test loss vs step, mean over seeds, one line-set per rung.
  v5_scaling.png       - gen EM (OOD, mean + min/max) vs params (log-x) and per-epoch FLOPs;
                         iso-FLOP pair (3L/d512 vs 6L/d256) highlighted.
  v5_gen_vs_test.png   - gen EM and in-dist test EM vs params, both rungs on one axis.

Reads eval_curve.tsv (step,train,dev,test) + final_metrics.json directly. d256 anchor
reused from exp/v3_prefix_lm/runs. Run with a python that has matplotlib (e.g. ~/venv/bin/python).
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]  # experiments/recogs
DOCS = Path(__file__).resolve().parents[1] / "figures"

# rung -> (glob(s), per-epoch TF, params(M), label, color, is_control)
# glob may be a single pattern or a list of patterns (d256 seeds span exp/v3_prefix_lm/runs + exp/v5_hg_scaling/runs).
RUNGS = [
    ("d128",  "exp/v5_hg_scaling/runs/hg_d128_*",   124, 0.94, "3L/d128",  "#9467bd", False),
    ("d256",  ["exp/v3_prefix_lm/runs/hg_gather_*", "exp/v5_hg_scaling/runs/hg_gather_hypergraph_cuda_*"],
                                      252, 3.15, "3L/d256*", "#1f77b4", False),
    ("d384",  "exp/v5_hg_scaling/runs/hg_d384_*",   385, 6.65, "3L/d384",  "#2ca02c", False),
    ("d512",  "exp/v5_hg_scaling/runs/hg_d512_*",   521, 11.4, "3L/d512",  "#d62728", False),
    ("6L256", "exp/v5_hg_scaling/runs/hg_6L256_*",  503, 6.30, "6L/d256",  "#ff7f0e", True),
    ("d768",  "exp/v5_hg_scaling/runs/hg_d768_*",   807, 24.8, "3L/d768",  "#8c564b", False),
]


def _dirs(run_glob):
    """Expand a glob (or list of globs) to sorted, de-duplicated run dirs."""
    globs = [run_glob] if isinstance(run_glob, str) else run_glob
    seen = []
    for g in globs:
        for d in sorted(glob.glob(str(ROOT / g))):
            if d not in seen:
                seen.append(d)
    return sorted(seen)


def gen_test_em(run_glob):
    """[(gen%, test%)] per seed that has final_metrics.json."""
    out = []
    for d in _dirs(run_glob):
        fm = Path(d) / "final_metrics.json"
        if not fm.exists():
            continue
        m = json.loads(fm.read_text())
        out.append((100 * m["gen"]["exact_match"], 100 * m["test"]["exact_match"]))
    return out


def load_curves(run_glob):
    """[(step, train, dev, test)] arrays per seed; drop non-finite/<=0 loss rows."""
    curves = []
    for d in _dirs(run_glob):
        f = Path(d) / "eval_curve.tsv"
        if not f.exists():
            continue
        a = np.genfromtxt(f, delimiter="\t", names=True)
        if a.size == 0:
            continue
        a = np.atleast_1d(a)
        ok = np.isfinite(a["dev_loss"]) & (a["dev_loss"] > 0) & np.isfinite(a["train_loss"]) & (a["train_loss"] > 0)
        if ok.sum() < 2:
            continue
        curves.append((a["step"][ok], a["train_loss"][ok], a["dev_loss"][ok], a["test_loss"][ok]))
    return curves


def mean_on_grid(curves, idx):
    """Mean over seeds of column idx (1=train,2=dev,3=test) on a shared step grid."""
    if not curves:
        return None, None
    lo = max(c[0].min() for c in curves)
    hi = min(c[0].max() for c in curves)
    grid = np.linspace(lo, hi, 100)
    stack = [np.interp(grid, c[0], c[idx]) for c in curves]
    return grid, np.mean(stack, axis=0)


# ---- Panel 1: loss curves ----
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
for label_i, (which, idx) in enumerate([("train", 1), ("dev", 2), ("test", 3)]):
    ax = axes[label_i]
    for key, g, tf, pm, lab, col, ctrl in RUNGS:
        curves = load_curves(g)
        grid, mean = mean_on_grid(curves, idx)
        if grid is None:
            continue
        ls = "--" if ctrl else "-"
        ax.plot(grid, mean, ls, color=col, label=f"{lab} ({pm}M, n={len(curves)})", lw=1.8)
    ax.set_yscale("log"); ax.set_xlabel("optimizer step"); ax.set_title(f"{which} loss")
    if label_i == 0:
        ax.set_ylabel("teacher-forced loss (log)"); ax.legend(fontsize=8)
fig.suptitle("RUN_5 hg scaling — train / dev / test loss vs step (mean over seeds)")
fig.tight_layout(); fig.savefig(DOCS / "v5_loss_curves.png", dpi=130); plt.close(fig)

# ---- Panel 2: scaling curve (gen EM vs params and vs FLOPs) ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
for ax, (xkey, xlab) in zip(axes, [("params", "params (M, log)"), ("flops", "per-epoch fwd FLOPs (TF, log)")]):
    xs_line, ys_line = [], []
    for key, g, tf, pm, lab, col, ctrl in RUNGS:
        ems = [e[0] for e in gen_test_em(g)]
        if not ems:
            continue
        x = pm if xkey == "params" else tf
        mean = np.mean(ems)
        ax.errorbar(x, mean, yerr=[[mean - min(ems)], [max(ems) - mean]], fmt="o",
                    color=col, capsize=4, ms=8, zorder=3)
        ax.annotate(f"{lab}\n{mean:.1f}%", (x, mean), textcoords="offset points",
                    xytext=(8, 6), fontsize=8, color=col)
        if not ctrl:  # primary width ladder defines the trend line
            xs_line.append(x); ys_line.append(mean)
    order = np.argsort(xs_line)
    ax.plot(np.array(xs_line)[order], np.array(ys_line)[order], "-", color="#444", alpha=0.5, zorder=1)
    ax.set_xscale("log"); ax.set_xlabel(xlab); ax.set_ylabel("gen (OOD) EM %")
    ax.grid(True, which="both", alpha=0.25)
axes[0].set_title("OOD generalization vs model size")
axes[1].set_title("OOD vs compute — dashed=6L/d256 (iso-FLOP depth control)")
fig.suptitle("RUN_5 — hypergraph OOD generalization scaling (width ladder; * = reused d256 anchor)")
fig.tight_layout(); fig.savefig(DOCS / "v5_scaling.png", dpi=130); plt.close(fig)

# ---- Panel 3: gen vs test on one axis ----
fig, ax = plt.subplots(figsize=(8, 5.2))
for key, g, tf, pm, lab, col, ctrl in RUNGS:
    rows = gen_test_em(g)
    if not rows:
        continue
    gen = np.mean([r[0] for r in rows]); tst = np.mean([r[1] for r in rows])
    m = "s" if ctrl else "o"
    ax.scatter(pm, gen, color=col, marker=m, s=70, zorder=3)
    ax.scatter(pm, tst, color=col, marker=m, s=70, facecolors="none", zorder=3)
    ax.annotate(lab, (pm, gen), textcoords="offset points", xytext=(6, -10), fontsize=8, color=col)
ax.set_xscale("log"); ax.set_xlabel("params (M, log)"); ax.set_ylabel("EM %")
ax.set_title("filled = gen (OOD)   hollow = test (in-dist)")
ax.grid(True, which="both", alpha=0.25)
fig.tight_layout(); fig.savefig(DOCS / "v5_gen_vs_test.png", dpi=130); plt.close(fig)

print("wrote:", *(p.name for p in [DOCS / "v5_loss_curves.png", DOCS / "v5_scaling.png", DOCS / "v5_gen_vs_test.png"]))
for key, g, tf, pm, lab, col, ctrl in RUNGS:
    rows = gen_test_em(g)
    if rows:
        gens = [r[0] for r in rows]
        print(f"  {lab:9} n={len(rows)}  gen={np.mean(gens):5.2f}% [{min(gens):.2f}-{max(gens):.2f}]  test={np.mean([r[1] for r in rows]):5.2f}%")
