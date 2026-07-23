#!/usr/bin/env python3
"""Live monitor for the ReCOGS v3 overnight sweep.

Two things in one place:
  1. A status table over every run dir under --runs-dir: latest step, the most
     recent teacher-forced train/dev/test loss (from eval_curve.tsv), projected
     train time (timing.json), and final gen EM once the run has finished.
  2. A train/dev/test loss-curve PNG (mean over seeds per arm), so the
     in-distribution learning curve is visible *while training is running*.

Usage (on either pod, from experiments/recogs):
    /opt/conda/bin/python monitor_v3.py                 # status table
    /opt/conda/bin/python monitor_v3.py --plot          # + write loss-curve PNG
    watch -n 30 /opt/conda/bin/python monitor_v3.py     # refresh every 30s

Reads only files the trainer already writes (eval_curve.tsv, timing.json,
final_metrics.json) — safe to run against an in-progress sweep.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _arm_of(run_name: str) -> str:
    if "hypergraph_cuda" in run_name:
        return "hypergraph"
    if "graph_flash" in run_name:
        return "graph_flash"
    return "other"


def _read_curve_rows(f: Path):
    """Parse eval_curve.tsv into [(step, train, dev, test)], skipping malformed
    rows. Files can carry stray NUL bytes (concurrent writes / sparse blocks), so
    sanitize and tolerate partial last lines rather than crashing mid-sweep."""
    if not f.exists():
        return []
    text = f.read_text(errors="replace").replace("\x00", "")
    out = []
    for ln in text.splitlines():
        if not ln or ln.startswith("step"):
            continue
        parts = ln.split("\t")
        if len(parts) != 4:
            continue
        try:
            out.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])))
        except ValueError:
            continue
    return out


def _last_eval_row(run_dir: Path):
    rows = _read_curve_rows(run_dir / "eval_curve.tsv")
    if not rows:
        return None
    step, tr, dv, te = rows[-1]
    return step, tr, dv, te, len(rows)


def status(runs_dir: Path) -> None:
    run_dirs = sorted(p.parent for p in runs_dir.glob("*/config.json"))
    if not run_dirs:
        print(f"(no runs under {runs_dir} yet)")
        return
    hdr = f"{'run':38s} {'step':>6s} {'train':>8s} {'dev':>8s} {'test':>8s} {'proj_h':>6s} {'gen_EM':>7s}  state"
    print(hdr)
    print("-" * len(hdr))
    for rd in run_dirs:
        row = _last_eval_row(rd)
        step = f"{row[0]}" if row else "-"
        tr = f"{row[1]:.4f}" if row else "-"
        dv = f"{row[2]:.4f}" if row else "-"
        te = f"{row[3]:.4f}" if row else "-"
        proj = "-"
        t = rd / "timing.json"
        if t.exists():
            tj = json.loads(t.read_text())
            ps = tj.get("projected_train_sec")
            if ps:
                proj = f"{ps / 3600:.2f}"
        gen_em = "-"
        state = "running"
        fm = rd / "final_metrics.json"
        if fm.exists():
            m = json.loads(fm.read_text())
            if "gen" in m:
                gen_em = f"{100 * m['gen']['exact_match']:.2f}"
            state = "DONE"
        print(f"{rd.name:38s} {step:>6s} {tr:>8s} {dv:>8s} {te:>8s} {proj:>6s} {gen_em:>7s}  {state}")


def plot(runs_dir: Path, out: Path) -> None:
    import collections

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # arm -> metric -> list of (steps[], values[]) per seed
    curves: dict[str, dict[str, list]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for f in sorted(runs_dir.glob("*/eval_curve.tsv")):
        arm = _arm_of(f.parent.name)
        rows = _read_curve_rows(f)
        if not rows:
            continue
        steps = np.array([r[0] for r in rows])
        for j, metric in enumerate(("train", "dev", "test"), start=1):
            curves[arm][metric].append((steps, np.array([r[j] for r in rows])))

    if not curves:
        print("(no eval_curve.tsv data to plot yet)")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"hypergraph": "#d62728", "graph_flash": "#1f77b4", "other": "#555"}
    styles = {"train": ":", "dev": "-", "test": "--"}
    for arm, metrics in sorted(curves.items()):
        for metric, seeds in metrics.items():
            # Plot each seed faintly; the mean is implicit in their overlap.
            for steps, vals in seeds:
                ax.plot(steps, vals, styles[metric], color=colors.get(arm, "#555"),
                        alpha=0.35, lw=1.0)
        # legend proxy (one entry per arm/metric)
        for metric in ("train", "dev", "test"):
            if metrics.get(metric):
                ax.plot([], [], styles[metric], color=colors.get(arm, "#555"),
                        label=f"{arm} {metric}")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("teacher-forced loss")
    ax.set_yscale("log")
    ax.set_title("ReCOGS v3 in-distribution learning curves (per seed)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print("wrote", out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Monitor the ReCOGS v3 sweep")
    ap.add_argument("--runs-dir", type=Path, default=Path(__file__).parent / "runs")
    ap.add_argument("--plot", action="store_true", help="also write a loss-curve PNG")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "figures" / "v3_loss_curves.png")
    args = ap.parse_args()
    status(args.runs_dir)
    if args.plot:
        plot(args.runs_dir, args.out)


if __name__ == "__main__":
    main()
