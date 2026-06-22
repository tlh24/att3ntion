from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate ReCOGS final metrics across runs")
    parser.add_argument("--runs-dir", type=Path, default=Path(__file__).parent / "runs")
    parser.add_argument("--output-json", type=Path, default=Path(__file__).parent / "aggregate_summary.json")
    return parser.parse_args()


def _mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    arr = np.array(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)), "n": int(arr.size)}


def main() -> None:
    args = parse_args()
    metrics_files = sorted(args.runs_dir.glob("*/final_metrics.json"))
    if not metrics_files:
        raise FileNotFoundError(f"No final_metrics.json files found under {args.runs_dir}")

    split_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    per_cat_values: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for mf in metrics_files:
        run_name = mf.parent.name
        attn = "unknown"
        if "_hypergraph_cuda_" in run_name or run_name.endswith("hypergraph_cuda"):
            attn = "hypergraph_cuda"
        elif "_graph_flash_" in run_name or run_name.endswith("graph_flash"):
            attn = "graph_flash"
        elif "hypergraph_cuda" in run_name:
            attn = "hypergraph_cuda"
        elif "graph_flash" in run_name:
            attn = "graph_flash"

        payload = json.loads(mf.read_text(encoding="utf-8"))
        for split in ("dev", "test", "gen"):
            if split not in payload:
                continue
            split_values[attn][split].append(float(payload[split]["exact_match"]))
            for cat, cat_res in payload[split].get("per_category", {}).items():
                per_cat_values[attn][split][cat].append(float(cat_res["exact_match"]))

    summary = {"splits": {}, "per_category": {}}
    for attn, split_dict in split_values.items():
        summary["splits"][attn] = {split: _mean_std(vals) for split, vals in split_dict.items()}

    for attn, split_dict in per_cat_values.items():
        summary["per_category"][attn] = {}
        for split, cat_dict in split_dict.items():
            summary["per_category"][attn][split] = {
                cat: _mean_std(vals) for cat, vals in sorted(cat_dict.items())
            }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

