#!/usr/bin/env python3
"""
Benchmark CUDA kernels vs PyTorch reference across optimization iterations.

Tracks performance over time in benchmark_history.jsonl, showing deltas
between runs so you can see the impact of each kernel change.

Usage:
    python benchmarks/bench_regression.py                              # Standard benchmark
    python benchmarks/bench_regression.py --quick                      # Fast smoke test
    python benchmarks/bench_regression.py --save --note "description"  # Save with note
    python benchmarks/bench_regression.py --forward-only               # Forward pass only
    python benchmarks/bench_regression.py --roofline                   # Roofline analysis
    python benchmarks/bench_regression.py --show-history               # Show saved history
"""

import os
import sys
import json
import argparse
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from benchmarks._bench_utils import (
    create_inputs, benchmark_fn, get_gpu_specs,
    calc_forward_flops, calc_backward_flops, estimate_forward_bytes,
)

RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_history.jsonl")


# --- Configuration ---

@dataclass
class BenchConfig:
    name: str
    B: int
    H: int
    N: int  # I = J = K = N
    D: int

QUICK_CONFIGS = [
    BenchConfig("tiny",    B=1, H=2, N=32,  D=32),
    BenchConfig("small",   B=1, H=2, N=64,  D=32),
]

STANDARD_CONFIGS = [
    BenchConfig("N32_D32",   B=1, H=2, N=32,  D=32),
    BenchConfig("N64_D32",   B=1, H=2, N=64,  D=32),
    BenchConfig("N64_D64",   B=1, H=2, N=64,  D=64),
    BenchConfig("N128_D32",  B=1, H=2, N=128, D=32),
    BenchConfig("N128_D64",  B=1, H=2, N=128, D=64),
]

LARGE_CONFIGS = [
    BenchConfig("N192_D32",  B=1, H=2, N=192, D=32),
    BenchConfig("N256_D32",  B=1, H=2, N=256, D=32),
    BenchConfig("N256_D64",  B=1, H=1, N=256, D=64),
]


# --- Benchmark Runners ---

def benchmark_forward(cuda_ext, ref_ext, config: BenchConfig, warmup=5, iters=20):
    """Benchmark forward pass: CUDA vs reference."""
    B, H, N, D = config.B, config.H, config.N, config.D
    inputs = create_inputs(B, H, N, D)

    cuda_ms, cuda_times = benchmark_fn(lambda: cuda_ext.forward(*inputs, 0.0), warmup=warmup, iters=iters)
    ref_ms, ref_times = benchmark_fn(lambda: ref_ext.forward(*inputs, 0.0), warmup=warmup, iters=iters)

    return {
        "cuda_median_ms": round(cuda_ms, 3),
        "ref_median_ms": round(ref_ms, 3),
        "ratio": round(cuda_ms / ref_ms, 2) if ref_ms > 0 else float("inf"),
        "cuda_min_ms": round(min(cuda_times), 3),
        "cuda_max_ms": round(max(cuda_times), 3),
    }


def benchmark_backward(cuda_ext, ref_ext, config: BenchConfig, warmup=5, iters=20):
    """Benchmark backward pass: CUDA vs reference (autograd)."""
    B, H, N, D = config.B, config.H, config.N, config.D
    inputs = create_inputs(B, H, N, D)
    grad_output = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)

    fwd_out = cuda_ext.forward(*inputs, 0.0)
    m_i, l_i, m_j, l_j, m_k, l_k = fwd_out[6:12]

    def run_cuda_backward():
        return cuda_ext.backward(
            grad_output, *inputs,
            m_i, l_i, m_j, l_j, m_k, l_k, 0.0
        )

    def run_ref_backward():
        ref_inputs = [t.detach().clone().requires_grad_(True) for t in inputs]
        ref_out = ref_ext.forward(*ref_inputs, 0.0)
        total = sum(o.sum() for o in ref_out)
        total.backward()

    cuda_ms, cuda_times = benchmark_fn(run_cuda_backward, warmup=warmup, iters=iters)
    ref_ms, ref_times = benchmark_fn(run_ref_backward, warmup=warmup, iters=iters)

    return {
        "cuda_median_ms": round(cuda_ms, 3),
        "ref_median_ms": round(ref_ms, 3),
        "ratio": round(cuda_ms / ref_ms, 2) if ref_ms > 0 else float("inf"),
        "cuda_min_ms": round(min(cuda_times), 3),
        "cuda_max_ms": round(max(cuda_times), 3),
    }


# --- Roofline Analysis ---

def roofline_analysis(config: BenchConfig, actual_fwd_ms: float, actual_bwd_ms: float,
                      gpu_specs: Dict) -> Dict:
    B, H, N, D = config.B, config.H, config.N, config.D

    fwd_flops = calc_forward_flops(B, H, N, D)
    bwd_flops = calc_backward_flops(B, H, N, D)
    fwd_bytes = estimate_forward_bytes(B, H, N, D)

    peak_flops = gpu_specs["fp32_tflops"] * 1e12
    peak_bw = gpu_specs["mem_bw_gbs"] * 1e9

    fwd_compute_min_s = fwd_flops / peak_flops
    fwd_memory_min_s = fwd_bytes["total_bytes"] / peak_bw
    fwd_roofline_ms = max(fwd_compute_min_s, fwd_memory_min_s) * 1000

    bwd_compute_min_s = bwd_flops / peak_flops
    bwd_roofline_ms = bwd_compute_min_s * 1000

    return {
        "forward": {
            "total_gflops": fwd_flops / 1e9,
            "roofline_ms": round(fwd_roofline_ms, 4),
            "actual_ms": round(actual_fwd_ms, 3),
            "efficiency_pct": round(fwd_roofline_ms / max(actual_fwd_ms, 0.001) * 100, 1),
        },
        "backward": {
            "total_gflops": bwd_flops / 1e9,
            "roofline_ms": round(bwd_roofline_ms, 4),
            "actual_ms": round(actual_bwd_ms, 3),
            "efficiency_pct": round(bwd_roofline_ms / max(actual_bwd_ms, 0.001) * 100, 1),
        },
    }


# --- Results Tracking ---

def save_results(results: Dict, filepath: str = RESULTS_FILE):
    """Append benchmark results to the JSONL tracking file."""
    with open(filepath, "a") as f:
        f.write(json.dumps(results, default=str) + "\n")


def load_history(filepath: str = RESULTS_FILE) -> List[Dict]:
    """Load benchmark history from JSONL file."""
    if not os.path.exists(filepath):
        return []
    records = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# --- Printing ---

def print_header(gpu_specs: Dict, note: str = None):
    gpu_name = gpu_specs.get('name', 'Unknown GPU')
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"\n{'═'*80}")
    print(f"  BENCHMARK  |  {gpu_name}  |  {ts}")
    if note:
        print(f"  Note: {note}")
    print(f"{'═'*80}")


def print_timing_table(forward_results: Dict, backward_results: Dict, prev_run: Dict = None):
    """Print timing table with optional delta from previous run."""
    print(f"\n  {'Config':<12} │ {'fwd CUDA':>10} │ {'fwd Torch':>10} │ {'fwd Δ':>8} │ {'bwd CUDA':>10} │ {'bwd Torch':>10} │ {'bwd Δ':>8}")
    print(f"  {'─'*12}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*8}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*8}")

    all_configs = set(forward_results.keys()) | set(backward_results.keys())
    for cfg_name in sorted(all_configs):
        fwd = forward_results.get(cfg_name, {})
        bwd = backward_results.get(cfg_name, {})

        if fwd and "cuda_median_ms" in fwd:
            fwd_cuda = f"{fwd['cuda_median_ms']:.2f}ms"
            fwd_ref = f"{fwd['ref_median_ms']:.2f}ms"
            if prev_run:
                prev_fwd = prev_run.get("configs", {}).get(cfg_name, {}).get("forward", {})
                if prev_fwd and "cuda_median_ms" in prev_fwd:
                    delta = (fwd['cuda_median_ms'] - prev_fwd['cuda_median_ms']) / prev_fwd['cuda_median_ms'] * 100
                    if delta < -2:
                        fwd_delta = f"{delta:+.0f}% ↓"
                    elif delta > 2:
                        fwd_delta = f"{delta:+.0f}% ↑"
                    else:
                        fwd_delta = f"{delta:+.0f}%"
                else:
                    fwd_delta = "new"
            else:
                fwd_delta = "—"
        else:
            fwd_cuda, fwd_ref, fwd_delta = "—", "—", "—"

        if bwd and isinstance(bwd, dict) and "cuda_median_ms" in bwd:
            bwd_cuda = f"{bwd['cuda_median_ms']:.2f}ms"
            bwd_ref = f"{bwd['ref_median_ms']:.2f}ms"
            if prev_run:
                prev_bwd = prev_run.get("configs", {}).get(cfg_name, {}).get("backward", {})
                if prev_bwd and isinstance(prev_bwd, dict) and "cuda_median_ms" in prev_bwd:
                    delta = (bwd['cuda_median_ms'] - prev_bwd['cuda_median_ms']) / prev_bwd['cuda_median_ms'] * 100
                    if delta < -2:
                        bwd_delta = f"{delta:+.0f}% ↓"
                    elif delta > 2:
                        bwd_delta = f"{delta:+.0f}% ↑"
                    else:
                        bwd_delta = f"{delta:+.0f}%"
                else:
                    bwd_delta = "new"
            else:
                bwd_delta = "—"
        else:
            bwd_cuda, bwd_ref, bwd_delta = "—", "—", "—"

        print(f"  {cfg_name:<12} │ {fwd_cuda:>10} │ {fwd_ref:>10} │ {fwd_delta:>8} │ {bwd_cuda:>10} │ {bwd_ref:>10} │ {bwd_delta:>8}")


def print_history_compact(history: List[Dict], max_runs: int = None):
    """Print one line per run with key metrics and deltas."""
    if not history:
        print("\n  No history yet.")
        return

    runs = history[-max_runs:] if max_runs else history

    torch_fwd = None
    torch_bwd = None
    for run in reversed(runs):
        configs = run.get("configs", {})
        for cfg_name in ["N128_D64", "N128_D32", "N64_D64", "N64_D32", "N32_D32"]:
            if cfg_name in configs:
                fwd_data = configs[cfg_name].get("forward", {})
                bwd_data = configs[cfg_name].get("backward", {})
                if fwd_data and "ref_median_ms" in fwd_data:
                    torch_fwd = fwd_data['ref_median_ms']
                if bwd_data and isinstance(bwd_data, dict) and "ref_median_ms" in bwd_data:
                    torch_bwd = bwd_data['ref_median_ms']
                break
        if torch_fwd is not None:
            break

    print(f"\n  HISTORY (last {len(runs)} runs)")
    torch_str = f"Torch: fwd={torch_fwd:.1f}ms bwd={torch_bwd:.1f}ms" if torch_fwd else ""
    print(f"  {torch_str}")
    print(f"  {'─'*100}")

    prev_fwd = None
    prev_bwd = None

    for i, run in enumerate(runs):
        ts = run.get("timestamp", "")[:16].replace("T", " ")
        note = run.get("note", "")

        configs = run.get("configs", {})
        fwd_val = None
        bwd_val = None

        for cfg_name in ["N128_D64", "N128_D32", "N64_D64", "N64_D32", "N32_D32"]:
            if cfg_name in configs:
                fwd_data = configs[cfg_name].get("forward", {})
                bwd_data = configs[cfg_name].get("backward", {})
                if fwd_data and "cuda_median_ms" in fwd_data:
                    fwd_val = fwd_data['cuda_median_ms']
                if bwd_data and isinstance(bwd_data, dict) and "cuda_median_ms" in bwd_data:
                    bwd_val = bwd_data['cuda_median_ms']
                break

        if fwd_val is not None:
            fwd_str = f"{fwd_val:.1f}ms"
            if prev_fwd is not None:
                delta = (fwd_val - prev_fwd) / prev_fwd * 100
                if delta < -2:
                    fwd_str += f" ↓{abs(delta):.0f}%"
                elif delta > 2:
                    fwd_str += f" ↑{delta:.0f}%"
            prev_fwd = fwd_val
        else:
            fwd_str = "—"

        if bwd_val is not None:
            bwd_str = f"{bwd_val:.1f}ms"
            if prev_bwd is not None:
                delta = (bwd_val - prev_bwd) / prev_bwd * 100
                if delta < -2:
                    bwd_str += f" ↓{abs(delta):.0f}%"
                elif delta > 2:
                    bwd_str += f" ↑{delta:.0f}%"
            prev_bwd = bwd_val
        else:
            bwd_str = "—"

        note_str = f"  {note}" if note else ""
        print(f"  {i+1:2}. {ts}  fwd={fwd_str:<14} bwd={bwd_str:<14}{note_str}")


def print_roofline_table(roofline_results: Dict[str, Dict]):
    print(f"\n  ROOFLINE ANALYSIS")
    print(f"  {'─'*70}")
    print(f"  {'Config':<12} │ {'Pass':<4} │ {'GFLOPs':>8} │ {'Roofline':>10} │ {'Actual':>10} │ {'Eff':>6}")
    print(f"  {'─'*12}─┼─{'─'*4}─┼─{'─'*8}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*6}")

    for name, data in roofline_results.items():
        for pass_name, short in [("forward", "fwd"), ("backward", "bwd")]:
            p = data.get(pass_name)
            if p is None:
                continue
            print(f"  {name:<12} │ {short:<4} │ {p['total_gflops']:>7.1f} │ {p['roofline_ms']:>8.4f}ms │ {p['actual_ms']:>8.2f}ms │ {p['efficiency_pct']:>5.1f}%")


def print_complexity_summary():
    print(f"""
  ALGORITHMIC COMPLEXITY (per batch-head, I=J=K=N)
  {'─'*60}
  Forward:  6 fused gather/scatter kernels, each O(N³·D)
  Backward: V-grad, jacobian, and Q/R/S-grad kernels, each O(N³·D)

  Torch reference: Same O(N³·D) but uses cuBLAS/einsum
""")


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Benchmark hypergraph attention kernels")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test (2 configs)")
    parser.add_argument("--large", action="store_true", help="Include large N configs")
    parser.add_argument("--forward-only", action="store_true", help="Only benchmark forward")
    parser.add_argument("--backward-only", action="store_true", help="Only benchmark backward")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=20, help="Benchmark iterations")
    parser.add_argument("--save", action="store_true", help="Save results (requires --note)")
    parser.add_argument("--note", type=str, default="", help="Note for this run (required with --save)")
    parser.add_argument("--roofline", action="store_true", help="Show roofline analysis")
    parser.add_argument("--complexity", action="store_true", help="Show algorithmic complexity")
    parser.add_argument("--show-history", action="store_true", help="Show history only")
    parser.add_argument("--max-n-backward", type=int, default=128, help="Max N for backward")
    args = parser.parse_args()

    if args.save and not args.note:
        print("ERROR: --save requires --note 'description of changes'")
        sys.exit(1)

    if args.show_history:
        history = load_history()
        print_history_compact(history)
        return

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available.")
        sys.exit(1)

    try:
        import att3ntion._cuda_kernels as cuda_ext
    except ImportError as e:
        print(f"ERROR: Could not import CUDA extension: {e}")
        sys.exit(1)

    try:
        import att3ntion._torch_kernels as ref_ext
    except ImportError as e:
        print(f"ERROR: Could not import reference extension: {e}")
        sys.exit(1)

    if args.quick:
        configs = QUICK_CONFIGS
    else:
        configs = list(STANDARD_CONFIGS)
        if args.large:
            configs.extend(LARGE_CONFIGS)

    gpu_specs = get_gpu_specs()
    print_header(gpu_specs, args.note if args.note else None)

    if args.complexity:
        print_complexity_summary()

    do_forward = not args.backward_only
    do_backward = not args.forward_only

    forward_results = {}
    backward_results = {}
    roofline_results = {}

    print(f"\n  Running benchmarks...", end="", flush=True)

    for cfg in configs:
        try:
            torch.cuda.empty_cache()
            fwd_data = None
            bwd_data = None

            if do_forward:
                fwd_data = benchmark_forward(cuda_ext, ref_ext, cfg, warmup=args.warmup, iters=args.iters)
                forward_results[cfg.name] = fwd_data

            if do_backward and cfg.N <= args.max_n_backward:
                bwd_data = benchmark_backward(cuda_ext, ref_ext, cfg, warmup=args.warmup, iters=args.iters)
                backward_results[cfg.name] = bwd_data

            if args.roofline and gpu_specs and fwd_data:
                actual_fwd = fwd_data["cuda_median_ms"]
                actual_bwd = bwd_data["cuda_median_ms"] if bwd_data else 0
                roofline_results[cfg.name] = roofline_analysis(cfg, actual_fwd, actual_bwd, gpu_specs)

            print(".", end="", flush=True)

        except torch.cuda.OutOfMemoryError:
            print("OOM", end="", flush=True)
            forward_results[cfg.name] = {"error": "OOM"}
            backward_results[cfg.name] = {"error": "OOM"}
            torch.cuda.empty_cache()
        except Exception as e:
            print("E", end="", flush=True)
            forward_results[cfg.name] = {"error": str(e)[:30]}

    print(" done")

    history = load_history()
    prev_run = history[-1] if history else None

    print_timing_table(forward_results, backward_results, prev_run)

    if args.roofline and roofline_results:
        print_roofline_table(roofline_results)

    if args.save:
        run_data = {
            "timestamp": datetime.now().isoformat(),
            "gpu": gpu_specs.get("name", "unknown"),
            "note": args.note,
            "configs": {},
        }
        for cfg in configs:
            run_data["configs"][cfg.name] = {
                "B": cfg.B, "H": cfg.H, "N": cfg.N, "D": cfg.D,
                "forward": forward_results.get(cfg.name),
                "backward": backward_results.get(cfg.name),
            }
        save_results(run_data)
        print(f"\n  ✓ Saved to {RESULTS_FILE}")

    history = load_history()
    print_history_compact(history)

    print(f"\n{'═'*80}\n")


if __name__ == "__main__":
    main()
