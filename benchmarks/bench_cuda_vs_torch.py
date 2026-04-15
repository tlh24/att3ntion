#!/usr/bin/env python3
"""
CUDA vs Torch scaling benchmark.

Compares time and memory of the CUDA kernels against the PyTorch reference
across a sweep of sequence lengths N. Handles Torch OOM gracefully,
continuing CUDA-only measurements at large N.

Usage:
    python benchmarks/bench_cuda_vs_torch.py
    python benchmarks/bench_cuda_vs_torch.py --n-values 32,64,128,256,512
    python benchmarks/bench_cuda_vs_torch.py --no-backward
    python benchmarks/bench_cuda_vs_torch.py --save results.json
"""

import os
import sys
import json
import argparse
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from benchmarks._bench_utils import create_inputs, benchmark_fn


@dataclass
class ScalingResult:
    N: int
    cuda_fwd_ms: Optional[float]
    torch_fwd_ms: Optional[float]
    cuda_bwd_ms: Optional[float]
    torch_bwd_ms: Optional[float]
    cuda_mem_mb: Optional[float]
    torch_mem_mb: Optional[float]
    fwd_speedup: Optional[float]
    bwd_speedup: Optional[float]
    mem_ratio: Optional[float]
    torch_oom: bool = False
    cuda_oom: bool = False


# --- Memory ---

def measure_fwd_memory(ext, inputs):
    """Measure peak memory for forward pass in MB."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    _ = ext.forward(*inputs, 0.0)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


# --- Main Benchmark Loop ---

def run_scaling_benchmark(N_values: List[int], B=1, H=2, D=32,
                          include_backward=True, max_n_backward=1024):
    """Run scaling benchmark. Handles Torch OOM separately from CUDA OOM."""
    import att3ntion._cuda_kernels as cuda_ext
    import att3ntion._torch_kernels as ref_ext

    results = []
    torch_oom_started = False

    for N in N_values:
        print(f"  N={N}...", end="", flush=True)
        torch.cuda.empty_cache()

        cuda_fwd, torch_fwd = None, None
        cuda_bwd, torch_bwd = None, None
        cuda_mem, torch_mem = None, None
        torch_oom = torch_oom_started
        cuda_oom = False

        try:
            inputs = create_inputs(B, H, N, D)

            # CUDA forward + memory
            try:
                cuda_fwd = benchmark_fn(lambda: cuda_ext.forward(*inputs, 0.0))[0]
                torch.cuda.empty_cache()
                cuda_mem = measure_fwd_memory(cuda_ext, inputs)
            except torch.cuda.OutOfMemoryError:
                cuda_oom = True
                print(" CUDA OOM")
                torch.cuda.empty_cache()
                break

            # Torch forward + memory (may OOM at large N)
            if not torch_oom_started:
                try:
                    torch.cuda.empty_cache()
                    torch_fwd = benchmark_fn(lambda: ref_ext.forward(*inputs, 0.0))[0]
                    torch.cuda.empty_cache()
                    torch_mem = measure_fwd_memory(ref_ext, inputs)
                except torch.cuda.OutOfMemoryError:
                    torch_oom = True
                    torch_oom_started = True
                    torch.cuda.empty_cache()

            # Backward
            if include_backward and N <= max_n_backward:
                if cuda_fwd is not None:
                    try:
                        torch.cuda.empty_cache()
                        fwd_out = cuda_ext.forward(*inputs, 0.0)
                        m_i, l_i, m_j, l_j, m_k, l_k = fwd_out[6:12]
                        grad_output = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)

                        def run_cuda_bwd():
                            cuda_ext.backward(grad_output, *inputs,
                                              m_i, l_i, m_j, l_j, m_k, l_k, 0.0)

                        cuda_bwd = benchmark_fn(run_cuda_bwd, warmup=3, iters=10)[0]
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()

                if torch_fwd is not None and not torch_oom:
                    try:
                        torch.cuda.empty_cache()

                        def run_ref_bwd():
                            ref_inputs = [t.detach().clone().requires_grad_(True) for t in inputs]
                            out = ref_ext.forward(*ref_inputs, 0.0)
                            total = sum(o.sum() for o in out)
                            total.backward()

                        torch_bwd = benchmark_fn(run_ref_bwd, warmup=3, iters=10)[0]
                    except torch.cuda.OutOfMemoryError:
                        torch_oom = True
                        torch_oom_started = True
                        torch.cuda.empty_cache()

            fwd_speedup = round(torch_fwd / cuda_fwd, 2) if (cuda_fwd and torch_fwd) else None
            bwd_speedup = round(torch_bwd / cuda_bwd, 2) if (cuda_bwd and torch_bwd) else None
            mem_ratio = round(torch_mem / cuda_mem, 2) if (cuda_mem and torch_mem) else None

            result = ScalingResult(
                N=N,
                cuda_fwd_ms=round(cuda_fwd, 3) if cuda_fwd else None,
                torch_fwd_ms=round(torch_fwd, 3) if torch_fwd else None,
                cuda_bwd_ms=round(cuda_bwd, 3) if cuda_bwd else None,
                torch_bwd_ms=round(torch_bwd, 3) if torch_bwd else None,
                cuda_mem_mb=round(cuda_mem, 1) if cuda_mem else None,
                torch_mem_mb=round(torch_mem, 1) if torch_mem else None,
                fwd_speedup=fwd_speedup,
                bwd_speedup=bwd_speedup,
                mem_ratio=mem_ratio,
                torch_oom=torch_oom,
                cuda_oom=cuda_oom,
            )
            results.append(result)

            if torch_oom:
                print(" ✓ (Torch OOM)")
            else:
                print(" ✓")

        except torch.cuda.OutOfMemoryError:
            print(" OOM (input allocation)")
            break
        except Exception as e:
            print(f" Error: {e}")
            break

    return results


# --- Output ---

def _fmt_speedup(speedup: float) -> str:
    s = f"{speedup:.2f}x"
    if speedup > 1.05:
        return s + " ✓"
    elif speedup < 0.95:
        return s + " ✗"
    return s


def print_results_table(results: List[ScalingResult]):
    """Print timing and memory tables."""
    print("\n" + "="*80)
    print("SCALING BENCHMARK: CUDA Kernels vs PyTorch Reference")
    print("="*80)
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # --- Timing ---
    print("\n" + "─"*80)
    print("  TIMING COMPARISON")
    print("─"*80)

    backward_results = [r for r in results if r.cuda_bwd_ms is not None]
    has_backward = len(backward_results) > 0

    if has_backward:
        print(f"\n{'N':>6} │ {'CUDA fwd':>10} │ {'Torch fwd':>10} │ {'fwd Δ':>10} │ {'CUDA bwd':>10} │ {'Torch bwd':>10} │ {'bwd Δ':>10} │ {'Combined':>10}")
        print("─"*6 + "─┼─" + "─"*10 + "─┼─" + "─"*10 + "─┼─" + "─"*10 + "─┼─" + "─"*10 + "─┼─" + "─"*10 + "─┼─" + "─"*10 + "─┼─" + "─"*10)

        for r in results:
            cuda_fwd_str = f"{r.cuda_fwd_ms:.2f}ms" if r.cuda_fwd_ms else "—"

            if r.torch_oom:
                torch_fwd_str = "OOM"
                fwd_speedup_str = "N/A"
            elif r.torch_fwd_ms:
                torch_fwd_str = f"{r.torch_fwd_ms:.2f}ms"
                fwd_speedup_str = _fmt_speedup(r.fwd_speedup)
            else:
                torch_fwd_str = "—"
                fwd_speedup_str = "—"

            if r.cuda_bwd_ms is not None and r.torch_bwd_ms is not None:
                cuda_bwd_str = f"{r.cuda_bwd_ms:.2f}ms"
                torch_bwd_str = f"{r.torch_bwd_ms:.2f}ms"
                bwd_speedup_str = _fmt_speedup(r.bwd_speedup)
                combined_speedup = (r.torch_fwd_ms + r.torch_bwd_ms) / (r.cuda_fwd_ms + r.cuda_bwd_ms)
                combined_str = _fmt_speedup(combined_speedup)
            elif r.cuda_bwd_ms is not None and r.torch_oom:
                cuda_bwd_str = f"{r.cuda_bwd_ms:.2f}ms"
                torch_bwd_str = "OOM"
                bwd_speedup_str = "N/A"
                combined_str = "N/A"
            else:
                cuda_bwd_str = f"{r.cuda_bwd_ms:.2f}ms" if r.cuda_bwd_ms else "—"
                torch_bwd_str = "OOM" if r.torch_oom else "—"
                bwd_speedup_str = "N/A" if r.torch_oom else "—"
                combined_str = "N/A" if r.torch_oom else "—"

            print(f"{r.N:>6} │ {cuda_fwd_str:>10} │ {torch_fwd_str:>10} │ {fwd_speedup_str:>10} │ {cuda_bwd_str:>10} │ {torch_bwd_str:>10} │ {bwd_speedup_str:>10} │ {combined_str:>10}")
    else:
        print(f"\n{'N':>6} │ {'CUDA fwd':>10} │ {'Torch fwd':>10} │ {'Speedup':>10}")
        print("─"*6 + "─┼─" + "─"*10 + "─┼─" + "─"*10 + "─┼─" + "─"*10)

        for r in results:
            cuda_fwd_str = f"{r.cuda_fwd_ms:.2f}ms" if r.cuda_fwd_ms else "—"
            if r.torch_oom:
                torch_fwd_str = "OOM"
                fwd_speedup_str = "N/A"
            elif r.torch_fwd_ms:
                torch_fwd_str = f"{r.torch_fwd_ms:.2f}ms"
                fwd_speedup_str = _fmt_speedup(r.fwd_speedup)
            else:
                torch_fwd_str = "—"
                fwd_speedup_str = "—"
            print(f"{r.N:>6} │ {cuda_fwd_str:>10} │ {torch_fwd_str:>10} │ {fwd_speedup_str:>10}")

    # --- Memory ---
    print("\n" + "─"*80)
    print("  MEMORY COMPARISON")
    print("─"*80)

    print(f"\n{'N':>6} │ {'CUDA mem':>12} │ {'Torch mem':>12} │ {'Difference':>12} │ {'Ratio':>10}")
    print("─"*6 + "─┼─" + "─"*12 + "─┼─" + "─"*12 + "─┼─" + "─"*12 + "─┼─" + "─"*10)

    for r in results:
        cuda_mem_str = f"{r.cuda_mem_mb:.1f}MB" if r.cuda_mem_mb else "—"
        if r.torch_oom:
            torch_mem_str = "OOM"
            diff_str = "N/A"
            ratio_str = "N/A"
        elif r.torch_mem_mb:
            torch_mem_str = f"{r.torch_mem_mb:.1f}MB"
            mem_diff = r.torch_mem_mb - r.cuda_mem_mb
            diff_str = f"{mem_diff:+.1f}MB"
            ratio_str = _fmt_speedup(r.mem_ratio)
        else:
            torch_mem_str = "—"
            diff_str = "—"
            ratio_str = "—"
        print(f"{r.N:>6} │ {cuda_mem_str:>12} │ {torch_mem_str:>12} │ {diff_str:>12} │ {ratio_str:>10}")

    # --- Summary ---
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    valid_fwd = [r for r in results if r.fwd_speedup is not None]
    torch_oom_count = sum(1 for r in results if r.torch_oom)

    print(f"\n  Timing:")
    if valid_fwd:
        avg_fwd = sum(r.fwd_speedup for r in valid_fwd) / len(valid_fwd)
        faster_fwd = sum(1 for r in valid_fwd if r.fwd_speedup > 1.0)
        print(f"    Forward:  avg {avg_fwd:.2f}x speedup, CUDA faster in {faster_fwd}/{len(valid_fwd)} configs")

    valid_bwd = [r for r in backward_results if r.bwd_speedup is not None]
    if valid_bwd:
        avg_bwd = sum(r.bwd_speedup for r in valid_bwd) / len(valid_bwd)
        faster_bwd = sum(1 for r in valid_bwd if r.bwd_speedup > 1.0)
        print(f"    Backward: avg {avg_bwd:.2f}x speedup, CUDA faster in {faster_bwd}/{len(valid_bwd)} configs")

        combined = [(r.torch_fwd_ms + r.torch_bwd_ms) / (r.cuda_fwd_ms + r.cuda_bwd_ms) for r in valid_bwd]
        avg_combined = sum(combined) / len(combined)
        faster_combined = sum(1 for s in combined if s > 1.0)
        print(f"    Combined: avg {avg_combined:.2f}x speedup, CUDA faster in {faster_combined}/{len(valid_bwd)} configs")

    if torch_oom_count > 0:
        print(f"    Torch OOM: {torch_oom_count} configs (CUDA kept running)")

    valid_mem = [r for r in results if r.mem_ratio is not None]
    print(f"\n  Memory:")
    if valid_mem:
        avg_mem = sum(r.mem_ratio for r in valid_mem) / len(valid_mem)
        less_mem = sum(1 for r in valid_mem if r.mem_ratio > 1.0)
        total_cuda = sum(r.cuda_mem_mb for r in valid_mem)
        total_torch = sum(r.torch_mem_mb for r in valid_mem)
        print(f"    Avg ratio: {avg_mem:.2f}x (Torch/CUDA)")
        print(f"    CUDA uses less memory in {less_mem}/{len(valid_mem)} configs")
        print(f"    Total: CUDA {total_cuda:.1f}MB vs Torch {total_torch:.1f}MB")
    if torch_oom_count > 0:
        cuda_only = [r for r in results if r.torch_oom and r.cuda_fwd_ms]
        if cuda_only:
            max_cuda_n = max(r.N for r in cuda_only)
            torch_oom_n = min(r.N for r in results if r.torch_oom)
            print(f"    Torch OOM at N={torch_oom_n}, CUDA still running at N={max_cuda_n}")

    fwd_crossover = next((r.N for r in results if r.fwd_speedup is not None and r.fwd_speedup > 1), None)
    if fwd_crossover:
        print(f"\n  Crossover (CUDA faster): N >= {fwd_crossover}")

    print()


def save_results_json(results: List[ScalingResult], filepath: str):
    data = {
        "timestamp": datetime.now().isoformat(),
        "gpu": torch.cuda.get_device_name(0),
        "results": [vars(r) for r in results]
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n✓ Saved to {filepath}")


def main():
    parser = argparse.ArgumentParser(description="CUDA vs Torch scaling benchmark")
    parser.add_argument("--n-values", type=str, default="32,64,128,256,384,512,768,1024",
                        help="Comma-separated N values to test")
    parser.add_argument("--no-backward", action="store_true", help="Skip backward benchmarks")
    parser.add_argument("--max-n-backward", type=int, default=1024,
                        help="Max N for backward benchmarks")
    parser.add_argument("--save", type=str, help="Save results to JSON file")
    parser.add_argument("-B", type=int, default=1, help="Batch size")
    parser.add_argument("-H", type=int, default=2, help="Number of heads")
    parser.add_argument("-D", type=int, default=32, help="Head dimension")
    args = parser.parse_args()

    N_values = [int(n) for n in args.n_values.split(",")]

    print("\n" + "="*60)
    print("CUDA vs Torch Scaling Benchmark")
    print("="*60)
    print(f"Config: B={args.B}, H={args.H}, D={args.D}")
    print(f"N values: {N_values}")

    results = run_scaling_benchmark(
        N_values, B=args.B, H=args.H, D=args.D,
        include_backward=not args.no_backward,
        max_n_backward=args.max_n_backward
    )

    print_results_table(results)

    if args.save:
        save_results_json(results, args.save)


if __name__ == "__main__":
    main()
