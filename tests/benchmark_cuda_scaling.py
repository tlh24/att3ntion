#!/usr/bin/env python3
"""
CUDA kernel scaling analysis: time, memory & FLOP/s vs sequence length N.

Pure CUDA analysis — no Torch comparison. Measures how the kernel's
performance scales with increasing N and estimates empirical complexity.

Usage:
    python tests/benchmark_cuda_scaling.py
    python tests/benchmark_cuda_scaling.py --n-values 32,64,128,256,512
    python tests/benchmark_cuda_scaling.py --no-backward
    python tests/benchmark_cuda_scaling.py --save results.json
"""

import os
import sys
import json
import argparse
import math
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# --- GPU Specs ---

def get_gpu_specs() -> Dict:
    """Get GPU specs for utilization analysis. Falls back to estimation."""
    props = torch.cuda.get_device_properties(0)
    name = props.name

    known_gpus = {
        "NVIDIA GeForce RTX 4080 Laptop GPU": {"fp32_tflops": 33.9,  "mem_bw_gbs": 432.0},
        "NVIDIA GeForce RTX 4080":            {"fp32_tflops": 48.7,  "mem_bw_gbs": 716.8},
        "NVIDIA GeForce RTX 4080 SUPER":      {"fp32_tflops": 52.0,  "mem_bw_gbs": 736.3},
        "NVIDIA GeForce RTX 4090":            {"fp32_tflops": 82.6,  "mem_bw_gbs": 1008.0},
        "NVIDIA GeForce RTX 3090":            {"fp32_tflops": 35.6,  "mem_bw_gbs": 936.2},
        "NVIDIA A100-SXM4-40GB":              {"fp32_tflops": 19.5,  "mem_bw_gbs": 1555.0},
        "NVIDIA A100-SXM4-80GB":              {"fp32_tflops": 19.5,  "mem_bw_gbs": 2039.0},
        "NVIDIA H100":                        {"fp32_tflops": 51.2,  "mem_bw_gbs": 3350.0},
    }

    specs = known_gpus.get(name)
    if specs is None:
        for gpu_name, gpu_specs in known_gpus.items():
            if gpu_name in name or name in gpu_name:
                specs = gpu_specs
                break

    if specs is None:
        clock_ghz = props.clock_rate / 1e6
        cuda_cores = props.multi_processor_count * 128
        fp32_tflops = cuda_cores * 2 * clock_ghz / 1000
        mem_bw_gbs = props.total_memory / 1e9 * 10
        specs = {"fp32_tflops": round(fp32_tflops, 1), "mem_bw_gbs": round(mem_bw_gbs, 1)}

    return {
        "name": name,
        "sm_count": props.multi_processor_count,
        "total_memory_gb": round(props.total_memory / 1e9, 2),
        **specs,
    }


# --- FLOP Estimation ---

def calc_forward_flops(B: int, H: int, N: int, D: int) -> int:
    """Theoretical FLOPs for forward pass.

    3 gather kernels:  each B*H*N³*(4D+3)
    3 scatter kernels: each B*H*N³*(4D+5)
    """
    gather = B * H * N**3 * (4 * D + 3)
    scatter = B * H * N**3 * (4 * D + 5)
    return 3 * gather + 3 * scatter


def calc_backward_flops(B: int, H: int, N: int, D: int) -> int:
    """Theoretical FLOPs for backward pass.

    6 value gradients: 6*B*H*N³*4D
    Jacobian terms:    B*H*N³*24D
    Q/R/S gradients:   3*B*H*N³*24D
    """
    v_grad = 6 * B * H * N**3 * 4 * D
    jacobian = B * H * N**3 * 24 * D
    qrs_grad = 3 * B * H * N**3 * 24 * D
    return v_grad + jacobian + qrs_grad


def format_flops(flops: float) -> str:
    if flops >= 1e12:
        return f"{flops/1e12:.2f}T"
    elif flops >= 1e9:
        return f"{flops/1e9:.2f}G"
    elif flops >= 1e6:
        return f"{flops/1e6:.2f}M"
    return f"{flops:.0f}"


@dataclass
class ScalingPoint:
    N: int
    fwd_ms: Optional[float] = None
    bwd_ms: Optional[float] = None
    total_ms: Optional[float] = None
    mem_mb: Optional[float] = None
    fwd_flops: Optional[int] = None
    bwd_flops: Optional[int] = None
    total_flops: Optional[int] = None
    fwd_tflops: Optional[float] = None
    bwd_tflops: Optional[float] = None
    total_tflops: Optional[float] = None
    fwd_growth: Optional[float] = None
    bwd_growth: Optional[float] = None
    mem_growth: Optional[float] = None
    n_ratio: Optional[float] = None
    oom: bool = False


# --- Benchmarking ---

def create_inputs(B: int, H: int, N: int, D: int):
    torch.manual_seed(42)
    Q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
    R = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
    S = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
    Vq_1 = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
    Vq_2 = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
    Vr_1 = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
    Vr_2 = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
    Vs_1 = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
    Vs_2 = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
    return Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2


def _benchmark_median(fn, warmup, iters):
    """Run fn with warmup, return median time in ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))

    times.sort()
    return times[len(times) // 2]


def benchmark_forward(ext, Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2,
                      warmup=5, iters=20):
    def run():
        return ext.forward(Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2, 0.0)
    return _benchmark_median(run, warmup, iters)


def benchmark_backward(ext, Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2,
                       warmup=3, iters=10):
    B, H, N, D = Q.shape
    grad_output = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)

    fwd_out = ext.forward(Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2, 0.0)
    m_i, l_i, m_j, l_j, m_k, l_k = fwd_out[6:12]

    def run():
        return ext.backward(
            grad_output, Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2,
            m_i, l_i, m_j, l_j, m_k, l_k, 0.0
        )
    return _benchmark_median(run, warmup, iters)


def measure_peak_memory(ext, Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2):
    """Measure peak GPU memory during forward+backward."""
    B, H, N, D = Q.shape
    grad_output = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    fwd_out = ext.forward(Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2, 0.0)
    m_i, l_i, m_j, l_j, m_k, l_k = fwd_out[6:12]

    _ = ext.backward(
        grad_output, Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2,
        m_i, l_i, m_j, l_j, m_k, l_k, 0.0
    )
    torch.cuda.synchronize()

    return torch.cuda.max_memory_allocated() / (1024 ** 2)


# --- Scaling Loop ---

def adaptive_iters(N: int, base_warmup: int, base_iters: int):
    """Scale down warmup/iters for large N to keep total runtime reasonable."""
    if N <= 256:
        return base_warmup, base_iters
    elif N <= 512:
        return max(3, base_warmup), max(10, base_iters // 2)
    elif N <= 1024:
        return 2, max(5, base_iters // 4)
    elif N <= 2048:
        return 2, 5
    else:
        return 1, 3


def run_scaling_analysis(N_values: List[int], B=1, H=2, D=32,
                         include_backward=True, warmup=5, iters=20):
    import hyper_attn_cpp_manual as cuda_ext

    results = []
    prev_point = None

    for N in N_values:
        w, it = adaptive_iters(N, warmup, iters)
        bw, bit = max(1, w // 2), max(3, it // 2)
        print(f"  N={N:>5} (w={w},i={it})...", end="", flush=True)
        torch.cuda.empty_cache()

        point = ScalingPoint(N=N)

        try:
            inputs = create_inputs(B, H, N, D)

            point.fwd_ms = benchmark_forward(cuda_ext, *inputs, warmup=w, iters=it)
            point.fwd_flops = calc_forward_flops(B, H, N, D)
            point.fwd_tflops = point.fwd_flops / (point.fwd_ms / 1000) / 1e12

            if include_backward:
                torch.cuda.empty_cache()
                point.bwd_ms = benchmark_backward(cuda_ext, *inputs, warmup=bw, iters=bit)
                point.bwd_flops = calc_backward_flops(B, H, N, D)
                point.bwd_tflops = point.bwd_flops / (point.bwd_ms / 1000) / 1e12
                point.total_ms = point.fwd_ms + point.bwd_ms
                point.total_flops = point.fwd_flops + point.bwd_flops
                point.total_tflops = point.total_flops / (point.total_ms / 1000) / 1e12

            torch.cuda.empty_cache()
            point.mem_mb = measure_peak_memory(cuda_ext, *inputs)

            if prev_point and prev_point.fwd_ms:
                point.n_ratio = N / prev_point.N
                point.fwd_growth = point.fwd_ms / prev_point.fwd_ms
                if point.bwd_ms and prev_point.bwd_ms:
                    point.bwd_growth = point.bwd_ms / prev_point.bwd_ms
                if point.mem_mb and prev_point.mem_mb:
                    point.mem_growth = point.mem_mb / prev_point.mem_mb

            results.append(point)
            prev_point = point

            tflops_str = f"  {point.fwd_tflops:.2f} TFLOP/s" if point.fwd_tflops else ""
            print(f" ✓  fwd={point.fwd_ms:.2f}ms  mem={point.mem_mb:.1f}MB{tflops_str}")

        except torch.cuda.OutOfMemoryError:
            point.oom = True
            results.append(point)
            print(" OOM")
            break
        except Exception as e:
            print(f" Error: {e}")
            break

    return results


# --- Complexity Estimation ---

def estimate_complexity(results: List[ScalingPoint]):
    """Estimate time and memory complexity exponents from growth rates."""
    valid = [r for r in results if r.fwd_growth is not None and r.n_ratio is not None]
    if len(valid) < 2:
        return None, None, None

    def estimate_k(growth_ratios, n_ratios):
        ks = []
        for g, n in zip(growth_ratios, n_ratios):
            if g > 0 and n > 1:
                ks.append(math.log(g) / math.log(n))
        return sum(ks) / len(ks) if ks else None

    fwd_k = estimate_k([r.fwd_growth for r in valid], [r.n_ratio for r in valid])
    bwd_k = estimate_k(
        [r.bwd_growth for r in valid if r.bwd_growth],
        [r.n_ratio for r in valid if r.bwd_growth]
    )
    mem_k = estimate_k(
        [r.mem_growth for r in valid if r.mem_growth],
        [r.n_ratio for r in valid if r.mem_growth]
    )
    return fwd_k, bwd_k, mem_k


def format_complexity(k: Optional[float]) -> str:
    if k is None:
        return "?"
    if abs(k - 1.0) < 0.15:
        return "O(n)"
    elif abs(k - 1.5) < 0.15:
        return "O(n^1.5)"
    elif abs(k - 2.0) < 0.2:
        return "O(n^2)"
    elif abs(k - 3.0) < 0.25:
        return "O(n^3)"
    return f"O(n^{k:.2f})"


# --- Output ---

def print_results(results: List[ScalingPoint], B: int, H: int, D: int):
    gpu_specs = get_gpu_specs()
    peak_tflops = gpu_specs.get("fp32_tflops", None)

    print("\n" + "=" * 110)
    print("  CUDA KERNEL SCALING ANALYSIS")
    print("=" * 110)
    print(f"\n  GPU:    {gpu_specs['name']}")
    print(f"  VRAM:   {gpu_specs['total_memory_gb']:.1f} GB")
    if peak_tflops:
        print(f"  Peak:   {peak_tflops} TFLOP/s (FP32)")
    print(f"  Config: B={B}, H={H}, D={D}")
    print(f"  Date:   {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    has_backward = any(r.bwd_ms for r in results)

    def fmt_ms(ms):
        if ms is None:
            return "—"
        if ms >= 1000:
            return f"{ms/1000:.2f}s"
        return f"{ms:.2f}ms"

    # --- Time Scaling ---
    print("\n" + "─" * 110)
    print("  TIME SCALING")
    print("─" * 110)

    if has_backward:
        print(f"\n  {'N':>6} │ {'Forward':>11} │ {'fwd Δ':>7} │ {'Backward':>11} │ {'bwd Δ':>7} │ {'Total':>11}")
        print("  " + "─" * 6 + "─┼─" + "─" * 11 + "─┼─" + "─" * 7 + "─┼─" + "─" * 11 + "─┼─" + "─" * 7 + "─┼─" + "─" * 11)
    else:
        print(f"\n  {'N':>6} │ {'Forward':>11} │ {'Growth':>7}")
        print("  " + "─" * 6 + "─┼─" + "─" * 11 + "─┼─" + "─" * 7)

    for r in results:
        if r.oom:
            oom_line = f"  {r.N:>6} │ {'OOM':>11} │ {'':>7}"
            if has_backward:
                oom_line += f" │ {'':>11} │ {'':>7} │ {'':>11}"
            print(oom_line)
            continue

        fwd_str = fmt_ms(r.fwd_ms)
        fwd_g = f"{r.fwd_growth:.2f}x" if r.fwd_growth else "—"

        if has_backward:
            bwd_str = fmt_ms(r.bwd_ms)
            bwd_g = f"{r.bwd_growth:.2f}x" if r.bwd_growth else "—"
            total_str = fmt_ms(r.total_ms)
            print(f"  {r.N:>6} │ {fwd_str:>11} │ {fwd_g:>7} │ {bwd_str:>11} │ {bwd_g:>7} │ {total_str:>11}")
        else:
            print(f"  {r.N:>6} │ {fwd_str:>11} │ {fwd_g:>7}")

    # --- Compute Throughput ---
    print("\n" + "─" * 110)
    print("  COMPUTE THROUGHPUT (TFLOP/s)")
    print("─" * 110)

    util_hdr = " │ GPU util" if peak_tflops else ""
    if has_backward:
        print(f"\n  {'N':>6} │ {'fwd FLOPs':>12} │ {'fwd TF/s':>10} │ {'bwd FLOPs':>12} │ {'bwd TF/s':>10} │ {'total TF/s':>10}{util_hdr}")
        sep = "  " + "─" * 6 + "─┼─" + "─" * 12 + "─┼─" + "─" * 10 + "─┼─" + "─" * 12 + "─┼─" + "─" * 10 + "─┼─" + "─" * 10
        if peak_tflops:
            sep += "─┼─" + "─" * 9
        print(sep)
    else:
        print(f"\n  {'N':>6} │ {'FLOPs':>12} │ {'TFLOP/s':>10}{util_hdr}")
        sep = "  " + "─" * 6 + "─┼─" + "─" * 12 + "─┼─" + "─" * 10
        if peak_tflops:
            sep += "─┼─" + "─" * 9
        print(sep)

    for r in results:
        if r.oom:
            continue

        if has_backward:
            fwd_f = format_flops(r.fwd_flops) if r.fwd_flops else "—"
            fwd_t = f"{r.fwd_tflops:.2f}" if r.fwd_tflops else "—"
            bwd_f = format_flops(r.bwd_flops) if r.bwd_flops else "—"
            bwd_t = f"{r.bwd_tflops:.2f}" if r.bwd_tflops else "—"
            tot_t = f"{r.total_tflops:.2f}" if r.total_tflops else "—"
            line = f"  {r.N:>6} │ {fwd_f:>12} │ {fwd_t:>10} │ {bwd_f:>12} │ {bwd_t:>10} │ {tot_t:>10}"
            if peak_tflops and r.total_tflops:
                util = r.total_tflops / peak_tflops * 100
                line += f" │ {util:>7.1f}%"
            elif peak_tflops:
                line += f" │ {'—':>8}"
            print(line)
        else:
            fwd_f = format_flops(r.fwd_flops) if r.fwd_flops else "—"
            fwd_t = f"{r.fwd_tflops:.2f}" if r.fwd_tflops else "—"
            line = f"  {r.N:>6} │ {fwd_f:>12} │ {fwd_t:>10}"
            if peak_tflops and r.fwd_tflops:
                util = r.fwd_tflops / peak_tflops * 100
                line += f" │ {util:>7.1f}%"
            elif peak_tflops:
                line += f" │ {'—':>8}"
            print(line)

    # --- Memory Scaling ---
    print("\n" + "─" * 110)
    print("  MEMORY SCALING")
    print("─" * 110)

    print(f"\n  {'N':>6} │ {'Peak Memory':>14} │ {'Growth':>8} │ {'MB/token':>10} │ {'Bar':>30}")
    print("  " + "─" * 6 + "─┼─" + "─" * 14 + "─┼─" + "─" * 8 + "─┼─" + "─" * 10 + "─┼─" + "─" * 30)

    max_mem = max((r.mem_mb for r in results if r.mem_mb), default=1)

    for r in results:
        if r.oom:
            print(f"  {r.N:>6} │ {'OOM':>14} │ {'':>8} │ {'':>10} │")
            continue

        mem_str = f"{r.mem_mb:.1f}MB" if r.mem_mb else "—"
        growth_str = f"{r.mem_growth:.2f}x" if r.mem_growth else "—"
        mb_per_n = f"{r.mem_mb / r.N:.4f}" if r.mem_mb else "—"

        bar_len = int(30 * r.mem_mb / max_mem) if r.mem_mb else 0
        bar = "█" * bar_len

        print(f"  {r.N:>6} │ {mem_str:>14} │ {growth_str:>8} │ {mb_per_n:>10} │ {bar}")

    # --- Complexity Estimation ---
    fwd_k, bwd_k, mem_k = estimate_complexity(results)

    fwd_exp = f"{fwd_k:.2f}" if fwd_k else "?"
    bwd_exp = f"{bwd_k:.2f}" if bwd_k else "?"
    mem_exp = f"{mem_k:.2f}" if mem_k else "?"

    print("\n" + "─" * 110)
    print("  COMPLEXITY ESTIMATION")
    print("─" * 110)
    print(f"""
  Estimated from empirical growth rates:

    Forward time:   {format_complexity(fwd_k):>12}  (exponent ~ {fwd_exp})
    Backward time:  {format_complexity(bwd_k):>12}  (exponent ~ {bwd_exp})
    Memory:         {format_complexity(mem_k):>12}  (exponent ~ {mem_exp})

  Theoretical: time = O(n^3*d), memory = O(n*d)  [flash-style tiling]""")

    # --- Summary ---
    valid = [r for r in results if not r.oom]
    if valid:
        min_n, max_n = valid[0].N, valid[-1].N
        min_fwd = min(r.fwd_ms for r in valid if r.fwd_ms)
        max_fwd = max(r.fwd_ms for r in valid if r.fwd_ms)
        min_mem = min(r.mem_mb for r in valid if r.mem_mb)
        max_mem_val = max(r.mem_mb for r in valid if r.mem_mb)

        peak_fwd_tflops = max((r.fwd_tflops for r in valid if r.fwd_tflops), default=0)
        peak_bwd_tflops = max((r.bwd_tflops for r in valid if r.bwd_tflops), default=0)
        peak_total_tflops = max((r.total_tflops for r in valid if r.total_tflops), default=0)

        print("\n" + "─" * 110)
        print("  SUMMARY")
        print("─" * 110)
        print(f"""
  N range tested:     {min_n} -> {max_n}  ({max_n / min_n:.0f}x range)
  Forward time:       {min_fwd:.2f}ms -> {max_fwd:.2f}ms  ({max_fwd / min_fwd:.0f}x increase)
  Memory:             {min_mem:.1f}MB -> {max_mem_val:.1f}MB  ({max_mem_val / min_mem:.0f}x increase)

  Peak forward:       {peak_fwd_tflops:.2f} TFLOP/s""", end="")

        if peak_tflops:
            print(f"  ({peak_fwd_tflops / peak_tflops * 100:.1f}% of GPU peak)")
        else:
            print()

        if peak_bwd_tflops:
            print(f"  Peak backward:      {peak_bwd_tflops:.2f} TFLOP/s", end="")
            if peak_tflops:
                print(f"  ({peak_bwd_tflops / peak_tflops * 100:.1f}% of GPU peak)")
            else:
                print()

        if peak_total_tflops:
            print(f"  Peak combined:      {peak_total_tflops:.2f} TFLOP/s", end="")
            if peak_tflops:
                print(f"  ({peak_total_tflops / peak_tflops * 100:.1f}% of GPU peak)")
            else:
                print()

        if valid[-1].total_ms:
            throughput = max_n / (valid[-1].total_ms / 1000)
            print(f"\n  Throughput at N={max_n}: ~{throughput:.0f} tokens/sec (fwd+bwd)")

        vram_gb = gpu_specs.get("total_memory_gb", 0)
        if vram_gb and max_mem_val:
            used_pct = max_mem_val / (vram_gb * 1024) * 100
            print(f"  VRAM used at N={max_n}: {max_mem_val:.0f}MB / {vram_gb * 1024:.0f}MB ({used_pct:.1f}%)")

    print("\n" + "=" * 110 + "\n")


def save_results_json(results: List[ScalingPoint], filepath: str, B: int, H: int, D: int):
    gpu_specs = get_gpu_specs()
    data = {
        "timestamp": datetime.now().isoformat(),
        "gpu": gpu_specs,
        "config": {"B": B, "H": H, "D": D},
        "results": [asdict(r) for r in results]
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✓ Results saved to {filepath}")


# --- CLI ---

DEFAULT_N = "32,48,64,96,128,192,256,384,512,640,768,1024,1280,1536,2048"


def main():
    parser = argparse.ArgumentParser(description="CUDA Kernel Scaling Analysis")
    parser.add_argument("--n-values", type=str, default=DEFAULT_N,
                        help="Comma-separated N values to test")
    parser.add_argument("--n-range", type=str, default=None,
                        help="N range as start:end:step (e.g., 32:2048:32)")
    parser.add_argument("--no-backward", action="store_true",
                        help="Skip backward pass benchmarks")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--save", type=str, help="Save results to JSON file")
    parser.add_argument("-B", type=int, default=1, help="Batch size")
    parser.add_argument("-H", type=int, default=2, help="Number of heads")
    parser.add_argument("-D", type=int, default=32, help="Head dimension")
    args = parser.parse_args()

    if args.n_range:
        parts = args.n_range.split(":")
        start, end, step = int(parts[0]), int(parts[1]), int(parts[2])
        N_values = list(range(start, end + 1, step))
    else:
        N_values = [int(n.strip()) for n in args.n_values.split(",")]

    gpu_specs = get_gpu_specs()

    print("\n" + "=" * 60)
    print("  CUDA Scaling Analysis")
    print("=" * 60)
    print(f"  GPU:    {gpu_specs['name']}")
    print(f"  Config: B={args.B}, H={args.H}, D={args.D}")
    if len(N_values) > 8:
        print(f"  N vals: {N_values[:4]} ... {N_values[-2:]}  ({len(N_values)} points)")
    else:
        print(f"  N vals: {N_values}")
    print(f"  Warmup: {args.warmup}, Iters: {args.iters}")
    print("=" * 60 + "\n")

    results = run_scaling_analysis(
        N_values,
        B=args.B, H=args.H, D=args.D,
        include_backward=not args.no_backward,
        warmup=args.warmup, iters=args.iters,
    )

    print_results(results, args.B, args.H, args.D)

    if args.save:
        save_results_json(results, args.save, args.B, args.H, args.D)


if __name__ == "__main__":
    main()
