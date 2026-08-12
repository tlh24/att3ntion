"""Shared utilities for benchmark scripts."""

import torch
from typing import Dict, List, Optional, Tuple


# --- Input Creation ---

def create_inputs(B: int, H: int, N: int, D: int,
                  device='cuda', dtype=torch.float32, seed=42):
    """Create standard (Q, R, S, Vq_1, Vq_2, Vr_1, Vr_2, Vs_1, Vs_2) inputs."""
    torch.manual_seed(seed)
    return tuple(
        torch.randn(B, H, N, D, device=device, dtype=dtype) for _ in range(9)
    )


MASK_KINDS = ("none", "causal", "all_true", "random", "dead_rows", "prefix_lm_pad")


def make_mask(kind: str, B: int, N: int, device='cuda') -> Optional[torch.Tensor]:
    """Attention mask of the given kind, or None for 'none'.

    Mirrors `_make_mask` in tests/test_backward_tc.py so benchmark and
    correctness runs exercise the same shapes.
    """
    if kind == "none":
        return None
    eye = torch.arange(N, device=device)
    tri = torch.tril(torch.ones(B, N, N, device=device, dtype=torch.bool))
    if kind == "causal":
        return tri
    if kind == "all_true":
        return torch.ones(B, N, N, device=device, dtype=torch.bool)
    if kind == "random":
        m = torch.randint(0, 2, (B, N, N), device=device, dtype=torch.bool)
        m[:, eye, eye] = True          # keep every anchor's denominator alive
        return m
    if kind == "dead_rows":
        m = tri.clone()
        m[:, 5, :] = False
        m[:, N - 1, :] = False
        return m
    if kind == "prefix_lm_pad":
        P = N // 2
        m = torch.zeros(B, N, N, device=device, dtype=torch.bool)
        m[:, :, :P] = True
        m |= tri
        m[:, :, N - 4:] = False
        m[:, N - 4:, :] = False
        return m
    raise ValueError(f"unknown mask kind: {kind}")


# --- Timing ---

def benchmark_fn(fn, warmup=5, iters=20):
    """Benchmark fn() using CUDA events. Returns (median_ms, sorted_times_ms)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    return times[len(times) // 2], times


# --- GPU Specs ---

KNOWN_GPUS = {
    "NVIDIA GeForce RTX 4080 Laptop GPU": {"fp32_tflops": 33.9,  "mem_bw_gbs": 432.0},
    "NVIDIA GeForce RTX 4080":            {"fp32_tflops": 48.7,  "mem_bw_gbs": 716.8},
    "NVIDIA GeForce RTX 4080 SUPER":      {"fp32_tflops": 52.0,  "mem_bw_gbs": 736.3},
    "NVIDIA GeForce RTX 4090":            {"fp32_tflops": 82.6,  "mem_bw_gbs": 1008.0},
    "NVIDIA GeForce RTX 3090":            {"fp32_tflops": 35.6,  "mem_bw_gbs": 936.2},
    "NVIDIA GeForce RTX 3080":            {"fp32_tflops": 29.8,  "mem_bw_gbs": 760.3},
    "NVIDIA A100-SXM4-40GB":              {"fp32_tflops": 19.5,  "mem_bw_gbs": 1555.0},
    "NVIDIA A100-SXM4-80GB":              {"fp32_tflops": 19.5,  "mem_bw_gbs": 2039.0},
    "NVIDIA H100":                        {"fp32_tflops": 51.2,  "mem_bw_gbs": 3350.0},
}


def get_gpu_specs() -> Dict:
    """Get GPU specs for roofline/utilization analysis. Falls back to estimation."""
    if not torch.cuda.is_available():
        return {}

    props = torch.cuda.get_device_properties(0)
    name = props.name

    specs = KNOWN_GPUS.get(name)
    if specs is None:
        for gpu_name, gpu_specs in KNOWN_GPUS.items():
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
        "fp32_tflops": specs["fp32_tflops"],
        "mem_bw_gbs": specs["mem_bw_gbs"],
    }


# --- FLOP / Bandwidth Estimation ---

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


def estimate_forward_bytes(
    B: int,
    H: int,
    N: int,
    D: int,
    tensor_bytes: int = 2,
    stats_bytes: int = 4,
) -> Dict:
    """Estimate total bytes transferred during forward pass.

    Defaults reflect mixed-precision forward:
    - 9 inputs + 6 outputs in BF16 (`tensor_bytes=2`)
    - 6 softmax-stat tensors in FP32 (`stats_bytes=4`)
    """
    input_bytes = 9 * B * H * N * D * tensor_bytes
    output_bytes = 6 * B * H * N * D * tensor_bytes + 6 * B * H * N * stats_bytes
    total = input_bytes + output_bytes
    return {"total_bytes": total, "total_mb": total / 1e6}


def format_flops(flops: float) -> str:
    if flops >= 1e12:
        return f"{flops/1e12:.2f}T"
    elif flops >= 1e9:
        return f"{flops/1e9:.2f}G"
    elif flops >= 1e6:
        return f"{flops/1e6:.2f}M"
    return f"{flops:.0f}"


# --- GPU Peak Throughput ---

def measure_gpu_peak_tflops(dtype=torch.bfloat16, size=8192, iters=16) -> Optional[float]:
    """Measure peak GPU TFLOP/s via matrix multiplication."""
    if not torch.cuda.is_available():
        return None
    device = torch.device('cuda')
    a = torch.randn(size, size, dtype=dtype, device=device)
    b = torch.randn(size, size, dtype=dtype, device=device)
    c = torch.randn(size, size, dtype=dtype, device=device)

    for _ in range(3):
        _ = a @ b
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        d = a @ b
        d += b @ c
    end.record()
    torch.cuda.synchronize()

    time_ms = start.elapsed_time(end)
    flops_per_iter = 2 * (2 * size**3 + size**2)
    return iters * flops_per_iter / 1e12 / (time_ms / 1000)
