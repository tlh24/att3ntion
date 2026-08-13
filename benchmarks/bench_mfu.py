#!/usr/bin/env python3
"""
MFU (Model FLOP Utilization) per-kernel and total.

Wall-clock timing for total pass MFU; torch.profiler for per-kernel MFU.

Peak is dense bf16 tensor-core TFLOPS: the kernels take bf16 in and out, and at
D=64 with the resident dim within TC_MAX_K the gather forward and backward run
on tensor cores. The scalar fallbacks (D != 64, longer sequences) issue the same
bf16 operands through the CUDA cores, so they are measured against the same
peak and simply score lower -- which is the comparison worth reporting.
"""

import os, sys
import torch
from torch.profiler import profile, ProfilerActivity

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import att3ntion._cuda_kernels as cuda_ext
from benchmarks._bench_utils import (
    create_inputs, get_gpu_specs,
    calc_forward_flops, benchmark_fn,
)


# ── per-kernel FLOP formulas ──────────────────────────────────────────────────
# Keyed by substring to match against torch.profiler kernel names.
# QS/R grad kernels use separate keys for correction vs gradient passes.

def build_flop_map(B, H, N, D):
    base = B * H * N**3
    return {
        # Forward gather  (4D + 3 fma per (i,j,k) triple)
        "Yq_gather":            base * (4*D + 3),
        "Yr_gather":            base * (4*D + 3),
        "Ys_gather":            base * (4*D + 3),
        # Forward scatter  (4D + 5)
        "Yq_scatter":           base * (4*D + 5),
        "Yr_scatter":           base * (4*D + 5),
        "Ys_scatter":           base * (4*D + 5),
        # Backward V gradients  (4D each)
        "Vq_gather_grad":       base * 4 * D,
        "Vr_gather_grad":       base * 4 * D,
        "Vs_gather_grad":       base * 4 * D,
        "Vq_scatter_grad":      base * 4 * D,
        "Vr_scatter_grad":      base * 4 * D,
        "Vs_scatter_grad":      base * 4 * D,
        # QS_grad<true>:  correction sums for Q, S (and R) = jacobian term = B*H*N³*24D
        "QS_grad_kernel_TRUE":  base * 24 * D,
        # QS_grad<false>: grad_Q + grad_S = 2 * B*H*N³*24D
        "QS_grad_kernel_FALSE": base * 48 * D,
        # R_grad<false>:  grad_R = B*H*N³*24D  (no R_grad<true> exists)
        "R_grad_kernel_FALSE":  base * 24 * D,
        # Y_gather_tc: same function as one scalar gather, two GEMMs per anchor.
        "Y_gather_tc":          base * (4*D + 3),
        # Bwd_gather_tc: per anchor, 4 score GEMMs (logits, d_a, d_r, d_c) and
        # 3 output GEMMs (Ug, U1, U2), each contracting one dim = 2*N²*D.
        "Bwd_gather_tc":        base * 14 * D,
    }


def calc_backward_gather_flops(B, H, N, D):
    """bwd_fn below zeroes the scatter cotangents, which is exactly the gate for
    the TC backward: the cross terms vanish and every correction sum collapses
    to a rowsum, so calc_backward_flops (which prices the full scalar
    decomposition, ~120*B*H*N³*D) overstates this call by roughly 3x."""
    return 3 * B * H * N**3 * 14 * D


def match_kernel(name: str, flop_map: dict):
    """Map a profiler kernel name to its analytical FLOP count."""
    nl = name.lower()
    if "qs_grad_kernel" in nl or "QS_grad_kernel" in name:
        key = "QS_grad_kernel_TRUE" if ("<true" in name or ", true," in name or "<1," in name) \
              else "QS_grad_kernel_FALSE"
        return flop_map[key]
    if "r_grad_kernel" in nl or "R_grad_kernel" in name:
        return flop_map["R_grad_kernel_FALSE"]
    for substr, flops in flop_map.items():
        if substr in name:
            return flops
    return None


def profile_pass(fn, n_warmup=5, n_iters=10):
    """Warmup then profile fn() for n_iters under torch.profiler."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(n_iters):
            fn()
        torch.cuda.synchronize()
    return prof.key_averages()


def mfu(flops, time_ms, peak_tflops):
    return flops / (time_ms * 1e-3) / (peak_tflops * 1e12)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("-B", type=int, default=1)
    p.add_argument("-H", type=int, default=2)
    p.add_argument("-N", type=int, default=128)
    p.add_argument("-D", type=int, default=64)
    p.add_argument("--show-unmatched", action="store_true",
                   help="Print kernel names that had no FLOP formula")
    args = p.parse_args()
    B, H, N, D = args.B, args.H, args.N, args.D

    specs = get_gpu_specs()
    peak = specs["bf16_tflops"]
    if peak is None:
        sys.exit(f"No tensor-core peak on record for {specs['name']}; add it to "
                 "KNOWN_GPUS in benchmarks/_bench_utils.py before quoting MFU.")
    print(f"\nGPU : {specs['name']}")
    print(f"Peak: {peak} TFLOPS (bf16 tensor core, dense) | "
          f"{specs['fp32_tflops']} TFLOPS (fp32, CUDA cores)")
    print(f"Config: B={B} H={H} N={N} D={D}\n")

    # ── inputs ────────────────────────────────────────────────────────────────
    inputs_fp32 = create_inputs(B, H, N, D)
    inputs_bf16 = tuple(t.to(torch.bfloat16) for t in inputs_fp32)

    fwd_out = cuda_ext.forward(*inputs_bf16, 0.0)
    m_i, l_i, m_j, l_j, m_k, l_k = fwd_out[6:12]
    dY_q = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16)
    dY_r = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16)
    dY_s = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16)
    dY_q_ = torch.zeros_like(dY_q)
    dY_r_ = torch.zeros_like(dY_r)
    dY_s_ = torch.zeros_like(dY_s)

    fwd_fn = lambda: cuda_ext.forward(*inputs_bf16, 0.0)
    bwd_fn = lambda: cuda_ext.backward(
        dY_q, dY_r, dY_s, dY_q_, dY_r_, dY_s_,
        *inputs_bf16, m_i, l_i, m_j, l_j, m_k, l_k, 0.0
    )

    # ── total MFU (wall-clock median) ─────────────────────────────────────────
    fwd_ms = benchmark_fn(fwd_fn, warmup=5, iters=20)[0]
    bwd_ms = benchmark_fn(bwd_fn, warmup=5, iters=20)[0]
    fwd_flops = calc_forward_flops(B, H, N, D)
    bwd_flops = calc_backward_gather_flops(B, H, N, D)

    W = 70
    print("=" * W)
    print("TOTAL MFU  (wall-clock median)")
    print("=" * W)
    print(f"  {'Pass':<12} {'Time':>9} {'GFLOPs':>9} {'MFU':>8}")
    print(f"  {'─'*12} {'─'*9} {'─'*9} {'─'*8}")
    print(f"  {'Forward':<12} {fwd_ms:>8.3f}ms {fwd_flops/1e9:>8.2f}G {mfu(fwd_flops,fwd_ms,peak)*100:>7.1f}%")
    print(f"  {'Backward':<12} {bwd_ms:>8.3f}ms {bwd_flops/1e9:>8.2f}G {mfu(bwd_flops,bwd_ms,peak)*100:>7.1f}%")
    comb_ms = fwd_ms + bwd_ms
    comb_flops = fwd_flops + bwd_flops
    print(f"  {'─'*12} {'─'*9} {'─'*9} {'─'*8}")
    print(f"  {'Fwd+Bwd':<12} {comb_ms:>8.3f}ms {comb_flops/1e9:>8.2f}G {mfu(comb_flops,comb_ms,peak)*100:>7.1f}%")

    # ── per-kernel MFU (torch.profiler) ───────────────────────────────────────
    flop_map = build_flop_map(B, H, N, D)
    print(f"\n{'=' * W}")
    print("PER-KERNEL MFU  (torch.profiler, per-call average)")
    print("=" * W)

    for pass_label, fn, pass_flops, pass_ms in [
        ("FORWARD",  fwd_fn,  fwd_flops,  fwd_ms),
        ("BACKWARD", bwd_fn,  bwd_flops,  bwd_ms),
    ]:
        events = profile_pass(fn)
        cuda_evts = [e for e in events if e.device_time_total > 0]
        cuda_evts.sort(key=lambda e: -e.device_time_total)

        print(f"\n  {pass_label}  ({pass_ms:.3f} ms wall-clock, {pass_flops/1e9:.2f} GFLOPs total)")
        print(f"  {'Kernel':<36} {'ms/call':>9} {'GFLOPs':>9} {'MFU':>8}")
        print(f"  {'─'*36} {'─'*9} {'─'*9} {'─'*8}")

        sum_ms = 0.0
        for evt in cuda_evts:
            per_call_ms = evt.device_time_total / evt.count / 1000  # μs → ms
            sum_ms += per_call_ms
            kflops = match_kernel(evt.key, flop_map)
            short = evt.key if len(evt.key) <= 36 else evt.key[:34] + ".."
            if kflops is not None:
                print(f"  {short:<36} {per_call_ms:>8.3f}ms {kflops/1e9:>8.2f}G {mfu(kflops,per_call_ms,peak)*100:>7.1f}%")
            elif args.show_unmatched:
                print(f"  {short:<36} {per_call_ms:>8.3f}ms {'—':>9} {'—':>8}")

        print(f"  {'─'*36} {'─'*9} {'─'*9} {'─'*8}")
        print(f"  {'Profiler sum':<36} {sum_ms:>8.3f}ms")
        print(f"  {'Wall-clock':<36} {pass_ms:>8.3f}ms")

    print()


if __name__ == "__main__":
    main()
