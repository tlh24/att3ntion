"""Wall-time + per-kernel benchmark for the backward pass.

Times _cuda_kernels.backward() with CUDA events and prints a per-kernel
breakdown via torch.profiler. Scatter cotangents are zero by default
(scatter is unused in current models); pass --scatter-grads to feed ones.

Usage:
    python benchmarks/bench_backward_tc.py --dims 2,2,256,256,256,64
"""
import argparse
import torch
import att3ntion._cuda_kernels as ck


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dims", default="2,2,256,256,256,64",
                   help="B,H,I,J,K,D")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--scatter-grads", action="store_true",
                   help="feed nonzero scatter cotangents (legacy full path)")
    args = p.parse_args()
    B, H, I, J, K, D = (int(x) for x in args.dims.split(","))
    assert I == J == K, "benchmark assumes I == J == K"
    N = I

    torch.manual_seed(0)
    dev = "cuda"
    mk = lambda: torch.randn(B, H, N, D, device=dev).to(torch.bfloat16)
    Q, R, S = mk(), mk(), mk()
    Vq1, Vq2, Vr1, Vr2, Vs1, Vs2 = mk(), mk(), mk(), mk(), mk(), mk()

    out = ck.forward(Q, R, S, Vq1, Vq2, Vr1, Vr2, Vs1, Vs2, 0.0)
    Yq, Yr, Ys = out[0], out[1], out[2]
    m_i, l_i, m_j, l_j, m_k, l_k = out[6:12]

    gYq, gYr, gYs = mk(), mk(), mk()
    if args.scatter_grads:
        gYq_, gYr_, gYs_ = mk(), mk(), mk()
    else:
        gYq_ = torch.zeros_like(gYq)
        gYr_ = torch.zeros_like(gYr)
        gYs_ = torch.zeros_like(gYs)

    def run():
        # Y passed -> TC path when eligible; ATT3_BWD_TC=0 forces scalar.
        return ck.backward(gYq, gYr, gYs, gYq_, gYr_, gYs_,
                           Q, R, S, Vq1, Vq2, Vr1, Vr2, Vs1, Vs2,
                           m_i, l_i, m_j, l_j, m_k, l_k, 0.0,
                           None, Yq, Yr, Ys)

    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()

    # Sync-bracketed per-call latency: the TC path's scatter-zero gate syncs
    # the stream, so an unsynced loop serializes host/device and overstates
    # the per-call cost.
    import time
    times = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        run()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    wall_us = times[len(times) // 2] * 1e6
    print(f"backward() wall: {wall_us:.1f} us median (min {times[0]*1e6:.1f})  "
          f"(dims {args.dims}, scatter_grads={args.scatter_grads})")

    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            run()
    rows = prof.key_averages()
    print(f"\n{'kernel':<58s} {'calls':>5s} {'avg us':>10s}")
    for r in sorted(rows, key=lambda r: -r.device_time_total):
        if r.device_time_total > 0:
            name = r.key.split("(")[0][:57]
            print(f"{name:<58s} {r.count:>5d} {r.device_time / 1.0:>10.1f}")


if __name__ == "__main__":
    main()
