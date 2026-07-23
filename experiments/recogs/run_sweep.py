from __future__ import annotations

"""Arm-aware GPU-semaphore launcher for the ReCOGS attention sweep.

One *process per (arm, seed)*, each pinned to a single GPU via
``CUDA_VISIBLE_DEVICES`` (NOT DDP). For the ~3M-param models in this study, N
independent single-GPU runs beat one N-GPU DDP run (comm overhead +
underutilisation), and the runs are embarrassingly parallel.

Arms are named so the two hypergraph variants (gather vs full-scatter) stay
distinct in run dirs / logs even though they share ``attn_impl=hypergraph_cuda``:

    graph_flash  - pairwise SDPA, post-attn GELU (symmetric baseline G)        4L
    graph_clean  - canonical pre-norm SDPA, no GELU (acceptance-gate control)  4L
    hg_gather    - hypergraph_cuda, scatter=False (3-way arity)                3L
    hg_full      - hypergraph_cuda, scatter=True  (3-way + scatter/write)      3L

Jobs are ordered seed-major (all arms for seed 1, then seed 2, ...) so that if
the run is cut short, completed seeds carry the full arm ladder rather than a
lopsided subset. ``--skip-existing`` skips any (arm, seed) whose run dir already
has ``final_metrics.json``, making the driver safe to re-run after a crash.
"""

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


# Param-matched ladder (d_model=256, heads=4, FFN 256->768->256, pos_emb=512):
#   graph_flash 4L ~3.22M | hg_gather 3L ~3.15M (arity contrast, <3% apart)
#   hg_full 3L ~3.74M     | matched to hg_gather at EQUAL depth; the extra
#                           params are intrinsic to scatter (doubled Wv).
ARMS: dict[str, dict] = {
    "graph_flash": {"attn": "graph_flash", "scatter": False, "layers": 4},
    "graph_clean": {"attn": "graph_clean", "scatter": False, "layers": 4},
    "hg_gather": {"attn": "hypergraph_cuda", "scatter": False, "layers": 3},
    "hg_full": {"attn": "hypergraph_cuda", "scatter": True, "layers": 3},
}


@dataclass
class Job:
    arm: str
    seed: int
    log_path: Path
    run_dir: Path
    gpu: int = -1
    cmd: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch ReCOGS arm x seed sweep across GPUs")
    parser.add_argument("--arms", type=str, default="graph_flash,hg_gather,hg_full")
    parser.add_argument("--seeds", type=str, default="1,2,3")
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument("--train-script", type=Path, default=Path(__file__).parent / "train.py")
    parser.add_argument("--out-dir", type=Path, required=True, help="run output dir, e.g. exp/v9_title/runs")
    parser.add_argument("--log-dir", type=Path, default=None, help="job/driver logs dir (default: <out-dir>/driver_logs)")
    parser.add_argument("--skip-existing", action="store_true", help="skip (arm,seed) with final_metrics.json")

    # Training regime (v2 plan: 512 + long train toward dev plateau).
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum-steps", type=int, default=4)  # effective batch 128
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--early-stop-patience", type=int, default=15)
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=1e-4,
        help="min dev-loss improvement counted for patience (microscopic lows stop resetting it)",
    )
    parser.add_argument("--eval-dev-every", type=int, default=400)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-max-new-tokens", type=int, default=512)
    parser.add_argument("--ckpt-every-epochs", type=int, default=0)
    parser.add_argument(
        "--prefix-lm",
        action="store_true",
        help="run every arm as a bidirectional prefix LM (passes --prefix-lm to train.py)",
    )
    parser.add_argument(
        "--graph-ffn-hidden",
        type=int,
        default=0,
        help="FFN hidden width applied to graph arms only (param-match to hypergraph); 0 = default 3*d_model",
    )
    parser.add_argument("--extra-args", type=str, default="")
    return parser.parse_args()


def _split_csv_int(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _split_csv_str(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def build_command(args: argparse.Namespace, arm: str, seed: int) -> list[str]:
    spec = ARMS[arm]
    cmd = [
        sys.executable, "-u", str(args.train_script),
        "--attn", spec["attn"],
        "--seed", str(seed),
        "--layers", str(spec["layers"]),
        "--max-seq-len", str(args.max_seq_len),
        "--batch-size", str(args.batch_size),
        "--grad-accum-steps", str(args.grad_accum_steps),
        "--epochs", str(args.epochs),
        "--early-stop-patience", str(args.early_stop_patience),
        "--early-stop-min-delta", str(args.early_stop_min_delta),
        "--eval-dev-every", str(args.eval_dev_every),
        "--eval-dev-max-batches", "0",
        "--eval-batch-size", str(args.eval_batch_size),
        "--eval-max-new-tokens", str(args.eval_max_new_tokens),
        "--eval-at-end",
        "--ckpt-every-epochs", str(args.ckpt_every_epochs),
        "--log-name", arm,
        "--out-dir", str(args.out_dir),
    ]
    if spec["scatter"]:
        cmd.append("--scatter")
    if args.prefix_lm:
        cmd.append("--prefix-lm")
    # Graph arms can be FFN-widened to parameter-match the hypergraph arm; the
    # hypergraph arms keep their default FFN (their extra params come from attention).
    if args.graph_ffn_hidden > 0 and spec["attn"] in ("graph_flash", "graph_clean"):
        cmd.extend(["--ffn-hidden", str(args.graph_ffn_hidden)])
    if args.extra_args:
        cmd.extend(args.extra_args.strip().split())
    return cmd


def main() -> None:
    args = parse_args()
    seeds = _split_csv_int(args.seeds)
    arms = _split_csv_str(args.arms)
    gpus = _split_csv_int(args.gpus)
    if not gpus:
        raise ValueError("At least one GPU id must be provided via --gpus")
    for arm in arms:
        if arm not in ARMS:
            raise ValueError(f"Unknown arm {arm!r}; choices: {sorted(ARMS)}")

    if args.log_dir is None:
        args.log_dir = args.out_dir / "driver_logs"
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Seed-major ordering: full arm ladder completes per seed.
    jobs: list[Job] = []
    for seed in seeds:
        for arm in arms:
            spec = ARMS[arm]
            run_dir = args.out_dir / f"{arm}_{spec['attn']}_s{seed}"
            log_path = args.log_dir / f"{arm}_s{seed}.log"
            jobs.append(Job(arm=arm, seed=seed, log_path=log_path, run_dir=run_dir))

    if args.skip_existing:
        kept = []
        for job in jobs:
            if (job.run_dir / "final_metrics.json").exists():
                print(f"[skip] {job.arm} s{job.seed} already has final_metrics.json", flush=True)
            else:
                kept.append(job)
        jobs = kept

    free_gpus = list(gpus)
    running: dict[subprocess.Popen, Job] = {}
    queue = list(jobs)
    n_fail = 0

    print(
        f"Launching {len(jobs)} jobs over GPUs {gpus} | arms={arms} seeds={seeds} "
        f"| seq_len={args.max_seq_len} eff_batch={args.batch_size * args.grad_accum_steps}",
        flush=True,
    )
    while queue or running:
        while queue and free_gpus:
            job = queue.pop(0)
            job.gpu = free_gpus.pop(0)
            job.cmd = build_command(args, job.arm, job.seed)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(job.gpu)
            env["PYTHONUNBUFFERED"] = "1"
            fout = job.log_path.open("w", encoding="utf-8", buffering=1)
            proc = subprocess.Popen(job.cmd, env=env, stdout=fout, stderr=subprocess.STDOUT)
            running[proc] = job
            print(f"[start] arm={job.arm} seed={job.seed} gpu={job.gpu} pid={proc.pid} log={job.log_path}", flush=True)

        time.sleep(3.0)
        for proc in [p for p in running if p.poll() is not None]:
            job = running.pop(proc)
            ret = proc.returncode
            free_gpus.append(job.gpu)
            if ret == 0:
                print(f"[done] arm={job.arm} seed={job.seed} ok", flush=True)
            else:
                n_fail += 1
                print(f"[done] arm={job.arm} seed={job.seed} FAILED(ret={ret}) -> see {job.log_path}", flush=True)

    print(f"Sweep complete. failures={n_fail}", flush=True)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
