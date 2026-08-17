from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from data import (
    DEFAULT_IGNORE_INDEX,
    RecogsSequenceDataset,
    build_dataloader,
    build_tokenizer_from_train,
)
from evaluate import evaluate_split_generation
from model import RecogsDecoderLM




def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_distributed() -> tuple[bool, int, int, int]:
    """Initialize DDP from torchrun env vars.

    Returns (distributed, rank, local_rank, world_size). When not launched
    under torchrun (no WORLD_SIZE>1), runs as a single process so local/CPU
    invocations keep working unchanged.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return True, rank, local_rank, world_size
    return False, 0, 0, 1


def _attn_tag(attn_impl: str) -> str:
    return {
        "hypergraph_cuda": "hgc",
        "graph_flash": "gflash",
        "graph_clean": "gclean",
    }[attn_impl]


def _jsonify(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value


@torch.no_grad()
def teacher_forced_eval_loss(
    model: RecogsDecoderLM,
    loader,
    device: torch.device,
    max_batches: int = 0,
) -> float:
    model.eval()
    losses: list[float] = []
    use_amp = device.type == "cuda"
    for batch_idx, batch in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attn_mask = batch["attn_mask"].to(device)
        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_amp
            else nullcontext()
        )
        with amp_ctx:
            logits = model(input_ids=input_ids, attn_mask=attn_mask)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.shape[-1]),
                shift_labels.reshape(-1),
                ignore_index=DEFAULT_IGNORE_INDEX,
            )
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ReCOGS prefix-LM with symmetric attention baselines")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data" / "raw")
    parser.add_argument("--attn", type=str, default="hypergraph_cuda", choices=["hypergraph_cuda", "graph_flash", "graph_clean"])
    parser.add_argument(
        "--scatter",
        action="store_true",
        help="hypergraph_cuda only: enable scatter/write (H-full arm). Default off = H-gather.",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="micro-batches accumulated per optimizer step; effective batch = batch_size * grad_accum_steps * world_size",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means no explicit cap")
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--min-freq", type=int, default=1)

    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument(
        "--ffn-hidden",
        type=int,
        default=0,
        help="FFN hidden width; 0 = 3*d_model. Used to parameter-match the graph arm to hypergraph.",
    )
    parser.add_argument(
        "--prefix-lm",
        action="store_true",
        help="Bidirectional prefix LM: full attention within [BOS] src [SEP], causal over the LF span.",
    )

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-steps", type=int, default=400)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--divergence-loss-threshold",
        type=float,
        default=1000.0,
        help="abort if a (finite) train loss exceeds this; catches blow-ups the NaN guard misses",
    )

    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="0 disables; otherwise stop after this many dev evals without improvement",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=1e-4,
        help=(
            "minimum dev-loss improvement that counts as 'better' for early-stop "
            "patience. Microscopic new lows (e.g. 0.0078 -> 0.0077) below this no "
            "longer reset the counter, so a converged run actually stops instead of "
            "grinding to the epoch cap. best.pt selection still uses the true min."
        ),
    )
    parser.add_argument("--eval-dev-every", type=int, default=500)
    parser.add_argument("--eval-gen-every", type=int, default=0, help="0 disables periodic gen eval")
    parser.add_argument("--eval-gen-max-examples", type=int, default=512, help="for periodic gen eval only")
    parser.add_argument("--eval-dev-max-batches", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--eval-max-examples",
        type=int,
        default=0,
        help="cap examples per split in the --eval-at-end generation eval (0 = all); for fast smoke tests",
    )
    parser.add_argument("--eval-at-end", action="store_true", help="Run dev/test/gen generation eval after training")

    parser.add_argument("--log-name", type=str, default="recogs")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "runs")
    parser.add_argument("--save-every", type=int, default=0, help="0 disables periodic step-based ckpt")
    parser.add_argument(
        "--ckpt-every-epochs",
        type=int,
        default=0,
        help="0 disables; otherwise save a checkpoint at the end of every N epochs (epoch_{e}.pt)",
    )
    return parser.parse_args()


def _lr_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    """
    Linear warmup + cosine decay.
    """
    if warmup_steps > 0 and step < warmup_steps:
        return max(1e-8, step / max(1, warmup_steps))
    if total_steps <= warmup_steps:
        return 1.0
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return 0.1 + 0.9 * cosine


def main() -> None:
    args = parse_args()

    distributed, rank, local_rank, world_size = setup_distributed()
    is_main = rank == 0

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if distributed:
        device = torch.device(f"cuda:{local_rank}")
    if args.attn == "hypergraph_cuda" and device.type != "cuda":
        raise RuntimeError("hypergraph_cuda requires CUDA")

    if args.d_model % args.heads != 0:
        raise ValueError(f"d_model ({args.d_model}) must be divisible by heads ({args.heads})")
    d_head = args.d_model // args.heads
    if args.attn == "hypergraph_cuda" and d_head not in (16, 32, 64):
        raise ValueError(
            f"hypergraph_cuda requires d_head in {{16, 32, 64}}; got d_head={d_head} from d_model={args.d_model}, heads={args.heads}"
        )
    if args.scatter and args.attn != "hypergraph_cuda":
        raise ValueError(f"--scatter is only valid with --attn hypergraph_cuda; got --attn {args.attn}")
    if args.grad_accum_steps < 1:
        raise ValueError(f"--grad-accum-steps must be >= 1, got {args.grad_accum_steps}")

    set_global_seed(args.seed)
    if is_main:
        print(f"Using device: {device}")
        print(f"Seed: {args.seed}")
        if distributed:
            print(f"DDP: world_size={world_size} (per-rank batch={args.batch_size}, global batch={args.batch_size * world_size})")

    tokenizer = build_tokenizer_from_train(args.data_dir, min_freq=args.min_freq)
    train_loader = build_dataloader(
        data_dir=args.data_dir,
        split="train",
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        shuffle=True,
        max_seq_len=args.max_seq_len,
        num_workers=args.num_workers,
        ignore_index=DEFAULT_IGNORE_INDEX,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        seed=args.seed,
        prefix_lm=args.prefix_lm,
    )
    dev_loader = build_dataloader(
        data_dir=args.data_dir,
        split="dev",
        tokenizer=tokenizer,
        batch_size=args.eval_batch_size,
        shuffle=False,
        max_seq_len=args.max_seq_len,
        num_workers=args.num_workers,
        ignore_index=DEFAULT_IGNORE_INDEX,
        prefix_lm=args.prefix_lm,
    )
    # Teacher-forced test loss is tracked over time alongside dev (cheap: one
    # forward pass, no generation). Test is in-distribution held-out; it is NOT
    # used for model selection (dev only).
    test_loader = build_dataloader(
        data_dir=args.data_dir,
        split="test",
        tokenizer=tokenizer,
        batch_size=args.eval_batch_size,
        shuffle=False,
        max_seq_len=args.max_seq_len,
        num_workers=args.num_workers,
        ignore_index=DEFAULT_IGNORE_INDEX,
        prefix_lm=args.prefix_lm,
    )
    gen_dataset = RecogsSequenceDataset(
        split_path=args.data_dir / "gen.tsv",
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        ignore_index=DEFAULT_IGNORE_INDEX,
    )

    core_model = RecogsDecoderLM(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        attn_impl=args.attn,
        max_seq_len=args.max_seq_len,
        scatter=args.scatter,
        ffn_hidden=args.ffn_hidden,
    ).to(device)
    if is_main:
        core_model.printParamCount()

    if distributed:
        model = DDP(core_model, device_ids=[local_rank], find_unused_parameters=False)
    else:
        model = core_model

    optimizer = torch.optim.AdamW(core_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # train_step counts *optimizer* steps; with grad accumulation each step
    # consumes args.grad_accum_steps micro-batches.
    steps_per_epoch = len(train_loader) // args.grad_accum_steps
    if args.max_steps > 0:
        total_steps = args.max_steps
    else:
        total_steps = args.epochs * max(1, steps_per_epoch)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: _lr_lambda(step, args.warmup_steps, total_steps)
    )

    run_dir = args.out_dir / f"{args.log_name}_{args.attn}_s{args.seed}"
    ckpt_dir = run_dir / "checkpoints"
    if is_main:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    losslog = None
    eval_curve = None
    if is_main:
        losslog_path = run_dir / "losslog.txt"
        losslog = losslog_path.open("w", encoding="utf-8")
        print(f"writing loss log to {losslog_path}")

        # Over-time eval curve: one row per dev-eval step (train/dev/test loss).
        eval_curve = (run_dir / "eval_curve.tsv").open("w", encoding="utf-8")
        eval_curve.write("step\ttrain_loss\tdev_loss\ttest_loss\n")
        eval_curve.flush()

        config = vars(args).copy()
        config["vocab_size"] = tokenizer.vocab_size
        config["d_head"] = d_head
        config["world_size"] = world_size
        config["global_batch_size"] = args.batch_size * world_size
        (run_dir / "config.json").write_text(json.dumps(_jsonify(config), indent=2) + "\n", encoding="utf-8")

    def save_checkpoint(name: str, step: int, extra: dict[str, Any] | None = None) -> Path:
        ckpt_path = ckpt_dir / name
        payload = {
            "model_state": core_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "step": step,
            "config": {
                "vocab_size": tokenizer.vocab_size,
                "d_model": args.d_model,
                "n_heads": args.heads,
                "n_layers": args.layers,
                "attn_impl": args.attn,
                "max_seq_len": args.max_seq_len,
                "min_freq": args.min_freq,
                "scatter": args.scatter,
                "ffn_hidden": core_model.ffn_hidden,
                "prefix_lm": args.prefix_lm,
            },
        }
        if extra:
            payload.update(extra)
        torch.save(payload, ckpt_path)
        return ckpt_path

    train_step = 0
    use_amp = device.type == "cuda"
    finished = False
    best_dev_loss = float("inf")
    evals_without_improve = 0

    # Wall-clock instrumentation: time each epoch, then project the full
    # train + eval runtime from the measured rate. Written to timing.json so a
    # launcher / monitor can read whether the run fits the overnight window.
    epoch_times: list[float] = []
    timing_path = run_dir / "timing.json"

    def write_timing(extra: dict[str, Any] | None = None) -> None:
        if not is_main:
            return
        mean_epoch = float(np.mean(epoch_times)) if epoch_times else float("nan")
        payload: dict[str, Any] = {
            "epochs_done": len(epoch_times),
            "epochs_target": args.epochs,
            "steps_per_epoch": steps_per_epoch,
            "last_epoch_sec": epoch_times[-1] if epoch_times else None,
            "mean_epoch_sec": mean_epoch,
            "projected_train_sec": (mean_epoch * args.epochs) if epoch_times else None,
            "train_step": train_step,
        }
        if extra:
            payload.update(extra)
        timing_path.write_text(json.dumps(_jsonify(payload), indent=2) + "\n", encoding="utf-8")

    if is_main:
        print("\ntrain started...")
    accum = args.grad_accum_steps
    for epoch in range(args.epochs):
        epoch_start = time.time()
        if distributed:
            train_loader.batch_sampler.set_epoch(epoch)
        model.train()
        micro_idx = 0  # position within the current accumulation window
        accum_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attn_mask = batch["attn_mask"].to(device)

            if micro_idx == 0:
                optimizer.zero_grad(set_to_none=True)

            # Skip the DDP all-reduce on every micro-batch except the one that
            # triggers the optimizer step (the last in the window).
            is_step_micro = (micro_idx + 1) == accum
            sync_ctx = (
                model.no_sync() if (distributed and not is_step_micro) else nullcontext()
            )
            amp_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_amp
                else nullcontext()
            )
            with sync_ctx, amp_ctx:
                logits = model(input_ids=input_ids, attn_mask=attn_mask)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.shape[-1]),
                    shift_labels.reshape(-1),
                    ignore_index=DEFAULT_IGNORE_INDEX,
                )
            loss_val = float(loss.detach().cpu())
            if not torch.isfinite(loss):
                # A per-rank raise: torchrun propagates the failure and tears
                # down the whole group, so this does not deadlock the other ranks.
                raise RuntimeError(f"[rank {rank}] Non-finite loss at step {train_step}: {loss_val}")
            if loss_val > args.divergence_loss_threshold:
                # Catch finite-but-exploding loss (e.g. the scatter arm blowing up
                # to ~1e6) so a diverged run fails fast instead of wasting GPU-hours
                # training garbage and then eval-ing it to 0%.
                raise RuntimeError(
                    f"[rank {rank}] Divergence at step {train_step}: train loss {loss_val:.1f} "
                    f"> threshold {args.divergence_loss_threshold}"
                )

            # Scale so the accumulated gradient equals the mean over the window.
            (loss / accum).backward()
            accum_loss += float(loss.detach().cpu())
            micro_idx += 1
            if micro_idx < accum:
                continue
            # Window complete -> one optimizer step.
            micro_idx = 0
            train_step += 1
            torch.nn.utils.clip_grad_norm_(core_model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            scheduler.step()

            train_loss = accum_loss / accum
            accum_loss = 0.0
            if is_main:
                losslog.write(f"{train_step}\t{train_loss:.8f}\t0.0\n")
                if train_step % 50 == 0:
                    lr = float(optimizer.param_groups[0]["lr"])
                    print(f"step={train_step} epoch={epoch+1}/{args.epochs} train_loss={train_loss:.4f} lr={lr:.3e}")
                    losslog.flush()

            if args.eval_dev_every > 0 and train_step % args.eval_dev_every == 0:
                # Dev eval + model selection run on rank 0 only (dev loader is
                # not sharded). The other ranks block at the broadcast below, so
                # every rank stays in lockstep and resumes together.
                stop_flag = 0
                if is_main:
                    dev_loss = teacher_forced_eval_loss(
                        model=core_model,
                        loader=dev_loader,
                        device=device,
                        max_batches=args.eval_dev_max_batches,
                    )
                    print(f"[eval/dev] step={train_step} teacher_forced_loss={dev_loss:.4f}")
                    # Monitor-only: teacher-forced test loss over time (not used
                    # for selection). Same max_batches budget as dev.
                    test_loss = teacher_forced_eval_loss(
                        model=core_model,
                        loader=test_loader,
                        device=device,
                        max_batches=args.eval_dev_max_batches,
                    )
                    print(f"[eval/test] step={train_step} teacher_forced_loss={test_loss:.4f}")
                    if eval_curve is not None:
                        eval_curve.write(f"{train_step}\t{train_loss:.6f}\t{dev_loss:.6f}\t{test_loss:.6f}\n")
                        eval_curve.flush()
                    model.train()
                    # Model selection happens on dev ONLY (never gen), to avoid
                    # leaking ood signal into checkpoint choice. best.pt tracks the
                    # true minimum dev loss; the early-stop *counter* uses a
                    # min-delta so microscopic new lows don't reset patience.
                    if math.isfinite(dev_loss):
                        improved_enough = dev_loss < (best_dev_loss - args.early_stop_min_delta)
                        if dev_loss < best_dev_loss:
                            best_dev_loss = dev_loss
                            save_checkpoint(
                                name="best.pt",
                                step=train_step,
                                extra={"epoch": epoch + 1, "dev_loss": dev_loss},
                            )
                            print(f"[ckpt] new best dev loss {dev_loss:.4f} -> best.pt")
                        if improved_enough:
                            evals_without_improve = 0
                        else:
                            evals_without_improve += 1
                            if args.early_stop_patience > 0 and evals_without_improve >= args.early_stop_patience:
                                stop_flag = 1
                                print(
                                    f"[early-stop] no dev improvement >{args.early_stop_min_delta:g} for "
                                    f"{evals_without_improve} evals (patience={args.early_stop_patience}); stopping."
                                )
                if distributed:
                    stop_tensor = torch.tensor([stop_flag], device=device, dtype=torch.int)
                    dist.broadcast(stop_tensor, src=0)
                    stop_flag = int(stop_tensor.item())
                if stop_flag:
                    finished = True
                    break
                model.train()

            if args.eval_gen_every > 0 and train_step % args.eval_gen_every == 0:
                gen_res = evaluate_split_generation(
                    model=core_model,
                    dataset=gen_dataset,
                    tokenizer=tokenizer,
                    device=device,
                    max_new_tokens=args.eval_max_new_tokens,
                    batch_size=args.eval_batch_size,
                    max_examples=args.eval_gen_max_examples,
                    rank=rank,
                    world_size=world_size,
                    prefix_lm=args.prefix_lm,
                )
                if is_main:
                    gen_res.pop("per_example", None)
                    print(
                        f"[eval/gen] step={train_step} exact_match={100.0 * gen_res['exact_match']:.2f}% "
                        f"n={gen_res['n_examples']}"
                    )
                    eval_path = run_dir / f"gen_eval_step_{train_step}.json"
                    eval_path.write_text(json.dumps(gen_res, indent=2) + "\n", encoding="utf-8")
                model.train()

            if is_main and args.save_every > 0 and train_step % args.save_every == 0:
                save_checkpoint(name=f"step_{train_step}.pt", step=train_step, extra={"epoch": epoch + 1})

            if args.max_steps > 0 and train_step >= args.max_steps:
                finished = True
                break

        epoch_sec = time.time() - epoch_start
        epoch_times.append(epoch_sec)
        if is_main:
            mean_epoch = float(np.mean(epoch_times))
            proj_h = mean_epoch * args.epochs / 3600.0
            print(
                f"[timing] epoch {epoch + 1}/{args.epochs} took {epoch_sec:.1f}s "
                f"(mean {mean_epoch:.1f}s/epoch -> ~{proj_h:.2f}h for {args.epochs} epochs, eval excluded)"
            )
            write_timing()

        # Sparse epoch-boundary checkpoints: keep a trajectory of weights so gen
        # can be scored offline later on the dev-selected checkpoint.
        if is_main and not finished and args.ckpt_every_epochs > 0 and (epoch + 1) % args.ckpt_every_epochs == 0:
            ep_ckpt = save_checkpoint(
                name=f"epoch_{epoch + 1}.pt", step=train_step, extra={"epoch": epoch + 1}
            )
            print(f"[ckpt] epoch {epoch + 1} -> {ep_ckpt.name}")

        if finished:
            break

    if is_main:
        final_ckpt = save_checkpoint(name="last.pt", step=train_step)
        print(f"saved checkpoint: {final_ckpt}")

    if args.eval_at_end:
        # Final reporting uses the dev-selected best.pt (one locked policy for
        # both arms). Ensure rank 0 has written it, then every rank loads it from
        # the shared filesystem and decodes its shard of each split.
        if distributed:
            dist.barrier()
        best_path = ckpt_dir / "best.pt"
        if best_path.exists():
            state = torch.load(best_path, map_location=device)
            core_model.load_state_dict(state["model_state"])
            if is_main:
                print(f"[final] loaded dev-selected checkpoint {best_path}")
        elif is_main:
            print("[final] no best.pt found; evaluating last weights")

        metrics: dict[str, Any] = {}
        eval_timing: dict[str, float] = {}
        for split in ("dev", "test", "gen"):
            split_dataset = RecogsSequenceDataset(
                split_path=args.data_dir / f"{split}.tsv",
                tokenizer=tokenizer,
                max_seq_len=args.max_seq_len,
                ignore_index=DEFAULT_IGNORE_INDEX,
            )
            split_start = time.time()
            split_res = evaluate_split_generation(
                model=core_model,
                dataset=split_dataset,
                tokenizer=tokenizer,
                device=device,
                max_new_tokens=args.eval_max_new_tokens,
                batch_size=args.eval_batch_size,
                max_examples=args.eval_max_examples,
                rank=rank,
                world_size=world_size,
                prefix_lm=args.prefix_lm,
            )
            eval_timing[f"{split}_sec"] = time.time() - split_start
            # Per-example records (for paired McNemar/bootstrap across arms) are
            # bulky on the 21k gen split, so write them per-split and keep
            # final_metrics.json lean (aggregate + per-category only).
            per_example = split_res.pop("per_example", [])
            metrics[split] = split_res
            if is_main:
                (run_dir / f"per_example_{split}.json").write_text(
                    json.dumps(per_example) + "\n", encoding="utf-8"
                )
                print(
                    f"[final/{split}] exact_match={100.0 * split_res['exact_match']:.2f}% "
                    f"n={split_res['n_examples']} ({eval_timing[f'{split}_sec']:.1f}s)"
                )
        if is_main:
            (run_dir / "final_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
            write_timing({"eval_at_end": eval_timing, "eval_total_sec": float(sum(eval_timing.values()))})

    if is_main and losslog is not None:
        losslog.flush()
        losslog.close()
    if is_main and eval_curve is not None:
        eval_curve.flush()
        eval_curve.close()
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    if is_main:
        print("train complete.")


if __name__ == "__main__":
    main()

