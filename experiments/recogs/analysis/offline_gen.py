"""Offline generation eval for a trained run (e.g. RUN_5 d768 train-only).

Reconstructs the model from <run_dir>/config.json, loads the dev-selected best.pt,
and decodes dev/test/gen exactly as train.py's --eval-at-end path does — writing
final_metrics.json + per_example_{split}.json in the identical format.

Usage:
    python analysis/offline_gen.py <run_dir> [<run_dir> ...]
Single-GPU (rank 0, world_size 1). Run on whatever GPU CUDA_VISIBLE_DEVICES pins.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
RECOGS = HERE.parent
sys.path.insert(0, str(RECOGS))
sys.path.insert(0, str(RECOGS.parent.parent))  # repo root for att3ntion

from data import (  # noqa: E402
    RecogsSequenceDataset,
    build_tokenizer_from_train,
    DEFAULT_IGNORE_INDEX,
)
from model import RecogsDecoderLM  # noqa: E402
from evaluate import evaluate_split_generation  # noqa: E402


def eval_one(run_dir: Path, data_dir: Path, device: str) -> None:
    cfg = json.loads((run_dir / "config.json").read_text())
    tok = build_tokenizer_from_train(data_dir)
    model = RecogsDecoderLM(
        vocab_size=tok.vocab_size,
        d_model=cfg["d_model"],
        n_heads=cfg["heads"],
        n_layers=cfg["layers"],
        attn_impl=cfg["attn"],
        max_seq_len=cfg["max_seq_len"],
        scatter=cfg["scatter"],
        ffn_hidden=cfg["ffn_hidden"],
    ).to(device)
    best = run_dir / "checkpoints" / "best.pt"
    ckpt = best if best.exists() else (run_dir / "checkpoints" / "last.pt")
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()
    print(f"[{run_dir.name}] loaded {ckpt.name}; d_model={cfg['d_model']} heads={cfg['heads']} "
          f"layers={cfg['layers']} prefix_lm={cfg['prefix_lm']}", flush=True)

    metrics = {}
    for split in ("dev", "test", "gen"):
        ds = RecogsSequenceDataset(
            split_path=data_dir / f"{split}.tsv", tokenizer=tok,
            max_seq_len=cfg["max_seq_len"], ignore_index=DEFAULT_IGNORE_INDEX,
        )
        t0 = time.time()
        res = evaluate_split_generation(
            model=model, dataset=ds, tokenizer=tok, device=device,
            max_new_tokens=cfg["eval_max_new_tokens"], batch_size=cfg["eval_batch_size"],
            max_examples=cfg["eval_max_examples"], rank=0, world_size=1,
            prefix_lm=cfg["prefix_lm"],
        )
        per_example = res.pop("per_example", [])
        metrics[split] = res
        (run_dir / f"per_example_{split}.json").write_text(json.dumps(per_example) + "\n")
        print(f"[{run_dir.name}/{split}] exact_match={100.0 * res['exact_match']:.2f}% "
              f"n={res['n_examples']} ({time.time() - t0:.1f}s)", flush=True)
    (run_dir / "final_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"[{run_dir.name}] wrote final_metrics.json", flush=True)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = RECOGS / "data" / "raw"
    for rd in sys.argv[1:]:
        eval_one(Path(rd), data_dir, device)


if __name__ == "__main__":
    main()
