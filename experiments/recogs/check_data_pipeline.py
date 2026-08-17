from __future__ import annotations

# High-level: this script sanity-checks the ReCOGS tokenizer and dataloader pipeline.
# It reports vocab/truncation stats and verifies batch tensor shapes, dtypes, and masks.

import argparse
from pathlib import Path

import torch

from data import RecogsSequenceDataset, build_tokenizer_from_train, build_dataloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanity-check ReCOGS tokenizer and dataloader")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent / "data" / "raw",
        help="Directory with train/dev/test/gen TSV files",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=256,
        help="Max packed sequence length. Set <=0 to disable truncation.",
    )
    parser.add_argument("--min-freq", type=int, default=1, help="Tokenizer min frequency cutoff")
    return parser.parse_args()


def _format_pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def main() -> None:
    args = parse_args()
    max_seq_len = None if args.max_seq_len <= 0 else args.max_seq_len

    tokenizer = build_tokenizer_from_train(args.data_dir, min_freq=args.min_freq)
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(
        f"Special IDs: PAD={tokenizer.pad_id} BOS={tokenizer.bos_id} "
        f"SEP={tokenizer.sep_id} EOS={tokenizer.eos_id} UNK={tokenizer.unk_id}"
    )
    print("")

    for split in ("train", "dev", "test", "gen"):
        split_path = args.data_dir / f"{split}.tsv"
        ds = RecogsSequenceDataset(split_path, tokenizer, max_seq_len=max_seq_len)
        print(
            f"[{split}] rows={len(ds)} truncation_rate={_format_pct(ds.truncation_rate())} "
            f"(max_seq_len={max_seq_len})"
        )

        loader = build_dataloader(
            data_dir=args.data_dir,
            split=split,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            shuffle=False,
            max_seq_len=max_seq_len,
            num_workers=0,
        )

        batch = next(iter(loader))
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attn_mask = batch["attn_mask"]
        loss_mask = batch["loss_mask"]

        print(
            f"  input_ids={tuple(input_ids.shape)} labels={tuple(labels.shape)} "
            f"attn_mask={tuple(attn_mask.shape)} loss_mask={tuple(loss_mask.shape)}"
        )
        print(
            f"  dtypes: input_ids={input_ids.dtype} labels={labels.dtype} "
            f"attn_mask={attn_mask.dtype} loss_mask={loss_mask.dtype}"
        )

        # Quick mask sanity checks.
        if attn_mask.ndim != 3:
            raise RuntimeError(f"{split}: expected attn_mask ndim=3, got {attn_mask.ndim}")
        if attn_mask.shape[0] != input_ids.shape[0]:
            raise RuntimeError(f"{split}: attn_mask batch mismatch")
        if attn_mask.shape[1] != input_ids.shape[1] or attn_mask.shape[2] != input_ids.shape[1]:
            raise RuntimeError(f"{split}: attn_mask sequence dims do not match input length")
        if not torch.isfinite(input_ids.float()).all():
            raise RuntimeError(f"{split}: non-finite values in input_ids")

        print(f"  sample0 category={batch['category'][0]}")
        print(f"  sample0 source={batch['source'][0]}")
        print(f"  sample0 logical_form={batch['logical_form'][0]}")
        print("")


if __name__ == "__main__":
    main()
