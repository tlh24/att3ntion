from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from data import (
    DEFAULT_IGNORE_INDEX,
    RecogsSequenceDataset,
    build_tokenizer_from_train,
)
from model import RecogsDecoderLM
from recogs_eval import score_predictions, semantic_exact_match


@torch.no_grad()
def greedy_decode_one(
    model: RecogsDecoderLM,
    prefix_ids: list[int],
    eos_id: int,
    max_new_tokens: int,
    device: torch.device,
    prefix_lm: bool = False,
) -> list[int]:
    tokens = list(prefix_ids)
    if len(tokens) >= model.max_seq_len:
        return tokens
    # Under prefix-LM the prompt span (the initial prefix_ids) is bidirectional,
    # so every query position may attend the fixed prompt columns. This MUST match
    # the training mask built in data.collate_decoder_only or train/eval diverge.
    prefix_len = len(prefix_ids)
    for _ in range(max_new_tokens):
        if len(tokens) >= model.max_seq_len:
            break
        input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        ntok = input_ids.shape[1]
        attn_mask = torch.tril(torch.ones((ntok, ntok), device=device, dtype=torch.bool))
        if prefix_lm:
            attn_mask[:, :prefix_len] = True
        attn_mask = attn_mask.unsqueeze(0)
        logits = model(input_ids=input_ids, attn_mask=attn_mask)
        next_id = int(torch.argmax(logits[0, -1]).item())
        tokens.append(next_id)
        if next_id == eos_id:
            break
    return tokens


@torch.no_grad()
def greedy_decode_batch(
    model: RecogsDecoderLM,
    prefix_ids_list: list[list[int]],
    eos_id: int,
    pad_id: int,
    max_new_tokens: int,
    device: torch.device,
    prefix_lm: bool = False,
) -> list[list[int]]:
    """Greedy-decode a *batch* of prompts in lockstep.

    Equivalent (up to floating-point reduction order) to calling
    :func:`greedy_decode_one` on each prompt independently, but it runs one
    forward per step over the whole batch instead of one per example, which is
    where the speedup comes from (a single decode at batch 1 leaves the H100
    almost idle). There is no KV cache, so each step still recomputes the full
    sequence — the win is parallelism across examples, not per-step work.

    Each row is written *contiguously from its own prefix end*, so every real
    token keeps the absolute position it had in training (positions start at 0,
    no left-padding). Validity is tracked by a per-row write cursor (``cur_len``)
    rather than by token identity, so a generated token that happens to equal
    ``pad_id`` is still treated as a real token. Pad columns are masked exactly
    as in ``data.collate_decoder_only`` (``causal | prefix-key`` then ``& valid``).
    """
    batch = len(prefix_ids_list)
    if batch == 0:
        return []

    max_seq_len = model.max_seq_len
    prefix_lens = torch.tensor([len(p) for p in prefix_ids_list], dtype=torch.long, device=device)
    width = min(max_seq_len, int(prefix_lens.max().item()) + max_new_tokens)

    input_ids = torch.full((batch, width), pad_id, dtype=torch.long, device=device)
    for i, prefix in enumerate(prefix_ids_list):
        keep = min(len(prefix), width)
        if keep > 0:
            input_ids[i, :keep] = torch.tensor(prefix[:keep], dtype=torch.long, device=device)

    cur_len = prefix_lens.clamp(max=width).clone()
    finished = cur_len >= width
    row_idx = torch.arange(batch, device=device)

    for _ in range(max_new_tokens):
        if bool(finished.all()):
            break
        active_width = int(cur_len.max().item())
        cols = torch.arange(active_width, device=device)
        valid = cols.unsqueeze(0) < cur_len.unsqueeze(1)  # [B, W]
        causal = torch.tril(torch.ones((active_width, active_width), dtype=torch.bool, device=device))
        allowed = causal.unsqueeze(0)
        if prefix_lm:
            prefix_key = cols.unsqueeze(0) < prefix_lens.unsqueeze(1)  # [B, W]
            allowed = allowed | prefix_key.unsqueeze(1)
        attn_mask = allowed & valid.unsqueeze(1) & valid.unsqueeze(2)

        logits = model(input_ids=input_ids[:, :active_width], attn_mask=attn_mask)
        last_pos = (cur_len - 1).clamp(min=0)
        next_ids = torch.argmax(logits[row_idx, last_pos], dim=-1)  # [B]

        can_write = (~finished) & (cur_len < width)
        write_pos = cur_len.clamp(max=width - 1)
        input_ids[row_idx[can_write], write_pos[can_write]] = next_ids[can_write]
        cur_len = torch.where(can_write, cur_len + 1, cur_len)
        finished = finished | (can_write & (next_ids == eos_id)) | (cur_len >= width)

    return [input_ids[i, : int(cur_len[i].item())].tolist() for i in range(batch)]


def _decode_lf_from_generated(tokenizer, generated_ids: list[int], prefix_len: int) -> str:
    lf_ids: list[int] = []
    for tid in generated_ids[prefix_len:]:
        if tid == tokenizer.eos_id:
            break
        lf_ids.append(tid)
    return " ".join(tokenizer.decode_ids(lf_ids))


@torch.no_grad()
def evaluate_split_generation(
    model: RecogsDecoderLM,
    dataset: RecogsSequenceDataset,
    tokenizer,
    device: torch.device,
    max_new_tokens: int = 256,
    batch_size: int = 32,
    max_examples: int = 0,
    rank: int = 0,
    world_size: int = 1,
    prefix_lm: bool = False,
) -> dict[str, Any]:
    """Autoregressively decode a split and score per-category semantic EM.

    Decoding is **batched** (``greedy_decode_batch``): the owned examples are
    sorted by prefix length and chunked into ``batch_size`` groups so each
    forward stays close to the padding floor (matters for the O(N^3) hypergraph
    arm) while the GPU runs ``batch_size`` decodes at once. This is the dominant
    eval cost, so batching it is the main eval speedup (no KV cache).

    When ``world_size > 1`` (DDP), each rank decodes the examples it owns
    (``global_idx % world_size == rank``) and the raw predictions are gathered
    across ranks before scoring, so the returned metrics are identical on every
    rank.
    """
    model.eval()

    # global_idx counts every (non-degenerate) example in dataset order, so the
    # idx % world_size shard is disjoint and max_examples caps the global total.
    # The index is stable across arms (same dataset order), so it keys the
    # per-example records used for paired McNemar / bootstrap testing. We build
    # the owned work-list first, then decode it in length-sorted batches.
    owned: list[tuple[int, list[int], int, str, str]] = []  # (ex_idx, prefix_ids, prefix_len, gold_lf, category)
    global_idx = 0
    for ex in dataset.examples:
        supervised = any(lbl != DEFAULT_IGNORE_INDEX for lbl in ex.labels)
        if not supervised:
            continue
        ex_idx = global_idx
        global_idx += 1
        if max_examples > 0 and ex_idx >= max_examples:
            break
        if (ex_idx % world_size) != rank:
            continue
        prefix_len = ex.prefix_len
        owned.append((ex_idx, list(ex.input_ids[:prefix_len]), prefix_len, ex.logical_form, ex.category))

    local_preds: list[str] = []
    local_golds: list[str] = []
    local_categories: list[str] = []
    local_indices: list[int] = []

    # Length-bucket: sort by prefix length so each batch pads to near its own
    # max, not the split's global max (cubic cost for hypergraph).
    owned.sort(key=lambda r: r[2])
    for start in range(0, len(owned), batch_size):
        chunk = owned[start : start + batch_size]
        generated = greedy_decode_batch(
            model=model,
            prefix_ids_list=[r[1] for r in chunk],
            eos_id=tokenizer.eos_id,
            pad_id=tokenizer.pad_id,
            max_new_tokens=max_new_tokens,
            device=device,
            prefix_lm=prefix_lm,
        )
        for (ex_idx, _prefix_ids, prefix_len, gold_lf, category), gen in zip(chunk, generated):
            pred_lf = _decode_lf_from_generated(tokenizer, gen, prefix_len=prefix_len)
            local_preds.append(pred_lf)
            local_golds.append(gold_lf)
            local_categories.append(category)
            local_indices.append(ex_idx)

    # Gather raw predictions across ranks, then score once (no double counting).
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        gathered: list[Any] = [None] * world_size
        dist.all_gather_object(gathered, (local_preds, local_golds, local_categories, local_indices))
        all_preds: list[str] = []
        all_golds: list[str] = []
        all_categories: list[str] = []
        all_indices: list[int] = []
        for part in gathered:
            p, g, c, idx = part
            all_preds.extend(p)
            all_golds.extend(g)
            all_categories.extend(c)
            all_indices.extend(idx)
    else:
        all_preds, all_golds, all_categories, all_indices = (
            local_preds,
            local_golds,
            local_categories,
            local_indices,
        )

    per_cat_correct: dict[str, int] = defaultdict(int)
    per_cat_total: dict[str, int] = defaultdict(int)
    per_example: list[dict[str, Any]] = []
    for pred_lf, gold_lf, cat, idx in zip(all_preds, all_golds, all_categories, all_indices):
        correct = int(semantic_exact_match(pred_lf, gold_lf))
        per_cat_correct[cat] += correct
        per_cat_total[cat] += 1
        per_example.append({"idx": int(idx), "category": cat, "correct": correct})

    # Stable order by global index so records align across arms for paired tests.
    per_example.sort(key=lambda r: r["idx"])

    global_score = score_predictions(all_preds, all_golds)
    per_category = {}
    for cat in sorted(per_cat_total.keys()):
        total = per_cat_total[cat]
        correct = per_cat_correct[cat]
        per_category[cat] = {
            "exact_match": 0.0 if total == 0 else correct / total,
            "correct": correct,
            "total": total,
        }

    return {
        "n_examples": global_score.total,
        "exact_match": global_score.exact_match,
        "correct": global_score.correct,
        "per_category": per_category,
        "per_example": per_example,
    }


def _load_model_from_checkpoint(path: Path, device: torch.device) -> tuple[RecogsDecoderLM, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["config"]
    model = RecogsDecoderLM(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        attn_impl=cfg["attn_impl"],
        max_seq_len=cfg["max_seq_len"],
        scatter=cfg.get("scatter", False),
        ffn_hidden=cfg.get("ffn_hidden"),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ReCOGS model with autoregressive generation")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data" / "raw")
    parser.add_argument("--split", type=str, default="gen", choices=["dev", "test", "gen"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-examples", type=int, default=0, help="0 means all")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model, cfg = _load_model_from_checkpoint(args.checkpoint, device)
    tokenizer = build_tokenizer_from_train(args.data_dir, min_freq=cfg.get("min_freq", 1))
    dataset = RecogsSequenceDataset(
        split_path=args.data_dir / f"{args.split}.tsv",
        tokenizer=tokenizer,
        max_seq_len=cfg.get("max_seq_len"),
        ignore_index=DEFAULT_IGNORE_INDEX,
    )
    result = evaluate_split_generation(
        model=model,
        dataset=dataset,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        max_examples=args.max_examples,
        prefix_lm=cfg.get("prefix_lm", False),
    )

    payload = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "result": result,
        "config": cfg,
    }
    print(json.dumps(payload, indent=2))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

