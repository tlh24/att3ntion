from __future__ import annotations

# High-level: this module includes tokenizer and dataloader.

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader, Dataset


DEFAULT_IGNORE_INDEX = -100
DEFAULT_SPECIAL_TOKENS = ("[PAD]", "[BOS]", "[SEP]", "[EOS]", "[UNK]")


@dataclass(frozen=True)
class RecogsRawExample:
    source: str
    logical_form: str
    category: str


@dataclass(frozen=True)
class EncodedExample:
    input_ids: list[int]
    labels: list[int]
    category: str
    source: str
    logical_form: str
    was_truncated: bool


class WhitespaceTokenizer:
    def __init__(self, token_to_id: dict[str, int], special_tokens: tuple[str, ...] = DEFAULT_SPECIAL_TOKENS):
        self.token_to_id = token_to_id
        self.id_to_token = {v: k for k, v in token_to_id.items()}
        self.special_tokens = special_tokens

        self.pad_token = special_tokens[0]
        self.bos_token = special_tokens[1]
        self.sep_token = special_tokens[2]
        self.eos_token = special_tokens[3]
        self.unk_token = special_tokens[4]

        self.pad_id = self.token_to_id[self.pad_token]
        self.bos_id = self.token_to_id[self.bos_token]
        self.sep_id = self.token_to_id[self.sep_token]
        self.eos_id = self.token_to_id[self.eos_token]
        self.unk_id = self.token_to_id[self.unk_token]

    @classmethod
    def from_examples(
        cls,
        examples: Iterable[RecogsRawExample],
        min_freq: int = 1,
        special_tokens: tuple[str, ...] = DEFAULT_SPECIAL_TOKENS,
    ) -> "WhitespaceTokenizer":
        counter: Counter[str] = Counter()
        for ex in examples:
            counter.update(ex.source.split())
            counter.update(ex.logical_form.split())

        token_to_id: dict[str, int] = {}
        for tok in special_tokens:
            token_to_id[tok] = len(token_to_id)

        for tok, freq in sorted(counter.items()):
            if freq >= min_freq and tok not in token_to_id:
                token_to_id[tok] = len(token_to_id)

        return cls(token_to_id=token_to_id, special_tokens=special_tokens)

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def encode_tokens(self, tokens: list[str]) -> list[int]:
        return [self.token_to_id.get(tok, self.unk_id) for tok in tokens]

    def decode_ids(self, ids: Iterable[int]) -> list[str]:
        return [self.id_to_token.get(idx, self.unk_token) for idx in ids]


def read_recogs_split(path: Path) -> list[RecogsRawExample]:
    examples: list[RecogsRawExample] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                raise ValueError(f"{path}: row {line_num} has {len(parts)} columns; expected 3")
            source, logical_form, category = parts
            examples.append(RecogsRawExample(source=source, logical_form=logical_form, category=category))
    return examples


def build_tokenizer_from_train(data_dir: Path, min_freq: int = 1) -> WhitespaceTokenizer:
    train_path = data_dir / "train.tsv"
    train_examples = read_recogs_split(train_path)
    return WhitespaceTokenizer.from_examples(train_examples, min_freq=min_freq)


def _pack_decoder_only_example(
    ex: RecogsRawExample,
    tokenizer: WhitespaceTokenizer,
    max_seq_len: int | None,
    ignore_index: int,
) -> EncodedExample:
    src_ids = tokenizer.encode_tokens(ex.source.split())
    lf_ids = tokenizer.encode_tokens(ex.logical_form.split())

    was_truncated = False
    if max_seq_len is not None:
        if max_seq_len < 3:
            raise ValueError("max_seq_len must be at least 3 for [BOS] [SEP] [EOS]")

        # Keep LF supervision whenever possible; truncate source first.
        max_src_len = max_seq_len - (1 + 1 + len(lf_ids) + 1)
        if max_src_len < 0:
            keep_lf = max_seq_len - 3
            if keep_lf < len(lf_ids):
                lf_ids = lf_ids[:keep_lf]
                was_truncated = True
            src_ids = []
            was_truncated = True
        elif len(src_ids) > max_src_len:
            src_ids = src_ids[:max_src_len]
            was_truncated = True

    input_ids = [tokenizer.bos_id] + src_ids + [tokenizer.sep_id] + lf_ids + [tokenizer.eos_id]
    prefix_len = 1 + len(src_ids) + 1
    labels = [ignore_index] * prefix_len + lf_ids + [tokenizer.eos_id]

    return EncodedExample(
        input_ids=input_ids,
        labels=labels,
        category=ex.category,
        source=ex.source,
        logical_form=ex.logical_form,
        was_truncated=was_truncated,
    )


class RecogsSequenceDataset(Dataset[EncodedExample]):
    def __init__(
        self,
        split_path: Path,
        tokenizer: WhitespaceTokenizer,
        max_seq_len: int | None = None,
        ignore_index: int = DEFAULT_IGNORE_INDEX,
    ) -> None:
        super().__init__()
        raw_examples = read_recogs_split(split_path)
        self.examples = [
            _pack_decoder_only_example(ex, tokenizer, max_seq_len=max_seq_len, ignore_index=ignore_index)
            for ex in raw_examples
        ]
        self.split_path = split_path
        self.max_seq_len = max_seq_len
        self.ignore_index = ignore_index

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> EncodedExample:
        return self.examples[idx]

    def truncation_rate(self) -> float:
        if not self.examples:
            return 0.0
        n_trunc = sum(ex.was_truncated for ex in self.examples)
        return n_trunc / len(self.examples)


def collate_decoder_only(
    batch: list[EncodedExample],
    pad_token_id: int,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
) -> dict[str, torch.Tensor | list[str]]:
    if not batch:
        raise ValueError("Cannot collate empty batch")

    batch_size = len(batch)
    max_len = max(len(ex.input_ids) for ex in batch)

    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((batch_size, max_len), ignore_index, dtype=torch.long)

    for i, ex in enumerate(batch):
        seq_len = len(ex.input_ids)
        input_ids[i, :seq_len] = torch.tensor(ex.input_ids, dtype=torch.long)
        labels[i, :seq_len] = torch.tensor(ex.labels, dtype=torch.long)

    valid = input_ids != pad_token_id
    causal = torch.tril(torch.ones((max_len, max_len), dtype=torch.bool))
    attn_mask = causal.unsqueeze(0) & valid.unsqueeze(1) & valid.unsqueeze(2)
    loss_mask = labels != ignore_index

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attn_mask": attn_mask,
        "loss_mask": loss_mask,
        "category": [ex.category for ex in batch],
        "source": [ex.source for ex in batch],
        "logical_form": [ex.logical_form for ex in batch],
    }


def build_dataloader(
    data_dir: Path,
    split: str,
    tokenizer: WhitespaceTokenizer,
    batch_size: int,
    shuffle: bool,
    max_seq_len: int | None = None,
    num_workers: int = 0,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
) -> DataLoader:
    split_path = data_dir / f"{split}.tsv"
    dataset = RecogsSequenceDataset(
        split_path=split_path,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        ignore_index=ignore_index,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda b: collate_decoder_only(
            b,
            pad_token_id=tokenizer.pad_id,
            ignore_index=ignore_index,
        ),
    )
