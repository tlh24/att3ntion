from __future__ import annotations

# High-level: this module includes tokenizer and dataloader.

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader, Dataset, Sampler


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
    # Length of the prompt span `[BOS] src [SEP]`. Used by the prefix-LM mask to
    # make the prefix bidirectional (keys 0:prefix_len attendable by every query).
    prefix_len: int


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
        prefix_len=prefix_len,
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
    prefix_lm: bool = False,
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

    # attn_mask[b, i, j] is True when query i may attend key j.
    #   causal  : j <= i (default decoder-only).
    #   prefix-LM: additionally any key in the prompt span [BOS] src [SEP]
    #              (j < prefix_len) is attendable by every query, so the prefix
    #              is bidirectional while the LF span stays strictly causal.
    valid = input_ids != pad_token_id
    causal = torch.tril(torch.ones((max_len, max_len), dtype=torch.bool))
    allowed = causal.unsqueeze(0)
    if prefix_lm:
        prefix_key = torch.zeros((batch_size, max_len), dtype=torch.bool)
        for i, ex in enumerate(batch):
            prefix_key[i, : ex.prefix_len] = True
        allowed = allowed | prefix_key.unsqueeze(1)
    attn_mask = allowed & valid.unsqueeze(1) & valid.unsqueeze(2)
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


class BucketedBatchSampler(Sampler[list[int]]):
    """Length-bucketed ("sortish") batch sampler.

    Because hypergraph attention cost scales with the *padded* batch length cubed,
    mixing short and long sequences in one batch forces every member to pay the
    longest member's N^3 cost. This sampler keeps stochasticity (so SGD still sees
    fresh batch compositions each epoch) while ensuring each batch contains
    similar-length sequences, which collapses padding waste close to its floor.

    Each epoch:
      1. shuffle all indices,
      2. split into megabatches of ``batch_size * bucket_multiplier``,
      3. sort each megabatch by length,
      4. chunk into ``batch_size`` batches,
      5. shuffle the order in which those batches are yielded.

    Randomness is drawn from the global torch RNG, so seeding via
    ``torch.manual_seed`` upstream keeps runs reproducible, while each call to
    ``__iter__`` (i.e. each epoch) produces a different shuffle.
    """

    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        bucket_multiplier: int = 50,
        drop_last: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if bucket_multiplier <= 0:
            raise ValueError(f"bucket_multiplier must be positive, got {bucket_multiplier}")
        self.lengths = lengths
        self.batch_size = batch_size
        self.bucket_multiplier = bucket_multiplier
        self.drop_last = drop_last

    def __iter__(self):
        n = len(self.lengths)
        order = torch.randperm(n).tolist()
        megabatch_size = self.batch_size * self.bucket_multiplier

        batches: list[list[int]] = []
        for mb_start in range(0, n, megabatch_size):
            megabatch = order[mb_start : mb_start + megabatch_size]
            megabatch.sort(key=lambda idx: self.lengths[idx])
            for b_start in range(0, len(megabatch), self.batch_size):
                batch = megabatch[b_start : b_start + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                batches.append(batch)

        for batch_idx in torch.randperm(len(batches)).tolist():
            yield batches[batch_idx]

    def __len__(self) -> int:
        n = len(self.lengths)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size


class DistributedBucketedBatchSampler(Sampler[list[int]]):
    """Distributed (DDP) variant of :class:`BucketedBatchSampler`.

    Each rank must yield the *same number* of batches per epoch, otherwise DDP
    deadlocks when a rank with fewer steps stops calling collectives. To get
    that while preserving the length-bucketing rationale (hypergraph cost scales
    with the padded length cubed), every rank:

      1. rebuilds the **identical** global batch list from a generator seeded by
         ``seed + epoch`` (so all ranks agree without any communication),
      2. truncates it to a multiple of ``world_size`` (equal count per rank),
      3. takes a strided slice ``batches[rank::world_size]`` so each rank sees a
         balanced mix of short/long batches.

    Per-rank ``batch_size`` times ``world_size`` is the effective global batch.
    Call :meth:`set_epoch` each epoch so the shuffle changes while staying
    consistent across ranks.
    """

    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        rank: int,
        world_size: int,
        bucket_multiplier: int = 50,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if bucket_multiplier <= 0:
            raise ValueError(f"bucket_multiplier must be positive, got {bucket_multiplier}")
        if world_size <= 0:
            raise ValueError(f"world_size must be positive, got {world_size}")
        if not (0 <= rank < world_size):
            raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
        self.lengths = lengths
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.bucket_multiplier = bucket_multiplier
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _num_global_batches(self) -> int:
        # Deterministic and independent of the shuffle: depends only on dataset
        # size, megabatch size and batch size. Mirrors the chunking in __iter__.
        n = len(self.lengths)
        megabatch_size = self.batch_size * self.bucket_multiplier
        total = 0
        for mb_start in range(0, n, megabatch_size):
            mb_len = min(megabatch_size, n - mb_start)
            if self.drop_last:
                total += mb_len // self.batch_size
            else:
                total += (mb_len + self.batch_size - 1) // self.batch_size
        return total

    def __iter__(self):
        # Seeded generator -> every rank derives the same global batch list.
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        n = len(self.lengths)
        order = torch.randperm(n, generator=g).tolist()
        megabatch_size = self.batch_size * self.bucket_multiplier

        batches: list[list[int]] = []
        for mb_start in range(0, n, megabatch_size):
            megabatch = order[mb_start : mb_start + megabatch_size]
            megabatch.sort(key=lambda idx: self.lengths[idx])
            for b_start in range(0, len(megabatch), self.batch_size):
                batch = megabatch[b_start : b_start + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                batches.append(batch)

        order_perm = torch.randperm(len(batches), generator=g).tolist()
        batches = [batches[i] for i in order_perm]

        # Equal number of batches per rank (drop the ragged tail).
        n_per_rank = len(batches) // self.world_size
        total = n_per_rank * self.world_size
        batches = batches[:total]

        for batch in batches[self.rank :: self.world_size]:
            yield batch

    def __len__(self) -> int:
        return self._num_global_batches() // self.world_size


def build_dataloader(
    data_dir: Path,
    split: str,
    tokenizer: WhitespaceTokenizer,
    batch_size: int,
    shuffle: bool,
    max_seq_len: int | None = None,
    num_workers: int = 0,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
    bucket: bool | None = None,
    bucket_multiplier: int = 50,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 0,
    prefix_lm: bool = False,
) -> DataLoader:
    split_path = data_dir / f"{split}.tsv"
    dataset = RecogsSequenceDataset(
        split_path=split_path,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        ignore_index=ignore_index,
    )
    collate_fn = lambda b: collate_decoder_only(
        b,
        pad_token_id=tokenizer.pad_id,
        ignore_index=ignore_index,
        prefix_lm=prefix_lm,
    )

    # Default: bucket whenever we would otherwise shuffle (i.e. training).
    if bucket is None:
        bucket = shuffle

    if bucket:
        lengths = [len(ex.input_ids) for ex in dataset.examples]
        if distributed and world_size > 1:
            batch_sampler: Sampler[list[int]] = DistributedBucketedBatchSampler(
                lengths=lengths,
                batch_size=batch_size,
                rank=rank,
                world_size=world_size,
                bucket_multiplier=bucket_multiplier,
                seed=seed,
            )
        else:
            batch_sampler = BucketedBatchSampler(
                lengths=lengths,
                batch_size=batch_size,
                bucket_multiplier=bucket_multiplier,
            )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
