# ReCOGS

Compositional generalization experiment on a fragment of natural language. 

> **Status:** sections below are filled in incrementally. Only the sections
> marked *Done* are complete; the rest are placeholders.

## Contents

- [Data & tokenization](#data--tokenization) — *Done*
- [Dataloader & batching](#dataloader--batching) — *Done*
- [Model](#model) — *Done*
- [Training](#training) — *Done*
- [Evaluation](#evaluation) — *Done*
- [Generation / OOD](#generation--ood) — *Done*

---

## Data & tokenization

### Input format

Each split is a tab-separated file (`data/raw/{train,dev,test,gen}.tsv`) with three
columns — source sentence, logical form (LF), and category:

```
A rose was helped by a dog .	rose ( 1 ) ; dog ( 6 ) ; help ( 3 ) AND theme ( 3 , 1 ) AND agent ( 3 , 6 )	in_distribution
The sailor dusted a boy .	* sailor ( 1 ) ; boy ( 4 ) ; dust ( 2 ) AND agent ( 2 , 1 ) AND theme ( 2 , 4 )	in_distribution
```

Both sides arrive pre-spaced: every word, LF atom, punctuation mark (`(`, `)`,
`,`, `;`), connective (`AND`), and the definite marker `*` is its own
space-delimited token. The integer ids (`1`, `6`, `3`, …) are the variables that
bind predicates to their arguments. `read_recogs_split` parses each row into
`RecogsRawExample(source, logical_form, category)` (`data.py`).

### Tokenization

`WhitespaceTokenizer` (`data.py`) uses word-level tokenization with a closed
vocabulary:

- **Splitting** is `str.split()`, which works directly on the pre-spaced data.
- **Vocabulary** is built from `train.tsv` only (`build_tokenizer_from_train` →
  `from_examples`), over source and LF tokens jointly, so both sides share one
  vocabulary and the same ids. Tokens are added in sorted order, filtered by
  `min_freq` (default `1`).
- **Special tokens** occupy ids 0–4: `[PAD] [BOS] [SEP] [EOS] [UNK]`.
- **OOV tokens** map to `[UNK]` (`encode_tokens`). ReCOGS keeps vocabulary roughly
  constant across splits, so OOV is rare; OOD difficulty comes from novel
  structure.

Word-level tokenization is standard in the ReCOGS / COGS literature: the LF is
designed to split on whitespace into atoms, and the `ReCOGS_pos` variant assigns
each variable index its own vocabulary token. The scoring code relies on the same
convention, detecting variables as all-digit tokens (see [Evaluation](#evaluation)).

## Dataloader & batching

`build_dataloader` (`data.py`) builds a `RecogsSequenceDataset` for a split, picks
a sampler, and attaches the collator. Source and target share one causal sequence.

### Packing

`_pack_decoder_only_example` encodes one row into a single sequence:

```
input_ids:  [BOS] <source tokens> [SEP] <logical-form tokens> [EOS]
labels:     -100  -100 …    -100  -100  <logical-form tokens> [EOS]
```

The prompt (`[BOS] src [SEP]`) is masked in `labels` with `-100`
(`DEFAULT_IGNORE_INDEX`), so loss covers the LF tokens and final `[EOS]` only.
`RecogsSequenceDataset` packs every example in `__init__`, making `__getitem__` a
list lookup.

When `max_seq_len` is set, the source is truncated ahead of the LF to keep the LF
supervision intact; the LF itself is clipped only when it alone exceeds the
budget. Each example carries a `was_truncated` flag, and `truncation_rate()`
reports the affected fraction (≈0 for a well-chosen `max_seq_len`).

### Collation

`collate_decoder_only` pads a batch to its longest member and produces:

- `input_ids` / `labels` — right-padded with `pad_id` / `-100`.
- `attn_mask` — a 3-D per-example mask, `causal & valid_query & valid_key`, so a
  position attends only to non-pad positions at or before it.
- `loss_mask` — `labels != ignore_index` (the LF + `[EOS]` positions).
- `category`, `source`, `logical_form` — raw strings passed through for scoring.

### Length bucketing

`BucketedBatchSampler` is a "sortish" sampler. Hypergraph attention cost scales
with the padded sequence length cubed (N³), so a single long sequence makes every
batch member pay its N³ cost. Bucketing keeps each batch close in length while
preserving SGD stochasticity. Each epoch it:

1. shuffles all indices,
2. splits them into megabatches of `batch_size * bucket_multiplier` (default `50`),
3. sorts each megabatch by length,
4. chunks into `batch_size` batches,
5. shuffles the order in which batches are yielded.

Randomness uses the global torch RNG, so `torch.manual_seed` upstream keeps runs
reproducible while batch compositions still vary per epoch. Bucketing is enabled
whenever `shuffle=True` (training); eval splits use a sequential loader.

### Distributed (DDP) variant

`DistributedBucketedBatchSampler` keeps every rank on the same step count to avoid
DDP deadlock. Each rank:

1. rebuilds the same global batch list from a generator seeded by `seed + epoch`
   (no inter-rank communication),
2. truncates to a multiple of `world_size` (equal batch count per rank),
3. takes the strided slice `batches[rank::world_size]` for a balanced length mix.

Effective global batch is `batch_size * world_size`. Call `set_epoch(epoch)` each
epoch to advance the shuffle consistently across ranks.

## Model

`RecogsDecoderLM` (`model.py`) is a decoder-only, pre-norm Transformer LM over the
packed `[BOS] src [SEP] lf [EOS]` sequence:

```
x = token_emb(ids) + pos_emb(pos)
for layer:
    x = x + attn(norm1(x), mask)
    x = x + ffn(norm2(x))
logits = lm_head(x)
```

Defaults (from the run configs): `d_model=256`, `n_heads=4` (`d_head=64`),
`n_layers=3`, `max_seq_len` 256/512, `vocab_size=893`. Norm is **RMSNorm**
(pre-norm); positions are **learned absolute** embeddings (rotary unsupported);
FFN is `Linear(d, 3d) → ReLU → Linear(3d, d)`; `lm_head` is untied from
`token_emb`.

Attention is fully causal and loss is computed only over the LF span, so the model
is *partly* prefix-LM (prefix-masked loss) but reads the source left-to-right. A
bidirectional-prefix variant (full attention within `[BOS] src [SEP]`, causal over
the LF) would make it closer to seq2seq and may help, since the source is fully
observed; it is worth trying but not required for the current comparison.

Everything above is held constant; the **attention mechanism is the only variable**
(`attn_impl`):

- **`graph_clean`** — pairwise SDPA attention (flash on CUDA), canonical block.
- **`graph_flash`** — same SDPA attention with a post-attention GELU, matching the
  hypergraph branch's GELU placement.
- **`hypergraph_cuda`** — 3-way hypergraph attention (CUDA kernels) with QuickGELU.
  The `scatter` flag selects gather (`False`) or full (`True`, doubles the value
  projections). Cost is O(N³) in sequence length, which motivates length
  [bucketing](#length-bucketing).

Hypergraph attention has more parameters per layer than graph attention, so
comparisons give the graph arm extra layers to approach parity (e.g. graph `L=4`
vs hypergraph `L=3`); parity is approximate, not exact.

| arm | layers | params (@512) |
|---|---|---|
| graph (`clean`/`flash`) | 3 | 2.56M |
| graph | 4 | 3.22M |
| hypergraph (gather) | 3 | 3.15M |
| hypergraph (scatter) | 3 | 3.74M |

## Training

*Overview as of 2026-06-16 (`train.py`).*

**Objective**

- Next-token cross-entropy on shifted logits/labels (teacher forcing).
- `ignore_index=-100`, so supervision falls on the LF span only.

**Optimization**

- AdamW: `lr` default `1e-4`, `weight_decay 1e-2`.
- Schedule: linear warmup + cosine decay to a 10% floor (`warmup_steps` ~400).
- Gradient clipping to `grad_clip` (default `1.0`).
- bfloat16 autocast on CUDA.
- Gradient accumulation: effective batch = `batch_size * grad_accum_steps *
  world_size`; loss scaled by `1/accum`.

**Distributed (DDP)**

- Launched via `torchrun` (NCCL).
- `no_sync()` on every micro-batch except the step-triggering one.
- Bucketed sampler's `set_epoch` called each epoch.
- Dev eval and model selection run on rank 0 only; a broadcast stop flag keeps
  ranks in lockstep.

**Divergence guards**

- Non-finite loss raises immediately.
- Finite loss above `divergence_loss_threshold` (default `1000`) also aborts, so a
  diverged run fails fast instead of burning GPU-hours.

**Eval & model selection**

- Teacher-forced dev loss every `eval_dev_every` steps; best dev loss saves
  `best.pt`.
- Selection uses dev only (never gen), so no OOD signal leaks into the checkpoint
  choice — one locked policy applied to every arm.
- Optional periodic generation eval reports gen SEM.
- Early stopping after `early_stop_patience` dev evals without improvement.

**Checkpoints & logging**

- Saves `best.pt` (dev-selected), `last.pt`, and optional step/epoch checkpoints,
  each bundling model + optimizer + scheduler state and config.
- Each run writes `config.json` and a per-step loss log.
- With `--eval-at-end`: reloads `best.pt` and decodes dev/test/gen into
  `final_metrics.json` plus per-example records for paired significance tests.
- Runs live in `exp/<experiment>/runs/{log_name}_{attn}_s{seed}/`, alongside a
  per-step `losslog.txt`. Each experiment folder under `exp/` bundles its
  write-up (`RUN_N.md`), driver script, plot scripts, figures, and run
  artifacts — see `exp/README.md` for the index.

## Evaluation

The headline metric is **semantic exact match (SEM)**: a prediction counts as
correct when it is semantically equivalent to the gold logical form, up to the two
invariances ReCOGS defines:

1. **Conjunct order** — the order of conjuncts (separated by `;` and `AND`).
2. **Variable renaming** — the specific integer ids, up to a consistent
   (bijective) renaming between predicted and gold variables.

### Implementation (`metrics.py`)

`semantic_exact_match(pred_lf, gold_lf)` resolves these invariances:

- `_split_atoms` splits each LF into atomic facts on `;` and `AND`, absorbing
  conjunct order.
- `_parse_atoms` represents each atom as a `(shape, variables)` pair, replacing
  every variable id (all-digit token) with `#`. Atoms correspond only when their
  shapes match.
- `backtrack` searches for a bijection between predicted and gold variable ids
  that makes the atom sets correspond, using a most-constrained-variable (MRV)
  heuristic. Forward (`pred→gold`) and backward (`gold→pred`) maps are both
  enforced for a one-to-one renaming.

`score_predictions` aggregates this into an exact-match rate. `evaluate.py`
greedily decodes each example (`greedy_decode_one`) and reports per-category and
global SEM, plus per-example correctness records keyed by a stable global index
for paired significance tests (McNemar / bootstrap). Under DDP, each rank decodes
the shard it owns and predictions are gathered before scoring, so every rank
reports the same numbers.

### Relation to the official metric

This follows the official ReCOGS metric (`recogs_exact_match` in
[cs224u/compgen.py](https://github.com/cgpotts/cs224u/blob/main/compgen.py)): same
invariances, same notion of a variable (`\d+`). The official version compares
conjuncts as a set (deduplicated) and applies a fuller `normalize_formula`; this
implementation compares them as a multiset (length check + 1-to-1 matching) and
normalizes whitespace only. They diverge solely on LFs with duplicate identical
conjuncts (absent from gold LFs), where this version is marginally stricter. It
assumes the gold LFs are canonically pre-spaced (one space around every `(`, `)`,
`,`, `;`, `AND`), matching the tokenizer's convention.

## Generation / OOD

Compositional generalization is measured on the `gen.tsv` split
(`evaluate_split_generation`, `evaluate.py`):

- **Decoding** — greedy autoregressive (`greedy_decode_one`): start from the
  `[BOS] src [SEP]` prefix, take `argmax` each step until `[EOS]` or
  `max_new_tokens`, rebuilding the causal mask per step (no KV cache).
- **Scoring** — semantic exact match, reported per-category and global
  (see [Evaluation](#evaluation)).
- **Sharding** — under DDP each rank decodes its slice; predictions are gathered
  before scoring, so generation (the dominant eval cost) scales near-linearly.

---

## References

- Wu, Manning, Potts. *ReCOGS: How Incidental Details of a Logical Form
  Overshadow an Evaluation of Semantic Interpretation.* TACL 2023.
  [paper](https://aclanthology.org/2023.tacl-1.96.pdf) ·
  [repo](https://github.com/frankaging/ReCOGS)
- Official semantic exact match:
  [cs224u/compgen.py](https://github.com/cgpotts/cs224u/blob/main/compgen.py)
