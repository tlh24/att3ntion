# Experiments

Training experiments comparing **hypergraph attention** vs **graph (standard) attention** on synthetic arithmetic tasks.

## Directory Structure

### `analogy/` — Simple Analogy Task
Three-term analogy: if `A op B = C` then `D op E = ?`. The model must infer the operator and predict the result.

- **`gen_data.py`** — Data generation for unambiguous modular arithmetic analogies
- **`model.py`** — `SimpleAnalogyModel` definition with support for `torch`, `cuda`, and `torch_cpp` attention backends
- **`train.py`** — Benchmarks CUDA vs Torch C++ implementations: timing, loss curves, and accuracy comparison plots

### `compositional/` — Compositional Arithmetic Tasks
More complex tasks involving composed operations, expression tree evaluation, and pointer-based reasoning.

- **`gen_data_comp.py`** — Data generation library with multiple tasks:
  - Task 1: Single modular arithmetic
  - Task 2: Composed arithmetic `(a op b) op (c op d)`
  - Task 3: Modular arithmetic with pointer arguments
  - Task 4: Parse tree evaluation with positional encoding
  - Task 5: Token insertion/deletion
  - Task 6: One-digit hex multiplication
  - Task 7: Full expression tree with positional pointers
  - Task 8: Two-digit hex multiplication with local allocation
- **`train.py`** — Main training script with `SimpleCompModel` (RMSNorm, SwiGLU, head subspaces). Supports tasks 3, 4, and 7 via `--task` flag. Compares hypergraph vs graph attention.
- **`run_sweep.sh`** — Runs `train.py` across all attention types and tasks 3/4/7

### `cuda_comparison/` — CUDA vs Torch C++ Kernel Benchmarks
Side-by-side training comparison of the custom CUDA kernel vs the Torch C++ reference implementation.

- **`train.py`** — Trains `CompModelComparison` with both backends, reports epoch times, loss, accuracy, and speedup

### `archive/` — Older Experiments (kept for reference)
Earlier iterations of compositional model experiments. These are superseded by `compositional/train.py`.

- **`comp_model.py`** — Original `SimpleCompModel` + `CompModel` with curriculum learning / weight freezing
- **`comp_model_6_mul.py`** — Specialized experiment for task 6 (hex multiplication)
- **`comp_model_shift.py`** — Specialized experiment for task 5 (token shifting)
- **`checkpoints/`** — Saved model weights from earlier runs
