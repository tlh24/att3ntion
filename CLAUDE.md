# CLAUDE.md — Dispatcher

## Project Overview 

This project implements the hypergraph (three-way) attention in CUDA without ever materializing the NxNxN attention tensor. It uses tiling and streaming techniques from flash attention.

## Environment

```bash
source myenv/bin/activate
```

## File structure

```
att3ntion/
├── cuda/                          # ★ CUDA kernels (primary work area)
│   ├── forward.cu                 #   Forward pass kernels
│   ├── backward.cu                #   Backward pass kernels (Q/R/S grad)
│   └── common.cuh                 #   Shared constants, tile sizes, utilities
│
├── cpp/                           # C++/PyTorch bindings (glue layer)
│   ├── cuda_bindings.h            #   Declarations for CUDA entry points
│   ├── cuda_bindings.cpp          #   Pybind11 wrappers for CUDA kernels
│   └── torch_reference.cpp        #   Pure-PyTorch reference kernels
│
├── att3ntion/                     # Python package
│   ├── __init__.py                #   Exports: HypergraphAttention, QuickGELU, naive variants
│   ├── _autograd.py               #   Autograd bridge + nn.Module (HypergraphAttention)
│   └── _naive.py                  #   Naive O(N³) PyTorch impl for correctness testing
│
├── tests/
│   └── test_kernel_correctness.py #   Correctness tests (CUDA vs naive)
│
├── benchmarks/
│   ├── bench_regression.py        #   Regression benchmark (--save, --show-history)
│   ├── bench_cuda_scaling.py      #   CUDA scaling over N
│   ├── bench_cuda_vs_torch.py     #   CUDA vs Torch comparison
│   ├── bench_mfu.py               #   MFU (model FLOPs utilization)
│   ├── profile_kernels.py         #   Nsight kernel profiling launcher
│   ├── _bench_utils.py            #   Shared benchmark helpers
│   ├── benchmark_history.jsonl    #   Local benchmark history
│   └── benchmark_history_h100.jsonl # H100 benchmark history
│
├── experiments/                   # Training experiments using att3ntion
│   ├── analogy/                   #   Analogy task (gen_data, model, train)
│   ├── compositional/             #   Compositional task (gen_data, train, sweep)
│   ├── cuda_comparison/           #   CUDA vs Torch training comparison
│   └── archive/                   #   Old experiments / checkpoints
│
├── scripts/                       # Shell helpers
│   ├── kpush.sh / kpull.sh        #   Push/pull code to/from K8s pod
│   ├── ksync.sh                   #   Bidirectional sync wrapper
│   ├── profile.sh                 #   Local ncu profiling
│   └── profile_h100.sh            #   Remote H100 ncu profiling
│
├── profiling_reports/             # Nsight Compute outputs
│   ├── csv/                       #   Exported CSV metric tables
│   └── ncu_rep/                   #   .ncu-rep binary reports
│
├── docs/                          # Design notes & optimization logs
│   └── optimization_log/          #   Per-optimization write-ups
│
├── Makefile                       # build, test, bench, profile, history targets
├── setup.py                       # Extension build config (setuptools + CUDA)
├── pyproject.toml                 # Project metadata
├── requirements.txt               # Python deps
└── demo.py                        # Quick demo script
```

## Key Make targets

| Target | What it does |
|---|---|
| `make build` | Build/install the extension |
| `make test` | Quick correctness tests |
| `make test-full` | Detailed correctness tests |
| `make bench` | Run regression benchmark |
| `make bench-save NOTE="…"` | Save benchmark with note |
| `make bench-save-h100 NOTE="…"` | Push, bench on H100, pull results |
| `make history` / `make history-h100` | Show benchmark history |
| `make iterate` | Quick test + bench cycle |


## H100 clock locking (for bench-save-h100)

```bash
# Lock before benchmarking (run as root inside the pod, no sudo needed)
nvidia-smi -i 0 --lock-gpu-clocks=1980,1980

# Unlock after
nvidia-smi -i 0 --reset-gpu-clocks
```

## Available skills

- `profile` — `.claude/skills/profile/SKILL.md`
- `ksync` — `.claude/skills/ksync/SKILL.md`
