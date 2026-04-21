# CLAUDE.md — Dispatcher

Do nothing by default. Wait for the user to invoke a skill. Only the `profile`
skill exists right now.

## Environment

```bash
source myenv/bin/activate
```

## Editable CUDA files

- `cuda/forward.cu`
- `cuda/backward.cu`
- `cuda/common.cuh`

## Never edit

- `cpp/cuda_bindings.h`, `cpp/cuda_bindings.cpp`
- Anything under `tests/`
- Anything under `benchmarks/`
- `setup.py`, `Makefile`

## Emergency revert

```bash
git checkout -- cuda/forward.cu cuda/backward.cu cuda/common.cuh
```

## Available skills

- `profile` — `.claude/skills/profile/SKILL.md`
