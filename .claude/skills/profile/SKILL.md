---
name: profile
description: Run Nsight Compute on a single CUDA kernel and save both the .ncu-rep and a CSV under profiling_reports/. Requires a short tag. Use when the user says "profile kernel X", "run ncu on X", etc. Does nothing else — no classification, no planning, no logging.
---

# Profile — run ncu, save two files, stop

## Inputs

- **kernel** (required) — e.g. `QS_grad_kernel`, `Yq_gather`
- **tag** (required) — short slug describing this run: `baseline`, `after_float4`, `before_smem_fix`. If the user didn't give one, ASK. Do not invent one.
- **pass** (optional) — `forward` or `backward`, default `forward`
- **dims** (optional) — `B,H,I,J,K,D`, default `1,2,128,128,128,64` (H100: `2,2,256,256,256,64`)

## Output files

```
profiling_reports/ncu_rep/<kernel>_<tag>_<mon><day>_<HH:MM><am|pm>.ncu-rep
profiling_reports/csv/<kernel>_<tag>_<mon><day>_<HH:MM><am|pm>.csv
```

Example: `QS_grad_kernel_baseline_apr20_04:58pm.ncu-rep`

## Procedure

One command. Do not inline the shell logic — call the script:

```bash
# Local GPU:
./scripts/profile.sh <kernel> <tag> [<pass>] [<dims>]

# H100 pod (pushes sources, profiles remotely, pulls results):
./scripts/profile_h100.sh <kernel> <tag> [<pass>] [<dims>]
```

The script handles: env activation, stale-build check, timestamp, directory creation, and the ncu invocation.

Print the two paths it emits, then stop.

## Halt condition

After the script exits and the two paths are printed. No JSON, no bottleneck analysis, no session file, no further skills.

## What this skill MUST NOT do

- Inline the shell procedure (use the script)
- Invent a tag when the user didn't provide one (ask instead)
- Interpret metrics or classify bottlenecks
- Edit any CUDA source
- Commit anything
