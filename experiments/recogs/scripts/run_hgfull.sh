#!/usr/bin/env bash
# RUN hg_full (hypergraph_cuda, scatter=True, 3L) — same config as the on-file
# hg_gather (no-scatter) v3 runs, only scatter differs. Env-parametrized so each
# pod runs its own seed subset across its own GPUs. Launch DETACHED:
#   actl pod attach <pod> -- bash -lc 'cd /home/dev/workspace/experiments/recogs && \
#     SEEDS=1,2,3,4,5,6 GPUS=0,1,2,3 setsid nohup bash scripts/run_hgfull.sh >/dev/null 2>&1 </dev/null & echo launched'
#
# NOTE: hg_full is the known-divergent arm; seeds that exceed the loss-divergence
# guard abort without final_metrics.json. --skip-existing + the retry loop will
# re-attempt them (bounded by MAX_ATTEMPTS) — that is expected, not a bug.
set -uo pipefail
cd "$(dirname "$0")/.."                       # -> experiments/recogs
REPO_ROOT="$(cd ../.. && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p sweep_logs runs/v4

SEEDS=${SEEDS:-1,2,3,4,5,6,7,8,9,10}
GPUS=${GPUS:-0,1,2,3}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}
LOG=sweep_logs/v4_hgfull_$(hostname).log

{
  echo "=== hg_full driver start $(date -Is) on $(hostname) seeds=$SEEDS gpus=$GPUS ==="
  attempt=1; rc=1
  while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
    echo "--- attempt $attempt/$MAX_ATTEMPTS $(date -Is) ---"
    python -u run_sweep.py --arms hg_full --seeds "$SEEDS" --gpus "$GPUS" --skip-existing \
      --out-dir runs/v4 --max-seq-len 512 --batch-size 32 --grad-accum-steps 4 \
      --epochs 50 --early-stop-patience 60 --early-stop-min-delta 1e-4 \
      --eval-dev-every 100 --eval-batch-size 32 --eval-max-new-tokens 384 --prefix-lm
    rc=$?
    if [ "$rc" -eq 0 ]; then echo "--- all runs have final_metrics (attempt $attempt) ---"; break; fi
    echo "--- attempt $attempt rc=$rc; incomplete (diverged/crashed seeds), retry in 30s ---"
    sleep 30; attempt=$((attempt + 1))
  done
  echo "=== hg_full driver done rc=$rc $(date -Is) ==="
} >> "$LOG" 2>&1
