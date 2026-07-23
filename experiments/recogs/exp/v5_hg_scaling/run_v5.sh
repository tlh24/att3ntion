#!/usr/bin/env bash
# RUN_5 (v5) hypergraph Chinchilla-style scaling sweep — generic per-pod driver.
# Detached, crash-retrying, --skip-existing (safe to re-run). One pod invocation
# drives that pod's local GPUs; the full sweep is spread across recogs / recogs-scale
# / att3ntion (see RUN_5.md for the GPU map). Launch DETACHED, e.g.:
#
#   actl -n strange-loop pod attach <pod> -- bash -lc \
#     'cd /home/dev/workspace/experiments/recogs && \
#      ARMS=hg_d512 SEEDS=1,2,3,4 GPUS=0,1,2,3 \
#      setsid nohup bash scripts/run_v5.sh >/dev/null 2>&1 </dev/null & echo pid $!'
#
# Env knobs (all optional except ARMS/SEEDS/GPUS):
#   ARMS   comma list of arm names (run_sweep.py ARMS dict), e.g. hg_d512
#   SEEDS  comma list of seeds, e.g. 1,2,3,4
#   GPUS   comma list of local GPU ids, e.g. 0,1,2,3
#   EXTRA  extra flags passed through to run_sweep.py (e.g. --no-eval-at-end)
#   TAG    log tag (default: derived from ARMS)
set -uo pipefail
cd "$(dirname "$0")/.."                       # -> experiments/recogs
REPO_ROOT="$(cd ../.. && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p exp/v5_hg_scaling/runs/driver_logs exp/v5_hg_scaling/runs

: "${ARMS:?set ARMS}"; : "${SEEDS:?set SEEDS}"; : "${GPUS:?set GPUS}"
EXTRA="${EXTRA:-}"
TAG="${TAG:-${ARMS//,/_}}"
LOG="exp/v5_hg_scaling/runs/driver_logs/v5_${TAG}_driver.log"

{
  echo "=== v5 driver start $(date -Is) on $(hostname) | arms=$ARMS seeds=$SEEDS gpus=$GPUS extra='$EXTRA' ==="
  attempt=1; rc=1
  while [ "$attempt" -le 4 ]; do
    echo "--- run_sweep attempt $attempt/4 $(date -Is) ---"
    python -u run_sweep.py \
      --arms "$ARMS" --seeds "$SEEDS" --gpus "$GPUS" \
      --skip-existing --out-dir exp/v5_hg_scaling/runs \
      --max-seq-len 512 --batch-size 32 --grad-accum-steps 4 \
      --epochs 50 --early-stop-patience 0 --early-stop-min-delta 1e-4 \
      --eval-dev-every 100 --eval-batch-size 32 --eval-max-new-tokens 384 \
      --prefix-lm $EXTRA
    rc=$?
    if [ "$rc" -eq 0 ]; then echo "--- all runs complete (attempt $attempt) ---"; break; fi
    echo "--- attempt $attempt rc=$rc; some runs incomplete, retrying in 30s ---"
    sleep 30; attempt=$((attempt + 1))
  done
  echo "=== v5 driver done rc=$rc $(date -Is) ==="
} >> "$LOG" 2>&1
