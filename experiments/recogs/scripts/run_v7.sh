#!/usr/bin/env bash
# RUN_7 (v7) graph_flash width-ladder BASELINE sweep — generic per-pod driver.
# Identical regime to RUN_5's hg ladder (50 epochs, early-stop OFF, prefix-LM,
# seq 512, batch 32x4, MNT 384); only the attention mechanism differs. Detached,
# crash-retrying, --skip-existing (safe to re-run). Launch DETACHED, e.g.:
#
#   ARMS=graph_d768,graph_d512 SEEDS=1,2,3,4,5,6,7,8,9,10 GPUS=0,1,2,3,4,5,6,7 \
#     setsid nohup bash scripts/run_v7.sh >/dev/null 2>&1 </dev/null & echo pid $!
#
# Env knobs (all optional except ARMS/SEEDS/GPUS):
#   ARMS   comma list of arm names (run_sweep.py ARMS dict), e.g. graph_d512
#   SEEDS  comma list of seeds, e.g. 1,2,...,10
#   GPUS   comma list of local GPU ids
#   EXTRA  extra flags passed through to run_sweep.py
#   TAG    log tag (default: derived from ARMS)
set -uo pipefail
cd "$(dirname "$0")/.."                       # -> experiments/recogs
REPO_ROOT="$(cd ../.. && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p sweep_logs runs/v7

: "${ARMS:?set ARMS}"; : "${SEEDS:?set SEEDS}"; : "${GPUS:?set GPUS}"
EXTRA="${EXTRA:-}"
TAG="${TAG:-${ARMS//,/_}}"
LOG="sweep_logs/v7_${TAG}_driver.log"

{
  echo "=== v7 driver start $(date -Is) on $(hostname) | arms=$ARMS seeds=$SEEDS gpus=$GPUS extra='$EXTRA' ==="
  attempt=1; rc=1
  while [ "$attempt" -le 4 ]; do
    echo "--- run_sweep attempt $attempt/4 $(date -Is) ---"
    python -u run_sweep.py \
      --arms "$ARMS" --seeds "$SEEDS" --gpus "$GPUS" \
      --skip-existing --out-dir runs/v7 \
      --max-seq-len 512 --batch-size 32 --grad-accum-steps 4 \
      --epochs 50 --early-stop-patience 0 --early-stop-min-delta 1e-4 \
      --eval-dev-every 100 --eval-batch-size 32 --eval-max-new-tokens 384 \
      --prefix-lm $EXTRA
    rc=$?
    if [ "$rc" -eq 0 ]; then echo "--- all runs complete (attempt $attempt) ---"; break; fi
    echo "--- attempt $attempt rc=$rc; some runs incomplete, retrying in 30s ---"
    sleep 30; attempt=$((attempt + 1))
  done
  echo "=== v7 driver done rc=$rc $(date -Is) ==="
} >> "$LOG" 2>&1
