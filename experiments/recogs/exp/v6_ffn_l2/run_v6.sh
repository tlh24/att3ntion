#!/usr/bin/env bash
# RUN_6 (v6) FFN pre-activation L2 regularizer sweep — generic per-pod driver.
# Detached, crash-retrying, --skip-existing (safe to re-run). Mirrors run_v5.sh but
# writes to exp/v6_ffn_l2/runs. See RUN_6.md. Launch DETACHED, e.g.:
#
#   cd /home/dev/workspace/experiments/recogs && \
#   ARMS=hg_d256_l2 SEEDS=1,2,3,4,5,6,7,8 GPUS=0,1,2,3 \
#   setsid nohup bash scripts/run_v6.sh >/dev/null 2>&1 </dev/null & echo pid $!
#
# Env knobs (all optional except ARMS/SEEDS/GPUS):
#   ARMS   comma list of arm names (run_sweep.py ARMS dict), e.g. hg_d256_l2
#   SEEDS  comma list of seeds, e.g. 1,2,3,4,5,6,7,8
#   GPUS   comma list of local GPU ids, e.g. 0,1,2,3
#   EXTRA  extra flags passed through to run_sweep.py
#   TAG    log tag (default: derived from ARMS)
set -uo pipefail
cd "$(dirname "$0")/.."                       # -> experiments/recogs
REPO_ROOT="$(cd ../.. && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p exp/v6_ffn_l2/runs/driver_logs exp/v6_ffn_l2/runs

: "${ARMS:?set ARMS}"; : "${SEEDS:?set SEEDS}"; : "${GPUS:?set GPUS}"
EXTRA="${EXTRA:-}"
TAG="${TAG:-${ARMS//,/_}}"
LOG="exp/v6_ffn_l2/runs/driver_logs/v6_${TAG}_driver.log"

{
  echo "=== v6 driver start $(date -Is) on $(hostname) | arms=$ARMS seeds=$SEEDS gpus=$GPUS extra='$EXTRA' ==="
  attempt=1; rc=1
  while [ "$attempt" -le 4 ]; do
    echo "--- run_sweep attempt $attempt/4 $(date -Is) ---"
    python -u run_sweep.py \
      --arms "$ARMS" --seeds "$SEEDS" --gpus "$GPUS" \
      --skip-existing --out-dir exp/v6_ffn_l2/runs \
      --max-seq-len 512 --batch-size 32 --grad-accum-steps 4 \
      --epochs 50 --early-stop-patience 0 --early-stop-min-delta 1e-4 \
      --eval-dev-every 100 --eval-batch-size 32 --eval-max-new-tokens 384 \
      --prefix-lm $EXTRA
    rc=$?
    if [ "$rc" -eq 0 ]; then echo "--- all runs complete (attempt $attempt) ---"; break; fi
    echo "--- attempt $attempt rc=$rc; some runs incomplete, retrying in 30s ---"
    sleep 30; attempt=$((attempt + 1))
  done
  echo "=== v6 driver done rc=$rc $(date -Is) ==="
} >> "$LOG" 2>&1
