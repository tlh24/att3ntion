#!/usr/bin/env bash
# Durable overnight driver for the ReCOGS 3-arm ladder sweep.
#
# Designed to be launched DETACHED so it survives the actl attach session /
# SSH disconnect, e.g.:
#
#   actl pod attach recogs -- bash -lc \
#     'cd /home/dev/workspace/experiments/recogs && \
#      TAG=recogs GPUS=0,1,2,3 ARMS=graph_flash,hg_gather,hg_full SEEDS=1,2,3 \
#      setsid nohup bash run_overnight.sh > /dev/null 2>&1 < /dev/null & echo started pid $!'
#
# It runs run_sweep.py and RETRIES (with --skip-existing) so a transient
# per-run crash (e.g. an OOM on one GPU) doesn't lose the whole night: each
# retry only re-launches (arm,seed) runs that still lack final_metrics.json.
#
# Configure via env vars (all optional):
#   TAG           label for the driver log + pid file        (default: overnight)
#   GPUS          comma-sep GPU ids                           (default: 0,1,2,3)
#   ARMS          comma-sep arm names                         (default: graph_flash,hg_gather,hg_full)
#   SEEDS         comma-sep seeds                             (default: 1,2,3)
#   MAX_ATTEMPTS  sweep retries before giving up              (default: 3)
#   SEQ_LEN BATCH ACCUM EPOCHS PATIENCE EVAL_DEV_EVERY MNT    (regime knobs)
#   EXTRA         extra args passed straight to run_sweep.py
set -uo pipefail
cd "$(dirname "$0")"                       # -> experiments/recogs
REPO_ROOT="$(cd ../.. && pwd)"             # -> repo root (has the att3ntion/ package)
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

TAG=${TAG:-overnight}
GPUS=${GPUS:-0,1,2,3}
ARMS=${ARMS:-graph_flash,hg_gather,hg_full}
SEEDS=${SEEDS:-1,2,3}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}

SEQ_LEN=${SEQ_LEN:-512}
BATCH=${BATCH:-32}
ACCUM=${ACCUM:-4}
EPOCHS=${EPOCHS:-60}
PATIENCE=${PATIENCE:-60}
MIN_DELTA=${MIN_DELTA:-1e-4}
EVAL_DEV_EVERY=${EVAL_DEV_EVERY:-100}
MNT=${MNT:-512}
EXTRA=${EXTRA:-}

LOG_DIR=exp/v2_three_arm/runs/driver_logs
mkdir -p "$LOG_DIR"
DRIVER_LOG="$LOG_DIR/overnight_${TAG}.log"
PID_FILE="$LOG_DIR/overnight_${TAG}.pid"
echo $$ > "$PID_FILE"

{
  echo "=== overnight driver TAG=$TAG started $(date -Is) ==="
  echo "host=$(hostname) gpus=$GPUS arms=$ARMS seeds=$SEEDS"
  echo "regime: seq_len=$SEQ_LEN batch=$BATCH accum=$ACCUM (eff $((BATCH*ACCUM))) epochs=$EPOCHS patience=$PATIENCE min_delta=$MIN_DELTA eval_dev_every=$EVAL_DEV_EVERY"
  echo "PYTHONPATH=$PYTHONPATH  python=$(command -v python)"

  attempt=1
  rc=1
  while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
    echo "--- sweep attempt $attempt/$MAX_ATTEMPTS $(date -Is) ---"
    python -u run_sweep.py \
      --arms "$ARMS" --seeds "$SEEDS" --gpus "$GPUS" --skip-existing \
      --max-seq-len "$SEQ_LEN" --batch-size "$BATCH" --grad-accum-steps "$ACCUM" \
      --epochs "$EPOCHS" --early-stop-patience "$PATIENCE" \
      --early-stop-min-delta "$MIN_DELTA" \
      --eval-dev-every "$EVAL_DEV_EVERY" --eval-max-new-tokens "$MNT" $EXTRA
    rc=$?
    if [ "$rc" -eq 0 ]; then
      echo "--- all runs have final_metrics.json (attempt $attempt) ---"
      break
    fi
    echo "--- attempt $attempt exited rc=$rc; some runs incomplete, retrying after 30s ---"
    sleep 30
    attempt=$((attempt + 1))
  done

  echo "=== overnight driver TAG=$TAG finished rc=$rc $(date -Is) ==="
} >> "$DRIVER_LOG" 2>&1
