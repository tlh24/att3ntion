#!/usr/bin/env bash
# DDP launcher: train both attention arms sequentially, each using all GPUs on
# the node via torchrun. Keeps global batch = per-rank batch * world_size, so a
# 4-GPU run matches a 1-GPU batch-64 run (16 * 4) in optimizer dynamics while
# finishing ~world_size times faster.
#
# Usage (on a 4xH100 actl pod, from anywhere):
#   bash experiments/recogs/run_ddp.sh                 # full run, both arms
#   SMOKE=1 bash experiments/recogs/run_ddp.sh         # 50-step correctness smoke
#   NPROC=2 bash experiments/recogs/run_ddp.sh         # override GPU count
set -euo pipefail
cd "$(dirname "$0")"

NPROC=${NPROC:-4}
SEED=${SEED:-1}
BATCH=${BATCH:-16}                  # per-rank micro-batch; global = BATCH * NPROC
EPOCHS=${EPOCHS:-100}               # cap; early-stop usually ends sooner
PATIENCE=${PATIENCE:-10}            # dev evals without improvement before stopping
EVAL_DEV_EVERY=${EVAL_DEV_EVERY:-400}   # ~once per epoch at global batch 64
ARMS=${ARMS:-"hypergraph_cuda graph_flash"}
LOG_DIR=sweep_logs
mkdir -p "$LOG_DIR"

if [[ "${SMOKE:-0}" == "1" ]]; then
  # Tiny correctness gate: confirms DDP init, hypergraph custom-op backward,
  # falling loss, and dev-gen EM climbing above 0% (validates the loss-shift fix).
  LOG_NAME=${LOG_NAME:-smoke_ddp}
  EXTRA="--max-steps 50 --eval-dev-every 25 --early-stop-patience 0 \
         --eval-at-end --eval-gen-every 50 --eval-gen-max-examples 256"
  export ATT3NTION_CHECK_GRADS=1
else
  LOG_NAME=${LOG_NAME:-ddp4}
  EXTRA="--epochs $EPOCHS --early-stop-patience $PATIENCE \
         --eval-dev-every $EVAL_DEV_EVERY --eval-dev-max-batches 0 --eval-at-end"
fi

for ARM in $ARMS; do
  echo "=== training $ARM (seed=$SEED, nproc=$NPROC, global_batch=$((BATCH * NPROC))) ==="
  torchrun --standalone --nproc_per_node="$NPROC" train.py \
    --attn "$ARM" \
    --seed "$SEED" \
    --batch-size "$BATCH" \
    --log-name "$LOG_NAME" \
    $EXTRA \
    2>&1 | tee "$LOG_DIR/${LOG_NAME}_${ARM}_s${SEED}.log"
done

echo "=== run_ddp.sh complete: arms=[$ARMS] ==="
