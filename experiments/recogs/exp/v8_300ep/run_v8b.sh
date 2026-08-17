#!/usr/bin/env bash
# RUN_8b (v8b): 300-epoch PARAM-MATCHED graph arm — graph_flash at 3L/d870/h15/ffn2610
# (24.73M params ~= hg_d768's 24.79M, -0.3%; the RUN_7 param-match shape), DDP-4 per run.
# Fills the missing cell: v8's graph_d768 (19.5M) is shape-matched to hg_d768 but NOT
# param-matched, so this arm gives graph EQUAL params at 300 epochs for a fair A/B against
# v8's hg_d768_300ep (24.12% gen). Everything else is byte-identical to run_v8.sh's graph
# invocation: eff batch 128 (per-rank 32 x 4 ranks x accum 1), prefix-LM, seq 512,
# early-stop OFF, MNT 384, eval-at-end (DDP-sharded). Out-dir exp/v8_300ep/runs alongside v8 arms.
#
# One invocation runs a LIST of seeds sequentially on a fixed 4-GPU slice. Launch two
# invocations for 8 seeds across an 8-GPU pod (distinct GPUS + distinct PORT):
#   SEEDS=1,2,3,4 GPUS=0,1,2,3 PORT=29500 setsid nohup bash scripts/run_v8b.sh &
#   SEEDS=5,6,7,8 GPUS=4,5,6,7 PORT=29600 setsid nohup bash scripts/run_v8b.sh &
#
# Env: SEEDS (req, comma list), GPUS (req, 4 ids), PORT (rdzv, default 29500 — MUST differ
# between concurrent invocations on the same pod), EPOCHS (default 300), SMOKE=1 gate.
set -uo pipefail
cd "$(dirname "$0")/.."                       # -> experiments/recogs
REPO_ROOT="$(cd ../.. && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p exp/v8_300ep/runs/driver_logs exp/v8_300ep/runs

: "${SEEDS:?set SEEDS (comma list)}"; : "${GPUS:?set GPUS (comma list of 4 ids)}"
PORT="${PORT:-29500}"
EPOCHS="${EPOCHS:-300}"
NAME=graph_d870_300ep
ARM=graph_flash
IFS=',' read -ra _G <<< "$GPUS"; NPROC=${#_G[@]}

for SEED in ${SEEDS//,/ }; do
  LOG="exp/v8_300ep/runs/driver_logs/v8b_${NAME}_s${SEED}.log"
  if [ -f "exp/v8_300ep/runs/${NAME}_${ARM}_s${SEED}/final_metrics.json" ]; then
    echo "=== v8b skip ${NAME} s${SEED}: final_metrics exists ===" >> "$LOG"; continue
  fi
  if [ "${SMOKE:-0}" = "1" ]; then
    EXTRA="--epochs 1 --max-steps 30 --eval-dev-every 15 --eval-max-new-tokens 32 --eval-max-examples 64"
  else
    EXTRA="--epochs $EPOCHS"
  fi
  {
    echo "=== v8b $ARM seed=$SEED gpus=$GPUS nproc=$NPROC port=$PORT epochs=$EPOCHS start $(date -Is) ==="
    CUDA_VISIBLE_DEVICES="$GPUS" torchrun --nproc_per_node="$NPROC" \
      --rdzv-backend=c10d --rdzv-endpoint="localhost:$PORT" \
      train.py \
      --attn "$ARM" --seed "$SEED" --layers 3 --d-model 870 --heads 15 --ffn-hidden 2610 \
      --max-seq-len 512 --batch-size 32 --grad-accum-steps 1 \
      --early-stop-patience 0 --early-stop-min-delta 1e-4 \
      --eval-dev-every 100 --eval-dev-max-batches 0 \
      --eval-batch-size 32 --eval-max-new-tokens 384 --ckpt-every-epochs 0 \
      --log-name "$NAME" --out-dir exp/v8_300ep/runs --eval-at-end --prefix-lm $EXTRA
    echo "=== v8b $ARM seed=$SEED done rc=$? $(date -Is) ==="
  } >> "$LOG" 2>&1
done
echo "=== v8b driver (seeds $SEEDS) complete $(date -Is) ===" >> "exp/v8_300ep/runs/driver_logs/v8b_driver_${PORT}.log"
