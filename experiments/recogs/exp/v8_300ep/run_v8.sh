#!/usr/bin/env bash
# RUN_8 (v8): 300-epoch d768 A/B — hypergraph vs graph at 3L/d768, DDP-4 per run.
# One invocation drives ONE seed through BOTH arms sequentially (hg first, then
# graph) on a fixed 4-GPU slice, so phase 2 launches automatically per pod.
# Regime matches the ladder except epochs: eff batch 128 (per-rank 32 x 4 ranks
# x accum 1), prefix-LM, seq 512, early-stop OFF, MNT 384, eval-at-end (DDP-sharded).
#
#   SEED=1 GPUS=0,1,2,3 PORT=29500 setsid nohup bash scripts/run_v8.sh &
#
# Env: SEED (req), GPUS (req, 4 ids), PORT (rdzv port, default 29500 — MUST
# differ between concurrent invocations on the same pod), ARMS (default both),
# EPOCHS (default 300), SMOKE=1 for a 30-step correctness gate.
set -uo pipefail
cd "$(dirname "$0")/.."                       # -> experiments/recogs
REPO_ROOT="$(cd ../.. && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p exp/v8_300ep/runs/driver_logs exp/v8_300ep/runs

: "${SEED:?set SEED}"; : "${GPUS:?set GPUS (comma list of 4 ids)}"
PORT="${PORT:-29500}"
EPOCHS="${EPOCHS:-300}"
ARMS="${ARMS:-hypergraph_cuda,graph_flash}"
IFS=',' read -ra _G <<< "$GPUS"; NPROC=${#_G[@]}

for ARM in ${ARMS//,/ }; do
  case "$ARM" in
    hypergraph_cuda) NAME=hg_d768_300ep ;;
    graph_flash)     NAME=graph_d768_300ep ;;
    *)               NAME="${ARM}_d768_300ep" ;;
  esac
  LOG="exp/v8_300ep/runs/driver_logs/v8_${NAME}_s${SEED}.log"
  if [ -f "exp/v8_300ep/runs/${NAME}_${ARM}_s${SEED}/final_metrics.json" ]; then
    echo "=== v8 skip ${NAME} s${SEED}: final_metrics exists ===" >> "$LOG"; continue
  fi
  if [ "${SMOKE:-0}" = "1" ]; then
    EXTRA="--epochs 1 --max-steps 30 --eval-dev-every 15 --eval-max-new-tokens 32 --eval-max-examples 64"
  else
    EXTRA="--epochs $EPOCHS"
  fi
  {
    echo "=== v8 $ARM seed=$SEED gpus=$GPUS nproc=$NPROC port=$PORT epochs=$EPOCHS start $(date -Is) ==="
    CUDA_VISIBLE_DEVICES="$GPUS" torchrun --nproc_per_node="$NPROC" \
      --rdzv-backend=c10d --rdzv-endpoint="localhost:$PORT" \
      train.py \
      --attn "$ARM" --seed "$SEED" --layers 3 --d-model 768 --heads 12 --ffn-hidden 2304 \
      --max-seq-len 512 --batch-size 32 --grad-accum-steps 1 \
      --early-stop-patience 0 --early-stop-min-delta 1e-4 \
      --eval-dev-every 100 --eval-dev-max-batches 0 \
      --eval-batch-size 32 --eval-max-new-tokens 384 --ckpt-every-epochs 0 \
      --log-name "$NAME" --out-dir exp/v8_300ep/runs --eval-at-end --prefix-lm $EXTRA
    echo "=== v8 $ARM seed=$SEED done rc=$? $(date -Is) ==="
  } >> "$LOG" 2>&1
done
echo "=== v8 driver (seed $SEED) complete $(date -Is) ===" >> "exp/v8_300ep/runs/driver_logs/v8_driver_s${SEED}.log"
