#!/usr/bin/bash

set -uo pipefail   # no -e: we don't want one failing child to kill a parallel batch

cd "$(dirname "$0")"

BATCH_SIZE=32
EPOCHS=10
LOG_NAME="comparison"
NREPL=3
HIDDEN=256
HEADS=8
PARALLEL=0
MAX_PARALLEL=0   # 0 = no cap (background all at once when --parallel is set)
ATTNS="hypergraph,hypergraph_scatter,tree,strassen,tensor"
TASKS="3,4,7"

while [[ "$#" -gt 0 ]]; do
	case $1 in
		--log-name)    LOG_NAME="$2"; shift ;;
		--nrepl)       NREPL="$2"; shift ;;
		--epochs)      EPOCHS="$2"; shift ;;
		--batch-size)  BATCH_SIZE="$2"; shift ;;
		--hidden)      HIDDEN="$2"; shift ;;
		--heads)       HEADS="$2"; shift ;;
		--attns)       ATTNS="$2"; shift ;;
		--tasks)       TASKS="$2"; shift ;;
		--parallel)    PARALLEL=1 ;;
		--max-parallel) MAX_PARALLEL="$2"; PARALLEL=1; shift ;;
		-h|--help)
			cat <<EOF
Usage: $0 [options]
  --log-name STR    Tag appended to losslog filenames (default: $LOG_NAME)
  --nrepl    INT    Number of replicates per (attn, task) cell (default: $NREPL)
  --epochs   INT    Epochs per run (default: $EPOCHS)
  --batch-size INT  Batch size (default: $BATCH_SIZE)
  --hidden   INT    Hidden dim (default: $HIDDEN)
  --heads    INT    Number of attention heads (default: $HEADS)
  --attns    CSV    Comma-separated attn impls (default: $ATTNS)
  --tasks    CSV    Comma-separated task ids (default: $TASKS)
  --parallel        Within each task, launch all (attn x replicate) cells
                    concurrently in the background, then wait before moving
                    to the next task. Outputs are line-prefixed for clarity.
  --max-parallel N  Throttle parallel mode to at most N concurrent runs at a
                    time (default 0 = no cap). Use to avoid CPU-RAM OOM when
                    each process needs hundreds of MB just to load the dataset.
                    Implies --parallel.
EOF
			exit 0 ;;
		*) echo "Unknown parameter: $1"; exit 1 ;;
	esac
	shift
done

IFS=',' read -ra ATTN_ARR <<< "$ATTNS"
IFS=',' read -ra TASK_ARR <<< "$TASKS"

N_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
N_GPUS=$(( N_GPUS < 1 ? 1 : N_GPUS ))

echo "===================================================================="
echo "comparison sweep:"
echo "  attns:      ${ATTN_ARR[*]}"
echo "  tasks:      ${TASK_ARR[*]}"
echo "  replicates: 1..$NREPL"
echo "  hidden:     $HIDDEN  heads: $HEADS"
echo "  epochs:     $EPOCHS  batch_size: $BATCH_SIZE"
echo "  log-name:   $LOG_NAME"
echo "  parallel:   $PARALLEL  (max concurrent: $MAX_PARALLEL, 0=uncapped)"
echo "  losslogs -> $(pwd)/losslogs/"
echo "===================================================================="

run_one() {
	local attn="$1" task="$2" r="$3"
	python train.py \
		--batch-size "$BATCH_SIZE" \
		--epochs "$EPOCHS" \
		--hidden "$HIDDEN" \
		--heads "$HEADS" \
		--attn "$attn" \
		--task "$task" \
		--log-name "$LOG_NAME" \
		--repl "$r"
}

trap 'echo "[run_comparison] caught SIGINT, killing background jobs"; kill 0 2>/dev/null; exit 130' INT TERM

if [[ "$PARALLEL" == "1" ]]; then
	for task in "${TASK_ARR[@]}"; do
		echo ""
		if [[ "$MAX_PARALLEL" -gt 0 ]]; then
			echo "==== task $task: ${#ATTN_ARR[@]} attns x $NREPL replicates, throttled to $MAX_PARALLEL concurrent ===="
		else
			echo "==== task $task: launching ${#ATTN_ARR[@]} attns x $NREPL replicates concurrently ===="
		fi
		running=0
		job_idx=0
		for attn in "${ATTN_ARR[@]}"; do
			for r in $(seq 1 "$NREPL"); do
				tag="t${task}_${attn}_r${r}"
				gpu=$(( job_idx % N_GPUS ))
				echo ">>> backgrounding $tag (GPU $gpu)"
				( CUDA_VISIBLE_DEVICES=$gpu run_one "$attn" "$task" "$r" 2>&1 | sed -u "s|^|[$tag] |" ) &
				running=$((running + 1))
				job_idx=$((job_idx + 1))
				if [[ "$MAX_PARALLEL" -gt 0 ]] && [[ "$running" -ge "$MAX_PARALLEL" ]]; then
					wait -n
					running=$((running - 1))
				fi
			done
		done
		wait
		echo "==== task $task: all parallel runs complete ===="
	done
else
	job_idx=0
	for r in $(seq 1 "$NREPL"); do
		for task in "${TASK_ARR[@]}"; do
			for attn in "${ATTN_ARR[@]}"; do
				gpu=$(( job_idx % N_GPUS ))
				echo ""
				echo ">>> attn=$attn  task=$task  repl=$r  log=$LOG_NAME  GPU=$gpu"
				CUDA_VISIBLE_DEVICES=$gpu run_one "$attn" "$task" "$r"
				job_idx=$((job_idx + 1))
			done
		done
	done
fi

echo ""
echo "===================================================================="
echo "sweep complete. logs in $(pwd)/losslogs/"
echo "===================================================================="
