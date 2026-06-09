#!/usr/bin/bash
# Sweep: hypergraph (gather-only) + hypergraph (gather + scatter) vs.
# tree / strassen / tensor polyattention variants, on compositional tasks
# 3, 4, 7, across N replicates. All variants share the same 1/sqrt(d)
# scaling, the same data per (task, replicate), and the same architecture
# scaffolding (RMSNorm, FFN, residuals, output proj).
#
# Loss logs land in experiments/compositional/losslogs/ regardless of where
# this script is invoked from (train.py resolves the dir from __file__).
#
# Usage:
#   ./run_comparison.sh                                                # all tasks, sequential
#   ./run_comparison.sh --tasks 3 --parallel                           # task 3 only, attns × replicates in parallel
#   ./run_comparison.sh --tasks 3 --nrepl 5 --parallel --epochs 20

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
# Comma-separated list of attentions and tasks; override on the cli if needed.
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

# Forward Ctrl-C from the foreground bash to every backgrounded python.
trap 'echo "[run_comparison] caught SIGINT, killing background jobs"; kill 0 2>/dev/null; exit 130' INT TERM

if [[ "$PARALLEL" == "1" ]]; then
	# Within each task, fan out (attn x replicate) cells concurrently.
	# When --max-parallel N is set, use `wait -n` to keep at most N workers alive
	# at any time — this prevents CPU-RAM OOM kills when launching e.g. 25 procs
	# at once, each of which allocates the full training dataset before GPU upload.
	# Per-line prefix '[tT_attn_rN]' lets you grep interleaved output.
	# Loss curves are unaffected by GPU contention (each process has its own
	# deterministic seed); only wall-clock batch times will be perturbed.
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
					# Block until any one of the running jobs finishes, then continue
					wait -n
					running=$((running - 1))
				fi
			done
		done
		wait
		echo "==== task $task: all parallel runs complete ===="
	done
else
	# Iteration order: replicate (outer), task, attn (inner). Gives you a full
	# pass over every (task, attn) cell at replicate 1 first, so live_plot.py
	# shows trends before the full sweep finishes.
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
