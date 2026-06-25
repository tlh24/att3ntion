#!/bin/bash

# 1. Parse command line arguments
START_REPL=""
while [[ "$#" -gt 0 ]]; do
	case $1 in
		--repl) START_REPL="$2"; shift ;;
		*) echo "Unknown parameter passed: $1"; exit 1 ;;
	esac
	shift
done

if [ -z "$START_REPL" ]; then
	echo "Error: --repl argument is required."
	echo "Usage: $0 --repl <starting_replicate_number>"
	exit 1
fi

REPL1=$START_REPL
REPL2=$((START_REPL + 1))
REPL3=$((START_REPL + 2))

# Clean up gracefully on Ctrl+C
trap 'echo -e "\nAborting... stopping all remote runs."; kill $(jobs -p) 2>/dev/null; exit' SIGINT SIGTERM

mkdir -p run_logs

LOG_FILES=(
	"run_logs/m1_graph_r${REPL1}.log"
	"run_logs/m1_hyper_r${REPL1}.log"
	"run_logs/m2_graph_r${REPL2}.log"
	"run_logs/m2_hyper_r${REPL2}.log"
	"run_logs/m2_graph_r${REPL3}.log"
	"run_logs/m2_hyper_r${REPL3}.log"
)
touch "${LOG_FILES[@]}"

PRE_CMD="cd ~ && source go.sh"
PY_CMD="python -u train.py --task 3 --epochs 100 --log-name d2r2"

echo "Launching 6 parallel tasks across the two machines starting at replicate ${START_REPL}..."

# Array to keep track of the SSH background processes
SSH_PIDS=()

# =========================================================
# MACHINE 1: "neuronizer" (1 GPU -> Handles Replicate $REPL1)
# =========================================================
ssh neuronizer "$PRE_CMD && CUDA_VISIBLE_DEVICES=0 $PY_CMD --attn graph --repl $REPL1" \
	> "${LOG_FILES[0]}" 2>&1 &
SSH_PIDS+=($!)

ssh neuronizer "$PRE_CMD && CUDA_VISIBLE_DEVICES=0 $PY_CMD --attn hypergraph --repl $REPL1" \
	> "${LOG_FILES[1]}" 2>&1 &
SSH_PIDS+=($!)


# =========================================================
# MACHINE 2: "ashtray" (2 GPUs -> Handles Replicates $REPL2 & $REPL3)
# =========================================================
# --- GPU 0 (Replicate $REPL2) ---
ssh ashtray "$PRE_CMD && CUDA_VISIBLE_DEVICES=0 $PY_CMD --attn graph --repl $REPL2" \
	> "${LOG_FILES[2]}" 2>&1 &
SSH_PIDS+=($!)

ssh ashtray "$PRE_CMD && CUDA_VISIBLE_DEVICES=0 $PY_CMD --attn hypergraph --repl $REPL2" \
	> "${LOG_FILES[3]}" 2>&1 &
SSH_PIDS+=($!)

# --- GPU 1 (Replicate $REPL3) ---
ssh ashtray "$PRE_CMD && CUDA_VISIBLE_DEVICES=1 $PY_CMD --attn graph --repl $REPL3" \
	> "${LOG_FILES[4]}" 2>&1 &
SSH_PIDS+=($!)

ssh ashtray "$PRE_CMD && CUDA_VISIBLE_DEVICES=1 $PY_CMD --attn hypergraph --repl $REPL3" \
	> "${LOG_FILES[5]}" 2>&1 &
SSH_PIDS+=($!)


echo "All 6 tasks launched successfully!"
echo "Streaming logs below. (Press Ctrl+C to safely kill all tasks at once)"
echo "---------------------------------------------------------------------"

# 4. Stream logs IN THE BACKGROUND
tail -f "${LOG_FILES[@]}" &
TAIL_PID=$!

# 5. Wait for all 6 SSH background jobs to finish
wait "${SSH_PIDS[@]}"

# 6. Clean up: Give tail a second to flush the very last output, then kill it
sleep 2
kill $TAIL_PID 2>/dev/null

echo "====================================================================="
echo "✅ All 6 training runs have completed successfully!"
