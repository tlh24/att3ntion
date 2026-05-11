#!/usr/bin/bash

# 1. Define Default Variables
BATCH_SIZE=32 # same as the python script
EPOCHS=10
LOG_NAME="test" # Default value if no argument is provided
REPL=1 # what replicate this is

# This looks for --log-name and assigns the next argument to the variable
while [[ "$#" -gt 0 ]]; do
	case $1 in
		--log-name)
			LOG_NAME="$2"
			shift
			;;
		--repl)
			REPL="$2"
			shift
			;;
		*)
			echo "Unknown parameter passed: $1"
			exit 1
			;;
	esac
	shift
done

ATTN_TYPES=("hypergraph" "graph")
TASKS=(3 4 7)

for attn in "${ATTN_TYPES[@]}"; do
	for task in "${TASKS[@]}"; do
		echo "Running: Attn=$attn | Task=$task | Log=$LOG_NAME | Replicate=$REPL"

		python train.py --bf16 \
			--batch-size "$BATCH_SIZE" \
			--epochs "$EPOCHS" \
			--attn "$attn" \
			--task "$task" \
			--log-name "$LOG_NAME" \
			--repl "$REPL"

	done
done
