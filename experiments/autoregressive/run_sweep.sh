#!/usr/bin/bash

cleanup() {
	echo -e "\nCaught Ctrl-C! Shutting down all parallel jobs..."
	# Grab the Process IDs (PIDs) of all running background jobs
	pids=$(jobs -p)

	# If there are background jobs running, kill them
	if [ -n "$pids" ]; then
		kill $pids 2>/dev/null
	fi

	echo "All jobs terminated. Exiting."
	# Exit the script
	exit 1
}

# 2. Trap SIGINT (Ctrl-C) and SIGTERM (kill), and route them to 'cleanup'
trap cleanup SIGINT SIGTERM

for seq in 1 2 4 8; do
	for attn in hypergraph ; do
		for repl in 1 2 3; do
			python train.py --attn "$attn" --repl "$repl" --log-name nogelu2 --seq-l "$seq" &
		done
		wait
	done
done
