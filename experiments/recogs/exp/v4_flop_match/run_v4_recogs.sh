#!/usr/bin/env bash
# RUN_4 (v4) unattended driver, recogs-only. Launch DETACHED so it survives the
# actl session / laptop disconnect:
#   actl -n strange-loop pod attach recogs -- bash -lc 'cd /home/dev/workspace/experiments/recogs && \
#     setsid nohup bash scripts/run_v4_recogs.sh >/dev/null 2>&1 </dev/null & echo pid $!'
#
# Calls run_sweep.py directly (not run_overnight.sh) with its own crash-retry, so
# it is independent of where run_overnight.sh lives. run_sweep.py manages the
# 4-GPU job queue and --skip-existing makes retries resume cleanly.
#
# 1. trains graph_flop6L (FLOP-matched) + graph_6L256 (intermediate), 10 seeds
#    each, across all 4 GPUs.
# 2. when done (or retries exhausted), plots graph (FLOP-matched) vs hg using
#    whatever runs completed.
set -uo pipefail
cd "$(dirname "$0")/.."                       # -> experiments/recogs
REPO_ROOT="$(cd ../.. && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p exp/v4_flop_match/runs/driver_logs exp/v4_flop_match/runs
LOG=exp/v4_flop_match/runs/driver_logs/v4_driver.log

{
  echo "=== v4 driver start $(date -Is) on $(hostname) ==="

  attempt=1; rc=1
  while [ "$attempt" -le 4 ]; do
    echo "--- run_sweep attempt $attempt/4 $(date -Is) ---"
    python -u run_sweep.py \
      --arms graph_flop6L,graph_6L256 \
      --seeds 1,2,3,4,5,6,7,8,9,10 \
      --gpus 0,1,2,3 --skip-existing \
      --out-dir exp/v4_flop_match/runs \
      --max-seq-len 512 --batch-size 32 --grad-accum-steps 4 \
      --epochs 50 --early-stop-patience 60 --early-stop-min-delta 1e-4 \
      --eval-dev-every 100 --eval-batch-size 32 --eval-max-new-tokens 384 \
      --prefix-lm
    rc=$?
    if [ "$rc" -eq 0 ]; then echo "--- all runs complete (attempt $attempt) ---"; break; fi
    echo "--- attempt $attempt rc=$rc; some runs incomplete, retrying in 30s ---"
    sleep 30; attempt=$((attempt + 1))
  done
  echo "=== sweep finished rc=$rc $(date -Is) ==="

  python -c "import matplotlib, numpy" 2>/dev/null || pip install -q matplotlib numpy
  echo "--- plotting graph vs hg ---"
  python plots/plot_v4_graph_vs_hg.py || echo "PLOT FAILED — data is in exp/v4_flop_match/runs, re-run plots/plot_v4_graph_vs_hg.py"

  echo "=== v4 driver done $(date -Is) ==="
} >> "$LOG" 2>&1
