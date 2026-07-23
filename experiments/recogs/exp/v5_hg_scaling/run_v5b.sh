#!/usr/bin/env bash
# Run-5b: seed extension — 8 additional seeds for every 3L rung.
# Chains 4 sequential phases on a single 4-GPU pod (all phases share GPUS=0,1,2,3).
# Launch DETACHED from the repo root:
#
#   cd /home/dev/workspace/experiments/recogs
#   setsid nohup bash scripts/run_v5b.sh >/dev/null 2>&1 </dev/null & echo "pid $!"
#
# Individual phase logs: exp/v5_hg_scaling/runs/driver_logs/v5b_p{1..4}_driver.log
# Master log (phase transitions + final summary): exp/v5_hg_scaling/runs/driver_logs/v5b_master.log
#
# Phase  Arms                 Seeds       ~GPU-h  ~Wall (4 GPU)
#   1    hg_d128 + hg_d384   3-10        ~98     ~25h
#   2    hg_gather (d256)    11-18       ~49     ~12h
#   3    hg_d512             5-12        ~101    ~25h
#   4    hg_d768 (train-only)3-10        ~122    ~31h
# Total                                  ~370    ~93h (~4 days)
set -uo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$(cd ../.. && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p exp/v5_hg_scaling/runs/driver_logs exp/v5_hg_scaling/runs

GPUS="0,1,2,3"
MASTER_LOG="exp/v5_hg_scaling/runs/driver_logs/v5b_master.log"
PY="${PY:-python}"

run_phase() {
    local phase="$1" arms="$2" seeds="$3" extra="${4:-}"
    local log="exp/v5_hg_scaling/runs/driver_logs/v5b_p${phase}_driver.log"
    echo "=== Phase ${phase} start $(date -Is) | arms=${arms} seeds=${seeds} extra='${extra}' ===" | tee -a "$MASTER_LOG"
    local attempt=1 rc=1
    while [ "$attempt" -le 4 ]; do
        echo "--- phase ${phase} run_sweep attempt ${attempt}/4 $(date -Is) ---" >> "$log" 2>&1
        "$PY" -u run_sweep.py \
            --arms "$arms" --seeds "$seeds" --gpus "$GPUS" \
            --skip-existing --out-dir exp/v5_hg_scaling/runs \
            --max-seq-len 512 --batch-size 32 --grad-accum-steps 4 \
            --epochs 50 --early-stop-patience 0 --early-stop-min-delta 1e-4 \
            --eval-dev-every 100 --eval-batch-size 32 --eval-max-new-tokens 384 \
            --prefix-lm $extra \
            >> "$log" 2>&1
        rc=$?
        if [ "$rc" -eq 0 ]; then echo "--- phase ${phase} complete (attempt ${attempt}) ---" >> "$log"; break; fi
        echo "--- phase ${phase} attempt ${attempt} rc=${rc}; retrying in 30s ---" >> "$log"
        sleep 30; attempt=$((attempt + 1))
    done
    echo "=== Phase ${phase} done rc=${rc} $(date -Is) ===" | tee -a "$MASTER_LOG"
    return "$rc"
}

{
echo "=== v5b master start $(date -Is) on $(hostname) gpus=$GPUS ==="

# Phase 1: d128 + d384 (both use seeds 3-10, both with --eval-at-end)
run_phase 1 "hg_d128,hg_d384" "3,4,5,6,7,8,9,10" "" || echo "WARN: phase 1 had failures"

# Phase 2: d256 (hg_gather) — new seeds go to exp/v5_hg_scaling/runs alongside v3 seeds in exp/v3_prefix_lm/runs
run_phase 2 "hg_gather" "11,12,13,14,15,16,17,18" "" || echo "WARN: phase 2 had failures"

# Phase 3: d512 (seeds 1-4 exist, adding 5-12)
run_phase 3 "hg_d512" "5,6,7,8,9,10,11,12" "" || echo "WARN: phase 3 had failures"

# Phase 4: d768 train-only (seeds 1-2 exist and have final_metrics.json — skipped automatically)
run_phase 4 "hg_d768" "3,4,5,6,7,8,9,10" "--no-eval-at-end" || echo "WARN: phase 4 had failures"

echo "=== v5b master done $(date -Is) ==="
} >> "$MASTER_LOG" 2>&1
