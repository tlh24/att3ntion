#!/usr/bin/env bash
# Usage: scripts/profile.sh <kernel> <tag> [pass] [dims]
#   kernel : kernel name, required (e.g. QS_grad_kernel)
#   tag    : short slug, required (e.g. baseline, after_float4)
#   pass   : forward | backward, default forward
#   dims   : B,H,I,J,K,D, default 1,2,128,128,128,64
#
# Output files (same basename in both):
#   profiling_reports/ncu_rep/<kernel>_<tag>_<mon><day>_<HH:MM><am|pm>.ncu-rep
#   profiling_reports/csv/<kernel>_<tag>_<mon><day>_<HH:MM><am|pm>.csv

set -euo pipefail

KERNEL="${1:?usage: scripts/profile.sh <kernel> <tag> [pass] [dims]}"
TAG="${2:?tag is required (e.g. baseline, after_float4)}"
PASS="${3:-forward}"
DIMS="${4:-1,2,128,128,128,64}"

if [ "$PASS" != "forward" ] && [ "$PASS" != "backward" ] && [ "$PASS" != "both" ]; then
    echo "pass must be 'forward', 'backward', or 'both' (got: $PASS)" >&2
    exit 2
fi

if [ -f myenv/bin/activate ]; then
    # shellcheck disable=SC1091
    source myenv/bin/activate
fi

SO=$(find . -name "_cuda_kernels*.so" 2>/dev/null | head -1)
if [ -z "$SO" ]; then
    echo "No compiled extension found — building."
    make build
elif [ -n "$(find cuda \( -name '*.cu' -o -name '*.cuh' \) -newer "$SO" 2>/dev/null)" ]; then
    echo "Sources newer than .so — rebuilding."
    make build
else
    echo "Build is fresh — skipping rebuild."
fi

STAMP=$(date +'%b%d_%I:%M%P' | tr '[:upper:]' '[:lower:]')
BASENAME="${KERNEL}_${TAG}_${STAMP}"

mkdir -p profiling_reports/ncu_rep profiling_reports/csv

NCU_OUT="profiling_reports/ncu_rep/${BASENAME}"
CSV_OUT="profiling_reports/csv/${BASENAME}.csv"

KERNEL_FLAG=()
[ "$KERNEL" != "all" ] && KERNEL_FLAG=(-k "$KERNEL")

PASS_FLAG=()
[ "$PASS" != "both" ] && PASS_FLAG=("--${PASS}-only")

ncu --set full --csv --page raw \
    "${KERNEL_FLAG[@]}" \
    -o "$NCU_OUT" -f \
    python benchmarks/profile_kernels.py --no-profile \
        --dims "$DIMS" "${PASS_FLAG[@]}" \
  > "$CSV_OUT"

echo "NCU: $(realpath "${NCU_OUT}.ncu-rep")"
echo "CSV: $(realpath "$CSV_OUT")"
