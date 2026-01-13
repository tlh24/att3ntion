#!/bin/bash
# Script to run the CUDA vs PyTorch comparison
# Usage: ./run_comparison.sh [options]

set -e

# Get the directory where this script is located and the project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root so relative paths work
cd "$PROJECT_ROOT"

# Activate virtual environment
source myenv/bin/activate

# Set up library paths for PyTorch
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")

# Default parameters
EPOCHS=100
BATCH_SIZE=128
HIDDEN_DIM=64
NUM_HEADS=4
DEVICE="cuda"
OUTPUT_DIR="./comparison_plots"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --hidden-dim)
            HIDDEN_DIM="$2"
            shift 2
            ;;
        --num-heads)
            NUM_HEADS="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --quick)
            # Quick test mode
            EPOCHS=10
            echo "Quick test mode: Running only 10 epochs"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--epochs N] [--batch-size N] [--hidden-dim N] [--num-heads N] [--device cuda|cpu] [--output-dir DIR] [--quick]"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "CUDA vs PyTorch Comparison"
echo "========================================"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Hidden dim: $HIDDEN_DIM"
echo "Num heads: $NUM_HEADS"
echo "Device: $DEVICE"
echo "Output dir: $OUTPUT_DIR"
echo "========================================"
echo ""

# Run the comparison
python tests/compare_implementations.py \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --hidden-dim $HIDDEN_DIM \
    --num-heads $NUM_HEADS \
    --device $DEVICE \
    --output-dir $OUTPUT_DIR

echo ""
echo "========================================"
echo "Comparison complete!"
echo "Plots saved to: $OUTPUT_DIR"
echo "========================================"
