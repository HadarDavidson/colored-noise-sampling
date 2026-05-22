#!/bin/bash

# --- Configuration ---
REF_STATS="VIRTUAL_imagenet256_labeled.npz"
SAMPLE_NPZ="samples/SiT-XL-2-pretrained-cfg-1.5-128-SDE-250-Euler-sigma-Mean-0.04seed-0.npz"

# --- Derived log path (no manual naming needed) ---
LOG_DIR="evaluation_logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
NPZ_STEM=$(basename "$SAMPLE_NPZ" .npz)
LOG_PATH="$LOG_DIR/evaluation_${TIMESTAMP}_${NPZ_STEM}.log"

# --- Pre-run checks ---
if [ ! -f "$REF_STATS" ]; then
    echo "ERROR: Reference file not found: $REF_STATS"
    exit 1
fi
if [ ! -f "$SAMPLE_NPZ" ]; then
    echo "ERROR: Sample file not found: $SAMPLE_NPZ"
    exit 1
fi

# --- Redirect stdout+stderr to log and terminal ---
exec > >(tee -i "$LOG_PATH") 2>&1

echo "-------------------------------------------------------"
echo "Starting SiT Evaluation Script"
echo "Log File: $LOG_PATH"
echo "Date:     $(date)"
echo "-------------------------------------------------------"
echo "Evaluating FID/IS for samples..."
echo "Reference: $REF_STATS"
echo "Target:    $SAMPLE_NPZ"
echo ""

python evaluator.py "$REF_STATS" "$SAMPLE_NPZ"

RET_CODE=$?
echo ""
if [ $RET_CODE -eq 0 ]; then
    echo "Evaluation completed successfully."
else
    echo "Evaluation failed with exit code $RET_CODE."
fi
echo "-------------------------------------------------------"
echo "End of Log"
echo "-------------------------------------------------------"
