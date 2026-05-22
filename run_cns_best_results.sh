#!/bin/bash
# =============================================================================
# CNS (Colored Noise Sampling) — Best Configuration Runner
# =============================================================================
# Reproduces our two best published results using the pre-trained SiT-XL/2 model:
#
#   Unguided (cfg=1.0): FID 6.27   (best SiT unguided result)
#   Guided   (cfg=1.45): FID 1.98  (best SiT guided result)
#
# The script also includes the standard SDE baseline for reference.
#
# Requirements:
#   - torchrun (comes with PyTorch ≥ 1.9)
#   - 4 GPUs (adjust NUM_GPUS below for your hardware)
#   - Pre-trained SiT-XL/2 checkpoint (auto-downloaded on first run via download.py)
#   - gamma_matrix/gamma_matrix_scaled.pt          (unguided — included in this repo)
#   - gamma_matrix/gamma_matrix_scaled_cfg_1.45.pt  (guided   — included in this repo)
#   - VIRTUAL_imagenet256_labeled.npz for FID evaluation (download separately from
#     https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz)
#
# Usage:
#   bash run_cns_best_results.sh [MODE]
#
#   MODE options:
#     unguided    — run best unguided CNS  (FID 6.27)
#     guided      — run best guided   CNS  (FID 1.98)
#     baseline    — run standard SDE baseline (cfg=1.0)
#     baseline_cfg— run standard SDE baseline (cfg=1.5)
#     all         — run all four (default)
#
# FID Evaluation (after sampling):
#   python evaluator.py VIRTUAL_imagenet256_labeled.npz <sample_folder>.npz
#   (evaluator.py from https://github.com/openai/guided-diffusion/tree/main/evaluations)
# =============================================================================

set -e

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
NUM_GPUS=${NUM_GPUS:-4}
NUM_FID_SAMPLES=${NUM_FID_SAMPLES:-50000}
PER_PROC_BATCH_SIZE=${PER_PROC_BATCH_SIZE:-128}
SAMPLE_DIR=${SAMPLE_DIR:-"samples"}
CKPT=${CKPT:-""}          # leave empty to auto-download pretrained SiT-XL/2
MODE=${1:-"all"}

# Derived
CKPT_ARG=""
if [ -n "$CKPT" ]; then
    CKPT_ARG="--ckpt $CKPT"
fi

GAMMA_MATRIX_UNGUIDED="gamma_matrix/gamma_matrix_scaled.pt"
GAMMA_MATRIX_GUIDED="gamma_matrix/gamma_matrix_scaled_cfg_1.45.pt"

echo "=========================================================="
echo "  CNS Best Results Runner"
echo "  GPUs: $NUM_GPUS | Samples: $NUM_FID_SAMPLES | Batch/GPU: $PER_PROC_BATCH_SIZE"
echo "  Mode: $MODE"
echo "=========================================================="

# --------------------------------------------------------------------------
# Helper: print a separator before each run
# --------------------------------------------------------------------------
run_header() {
    echo ""
    echo "----------------------------------------------------------"
    echo "  $1"
    echo "----------------------------------------------------------"
}

# ==========================================================================
# 1. BEST UNGUIDED — FID 6.27
# ==========================================================================
#
# Key parameters:
#   --cns               : enable CNS spectral noise injection
#   --gamma-matrix-path        : empirical DyPE gamma matrix (non-smoothed, scaled)
#   --power-gamma 0.75         : energy shaping power
#   --gamma-matrix-divider 1.73: residual noise divider
#   --alpha-tilting 0.15 -0.5  : time-varying frequency tilt (start → end)
#   --alpha-tilting-use-fnorm  : tilt guided by normalized frequency position
#   --alpha-exponential-interpolation
#   --alpha-exponential-interpolation-sharpness 0.75
#   --energy-scale 0.98: slight total-energy reduction after unit-std norm
#
run_best_unguided() {
    run_header "BEST UNGUIDED — CNS (FID 6.27, cfg=1.0)"
    torchrun \
        --nnodes=1 --nproc_per_node=$NUM_GPUS \
        sample_ddp.py SDE \
        --model SiT-XL/2 \
        --vae ema \
        --sample-dir "$SAMPLE_DIR" \
        --per-proc-batch-size $PER_PROC_BATCH_SIZE \
        --num-fid-samples $NUM_FID_SAMPLES \
        --image-size 256 \
        --num-classes 1000 \
        --cfg-scale 1.0 \
        --num-sampling-steps 250 \
        --global-seed 0 \
        --sampling-method Euler \
        --diffusion-form sigma \
        --last-step Mean \
        --last-step-size 0.04 \
        --cns \
        --gamma-matrix-path "$GAMMA_MATRIX_UNGUIDED" \
        --power-gamma 0.75 \
        --gamma-matrix-divider 1.73 \
        --alpha-tilting 0.15 -0.5 \
        --no-alpha-tilting-inside-exp \
        --alpha-tilting-use-fnorm \
        --alpha-exponential-interpolation \
        --alpha-exponential-interpolation-sharpness 0.75 \
        --energy-scale 0.98 \
        $CKPT_ARG
    echo "Done. Samples saved to: $SAMPLE_DIR/SiT-XL-2-pretrained-cfg-1.0-...-cns-..."
}

# ==========================================================================
# 2. BEST GUIDED — FID 1.98
# ==========================================================================
#
# Key parameters:
#   --cns               : enable CNS spectral noise injection
#   --gamma-matrix-path        : empirical DyPE gamma matrix (non-smoothed, scaled)
#   --sqrt-gamma               : apply sqrt transformation to residual energy amplitude
#   --gamma-matrix-divider 25.0: residual noise divider (large → near-deterministic)
#   --cfg-scale 1.45           : classifier-free guidance scale
#   --alpha-tilting -0.1 0.03  : time-varying frequency tilt (start → end)
#   --alpha-tilting-use-fnorm  : tilt guided by normalized frequency position
#   --energy-scale 0.998: marginal energy reduction
#
run_best_guided() {
    run_header "BEST GUIDED — CNS (FID 1.98, cfg=1.45)"
    torchrun \
        --nnodes=1 --nproc_per_node=$NUM_GPUS \
        sample_ddp.py SDE \
        --model SiT-XL/2 \
        --vae ema \
        --sample-dir "$SAMPLE_DIR" \
        --per-proc-batch-size $PER_PROC_BATCH_SIZE \
        --num-fid-samples $NUM_FID_SAMPLES \
        --image-size 256 \
        --num-classes 1000 \
        --cfg-scale 1.45 \
        --num-sampling-steps 250 \
        --global-seed 0 \
        --sampling-method Euler \
        --diffusion-form sigma \
        --last-step Mean \
        --last-step-size 0.04 \
        --cns \
        --gamma-matrix-path "$GAMMA_MATRIX_GUIDED" \
        --sqrt-gamma \
        --gamma-matrix-divider 25.0 \
        --alpha-tilting -0.1 0.03 \
        --no-alpha-tilting-inside-exp \
        --alpha-tilting-use-fnorm \
        --energy-scale 0.998 \
        $CKPT_ARG
    echo "Done. Samples saved to: $SAMPLE_DIR/SiT-XL-2-pretrained-cfg-1.45-...-cns-..."
}

# ==========================================================================
# 3. STANDARD SDE BASELINE — Unguided (cfg=1.0)
# ==========================================================================
run_baseline_unguided() {
    run_header "STANDARD SDE BASELINE — Unguided (cfg=1.0)"
    torchrun \
        --nnodes=1 --nproc_per_node=$NUM_GPUS \
        sample_ddp.py SDE \
        --model SiT-XL/2 \
        --vae ema \
        --sample-dir "$SAMPLE_DIR" \
        --per-proc-batch-size $PER_PROC_BATCH_SIZE \
        --num-fid-samples $NUM_FID_SAMPLES \
        --image-size 256 \
        --num-classes 1000 \
        --cfg-scale 1.0 \
        --num-sampling-steps 250 \
        --global-seed 0 \
        --sampling-method Euler \
        --diffusion-form sigma \
        --last-step Mean \
        --last-step-size 0.04 \
        $CKPT_ARG
    echo "Done. Samples saved to: $SAMPLE_DIR/SiT-XL-2-pretrained-cfg-1.0-...-SDE-..."
}

# ==========================================================================
# 4. STANDARD SDE BASELINE — Guided (cfg=1.5)
# ==========================================================================
run_baseline_guided() {
    run_header "STANDARD SDE BASELINE — Guided (cfg=1.5)"
    torchrun \
        --nnodes=1 --nproc_per_node=$NUM_GPUS \
        sample_ddp.py SDE \
        --model SiT-XL/2 \
        --vae ema \
        --sample-dir "$SAMPLE_DIR" \
        --per-proc-batch-size $PER_PROC_BATCH_SIZE \
        --num-fid-samples $NUM_FID_SAMPLES \
        --image-size 256 \
        --num-classes 1000 \
        --cfg-scale 1.5 \
        --num-sampling-steps 250 \
        --global-seed 0 \
        --sampling-method Euler \
        --diffusion-form sigma \
        --last-step Mean \
        --last-step-size 0.04 \
        $CKPT_ARG
    echo "Done. Samples saved to: $SAMPLE_DIR/SiT-XL-2-pretrained-cfg-1.5-...-SDE-..."
}

# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------
case "$MODE" in
    unguided)
        run_best_unguided
        ;;
    guided)
        run_best_guided
        ;;
    baseline)
        run_baseline_unguided
        ;;
    baseline_cfg)
        run_baseline_guided
        ;;
    all)
        run_best_unguided
        run_best_guided
        run_baseline_unguided
        run_baseline_guided
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Valid modes: unguided | guided | baseline | baseline_cfg | all"
        exit 1
        ;;
esac

echo ""
echo "=========================================================="
echo "  Sampling complete."
echo ""
echo "  To evaluate FID, run:"
echo "    python evaluator.py VIRTUAL_imagenet256_labeled.npz <sample_dir>_<timestamp>.npz"
echo ""
echo "  evaluator.py is from:"
echo "    https://github.com/openai/guided-diffusion/tree/main/evaluations"
echo "=========================================================="
