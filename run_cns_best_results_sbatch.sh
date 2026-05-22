#!/bin/bash -l
# =============================================================================
# CNS (Colored Noise Sampling) — Best Configuration Runner (SLURM / sbatch)
# =============================================================================
# Reproduces our two best published results using the pre-trained SiT-XL/2 model:
#
#   Unguided (cfg=1.0): FID 6.27   (best SiT unguided result)
#   Guided   (cfg=1.5): FID 1.98  (best SiT guided result)
#
# The script also includes the standard SDE baseline for reference.
#
# Submit:
#   sbatch run_cns_best_results_sbatch.sh [MODE]
#
#   MODE options (passed as the first argument after --):
#     unguided     — run best unguided CNS  (FID 6.27)
#     guided       — run best guided   CNS  (FID 1.98)
#     baseline     — run standard SDE baseline (cfg=1.0)
#     baseline_cfg — run standard SDE baseline (cfg=1.5)
#     all          — run all four (default)
#
#   Examples:
#     sbatch run_cns_best_results_sbatch.sh unguided
#     sbatch run_cns_best_results_sbatch.sh all
#
# FID Evaluation (after sampling):
#   python evaluator.py VIRTUAL_imagenet256_labeled.npz <sample_folder>.npz
#   (evaluator.py from https://github.com/openai/guided-diffusion/tree/main/evaluations)
# =============================================================================

# --- SLURM CONFIGURATION ---
#SBATCH --job-name=SiT_CNS
#SBATCH --output=sbatch_logs/%x_%j.out
#SBATCH --error=sbatch_logs/%x_%j.err
#SBATCH --mail-user=hadar.davidson@mail.huji.ac.il
#SBATCH --mail-type=BEGIN,END,TIME_LIMIT_90,FAIL

# --- RESOURCES ---
#SBATCH --time=9:59:00
#SBATCH --mem=150G
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:l40s:8

# --- ENVIRONMENT SETUP ---
cd /cs/labs/sagieb/hadar_davidson

module load nvidia/default
module load spack/x86-64-v1
module load cuda/12.2
module load miniconda3/

# >>> conda initialize >>>
__conda_setup="$('/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh" ]; then
        . "/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh"
    else
        export PATH="/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate cfd-l40s

export HF_HOME="/cs/labs/sagieb/hadar_davidson/cache/huggingface"
unset DISPLAY

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TCNN_CUDA_ARCHITECTURES=89
export TORCH_CUDA_ARCH_LIST="8.9"

cd /cs/labs/sagieb/hadar_davidson/4DGen/BlueNoiseInjection/SiT/SiT_for_github

mkdir -p sbatch_logs

# =============================================================================
# Configuration
# =============================================================================
NUM_GPUS=8
NUM_FID_SAMPLES=50000
PER_PROC_BATCH_SIZE=128
SAMPLE_DIR="samples"
CKPT=""          # leave empty to auto-download pretrained SiT-XL/2
MODE=${1:-"all"}

CKPT_ARG=""
if [ -n "$CKPT" ]; then
    CKPT_ARG="--ckpt $CKPT"
fi

GAMMA_MATRIX_UNGUIDED="gamma_matrix/gamma_matrix_scaled.pt"
GAMMA_MATRIX_GUIDED="gamma_matrix/gamma_matrix_scaled_cfg_1.5.pt"
REF_STATS="VIRTUAL_imagenet256_labeled.npz"

echo "=========================================================="
echo "  CNS Best Results Runner (SLURM)"
echo "  GPUs: $NUM_GPUS | Samples: $NUM_FID_SAMPLES | Batch/GPU: $PER_PROC_BATCH_SIZE"
echo "  Mode: $MODE"
echo "=========================================================="

# --------------------------------------------------------------------------
# Helper
# --------------------------------------------------------------------------
run_header() {
    echo ""
    echo "----------------------------------------------------------"
    echo "  $1"
    echo "----------------------------------------------------------"
}

run_eval() {
    local marker="$1"
    local npz
    npz=$(find "$SAMPLE_DIR" -maxdepth 1 -name "*.npz" -newer "$marker" 2>/dev/null | head -1)
    if [ -z "$npz" ]; then
        echo "WARNING: no new NPZ found in $SAMPLE_DIR since run started, skipping FID evaluation."
        return
    fi
    echo ""
    echo "--- FID Evaluation: $npz ---"
    python evaluator.py "$REF_STATS" "$npz"
}

# ==========================================================================
# 1. BEST UNGUIDED — FID 6.27
# ==========================================================================
run_best_unguided() {
    run_header "BEST UNGUIDED — CNS (FID 6.27, cfg=1.0)"
    local marker; marker=$(mktemp)
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
        --dtype bfloat16 \
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
    run_eval "$marker"; rm -f "$marker"
}

# ==========================================================================
# 2. BEST GUIDED — FID 1.98
# ==========================================================================
# run_best_guided() {
#     run_header "BEST GUIDED — CNS (FID 1.98, cfg=1.5)"
#     torchrun \
#         --nnodes=1 --nproc_per_node=$NUM_GPUS \
#         sample_ddp.py SDE \
#         --model SiT-XL/2 \
#         --vae ema \
#         --sample-dir "$SAMPLE_DIR" \
#         --per-proc-batch-size $PER_PROC_BATCH_SIZE \
#         --num-fid-samples $NUM_FID_SAMPLES \
#         --image-size 256 \
#         --num-classes 1000 \
#         --cfg-scale 1.5 \
#         --dtype bfloat16 \
#         --num-sampling-steps 250 \
#         --global-seed 0 \
#         --sampling-method Euler \
#         --diffusion-form sigma \
#         --last-step Mean \
#         --last-step-size 0.04 \
#         --cns \
#         --gamma-matrix-path "$GAMMA_MATRIX_GUIDED" \
#         --sqrt-gamma \
#         --gamma-matrix-divider 25.0 \
#         --alpha-tilting -0.1 0.03 \
#         --no-alpha-tilting-inside-exp \
#         --alpha-tilting-use-fnorm \
#         --energy-scale 0.998 \
#         $CKPT_ARG
#     echo "Done. Samples saved to: $SAMPLE_DIR/SiT-XL-2-pretrained-cfg-1.5-...-cns-..."
#     run_eval
# }

# ==========================================================================
# 3. STANDARD SDE BASELINE — Unguided (cfg=1.0)
# ==========================================================================
run_baseline_unguided() {
    run_header "STANDARD SDE BASELINE — Unguided (cfg=1.0)"
    local marker; marker=$(mktemp)
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
        --dtype bfloat16 \
        --num-sampling-steps 250 \
        --global-seed 0 \
        --sampling-method Euler \
        --diffusion-form sigma \
        --last-step Mean \
        --last-step-size 0.04 \
        $CKPT_ARG
    echo "Done. Samples saved to: $SAMPLE_DIR/SiT-XL-2-pretrained-cfg-1.0-...-SDE-..."
    run_eval "$marker"; rm -f "$marker"
}

# ==========================================================================
# 4. STANDARD SDE BASELINE — Guided (cfg=1.5)
# ==========================================================================
run_baseline_guided() {
    run_header "STANDARD SDE BASELINE — Guided (cfg=1.5)"
    local marker; marker=$(mktemp)
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
        --dtype bfloat16 \
        --num-sampling-steps 250 \
        --global-seed 0 \
        --sampling-method Euler \
        --diffusion-form sigma \
        --last-step Mean \
        --last-step-size 0.04 \
        $CKPT_ARG
    echo "Done. Samples saved to: $SAMPLE_DIR/SiT-XL-2-pretrained-cfg-1.5-...-SDE-..."
    run_eval "$marker"; rm -f "$marker"
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
