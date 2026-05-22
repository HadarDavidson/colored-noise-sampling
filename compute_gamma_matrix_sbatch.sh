#!/bin/bash -l
# =============================================================================
# Compute CNS Gamma Matrix via ODE Spectral Analysis
#
# Runs sample_ddp.py ODE --analyze-spectrum until the per-frequency 99.9%
# confidence interval converges, then saves the raw gamma matrix.
# After this completes, open scale_gamma_matrix.ipynb to produce the scaled
# version that CNS actually uses.
#
# Usage:
#   sbatch compute_gamma_matrix_sbatch.sh
#   sbatch compute_gamma_matrix_sbatch.sh 1.5    # guided (cfg=1.5)
# =============================================================================

# --- SLURM ---
#SBATCH --job-name=SiT_gamma_matrix
#SBATCH --output=sbatch_logs/%x_%j.out
#SBATCH --error=sbatch_logs/%x_%j.err
#SBATCH --mail-user=hadar.davidson@mail.huji.ac.il
#SBATCH --mail-type=BEGIN,END,TIME_LIMIT_90,FAIL
#SBATCH --time=3:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:l40s:4

# --- Environment ---
cd /cs/labs/sagieb/hadar_davidson

module load nvidia/default
module load spack/x86-64-v1
module load cuda/12.2
module load miniconda3/

__conda_setup="$('/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then eval "$__conda_setup"
else
    if [ -f "/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh" ]; then
        . "/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh"
    else
        export PATH="/usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/bin:$PATH"
    fi
fi
unset __conda_setup

source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh
conda activate cfd-l40s

export HF_HOME="/cs/labs/sagieb/hadar_davidson/cache/huggingface"
export PYTHONUNBUFFERED=1
export NVIDIA_TF32_OVERRIDE=1
unset DISPLAY

cd /cs/labs/sagieb/hadar_davidson/4DGen/BlueNoiseInjection/SiT/SiT_for_github
mkdir -p sbatch_logs

# =============================================================================
# Configuration
# =============================================================================
NUM_GPUS=4
NUM_SAMPLING_STEPS=250
PER_PROC_BATCH_SIZE=16   # small batch — we only need coverage, not throughput
MIN_SPECTRUM_SAMPLES=4096

# Pass cfg scale as first argument (e.g. "1.5"), or leave empty for unguided
CFG_SCALE=${1:-"1.0"}

RANDOM_PORT=$((10000 + RANDOM % 50000))

# =============================================================================
# Build command
# =============================================================================
CFG_ARG="--cfg-scale ${CFG_SCALE}"

echo "============================================================"
echo " CNS Gamma Matrix Computation"
echo " Steps: ${NUM_SAMPLING_STEPS} | GPUs: ${NUM_GPUS} | Batch/GPU: ${PER_PROC_BATCH_SIZE}"
echo " CFG scale: ${CFG_SCALE}"
echo " Min spectrum samples: ${MIN_SPECTRUM_SAMPLES}"
echo "============================================================"

torchrun \
    --nnodes=1 \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=${RANDOM_PORT} \
    sample_ddp.py ODE \
    --model SiT-XL/2 \
    --num-sampling-steps ${NUM_SAMPLING_STEPS} \
    --per-proc-batch-size ${PER_PROC_BATCH_SIZE} \
    --dtype bfloat16 \
    ${CFG_ARG} \
    --global-seed 0 \
    --analyze-spectrum \
    --min-spectrum-samples ${MIN_SPECTRUM_SAMPLES}

echo "============================================================"
echo " Done. Raw matrix saved inside the sample folder."
echo " Next: open scale_gamma_matrix.ipynb to produce the"
echo " scaled version for use with CNS."
echo "============================================================"
