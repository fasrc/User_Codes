#!/bin/bash
#SBATCH --job-name=mnist-naive
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=mnist_naive_%j.out
#SBATCH --error=mnist_naive_%j.err

module load python
mamba activate /n/holylabs/rc_admin/Everyone/checkpoint-training/pt2.14.0_cuda13.2

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

echo "============================================================"
echo "MNIST training on GPU - no checkpointing"
echo "============================================================"
echo "Job ID : ${SLURM_JOB_ID}"
echo "Node   : $(hostname)"
echo "GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start  : $(date)"
echo "============================================================"

srun python mnist_naive.py \
    --epochs 10 \
    --device cuda

echo "End: $(date)"
