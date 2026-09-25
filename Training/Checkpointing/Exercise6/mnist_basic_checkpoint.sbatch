#!/bin/bash
#SBATCH --job-name=mnist-basic-ckpt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:01:00
#SBATCH --output=mnist_basic_checkpoint_%j.out
#SBATCH --error=mnist_basic_checkpoint_%j.err

module load python
mamba activate /n/holylabs/rc_admin/Everyone/checkpoint-training/pt2.14.0_cuda13.2

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

echo "============================================================"
echo "MNIST basic checkpoint job"
echo "============================================================"
echo "Job ID : ${SLURM_JOB_ID}"
echo "Node   : $(hostname)"
echo "GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start  : $(date)"
echo "============================================================"

# Start a new run: remove any checkpoint from an earlier run.
rm -f mnist_checkpoint.pt mnist_checkpoint.pt.tmp

# 20 epochs at ~5 s per epoch (--sleep 2) need more than the 1-minute time
# limit, so Slurm stops the job (state TIMEOUT) and the last checkpoint stays.
srun python mnist_checkpoint.py \
    --epochs 20 \
    --sleep 2 \
    --device cuda \
    --checkpoint mnist_checkpoint.pt

echo "End: $(date)"
