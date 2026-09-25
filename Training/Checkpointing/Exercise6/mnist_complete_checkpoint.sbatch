#!/bin/bash
#SBATCH --job-name=mnist-complete-ckpt
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:01:00
#SBATCH --output=mnist_complete_checkpoint_%j.out
#SBATCH --error=mnist_complete_checkpoint_%j.err

module load python
mamba activate /n/holylabs/rc_admin/Everyone/checkpoint-training/pt2.14.0_cuda13.2

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

echo "============================================================"
echo "MNIST complete checkpoint job"
echo "============================================================"
echo "Job ID : ${SLURM_JOB_ID}"
echo "Node   : $(hostname)"
echo "GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start  : $(date)"
echo "============================================================"

# Start a new run: remove any checkpoint from an earlier run.
rm -f mnist_complete.pt mnist_complete.pt.tmp

# At ~10 s per epoch (--sleep 8) the 1-minute time limit stops the job (state
# TIMEOUT) after about 5 epochs, before early stopping can trigger. The last
# checkpoint stays.
srun python mnist_complete_checkpoint.py \
    --epochs 20 \
    --sleep 8 \
    --device cuda \
    --checkpoint mnist_complete.pt

echo "End: $(date)"
