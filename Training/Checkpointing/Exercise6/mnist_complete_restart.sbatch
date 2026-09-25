#!/bin/bash
#SBATCH --job-name=mnist-complete-restart
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=mnist_complete_restart_%j.out
#SBATCH --error=mnist_complete_restart_%j.err

module load python
mamba activate /n/holylabs/rc_admin/Everyone/checkpoint-training/pt2.14.0_cuda13.2

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

echo "============================================================"
echo "MNIST complete checkpoint restart"
echo "============================================================"
echo "Job ID : ${SLURM_JOB_ID}"
echo "Node   : $(hostname)"
echo "GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start  : $(date)"
echo "============================================================"

[[ -f mnist_complete.pt ]] || {
    echo "ERROR: mnist_complete.pt not found. Run mnist_complete_checkpoint.sbatch first."
    exit 1
}

# Same options as the checkpoint job, plus --resume.
srun python mnist_complete_checkpoint.py \
    --epochs 20 \
    --sleep 8 \
    --device cuda \
    --checkpoint mnist_complete.pt \
    --resume

echo "End: $(date)"
