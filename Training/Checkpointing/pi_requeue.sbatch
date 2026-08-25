#!/bin/bash
#SBATCH --job-name=pi-requeue
#SBATCH --partition=serial_requeue
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

#SBATCH --output=pi_%j.out
#SBATCH --error=pi_%j.err
#SBATCH --open-mode=append
#SBATCH --requeue

set -euo pipefail

echo "============================================================"
echo "Starting batch script"
echo "============================================================"
echo "Job ID        : ${SLURM_JOB_ID}"
echo "Restart count : ${SLURM_RESTART_COUNT:-0}"
echo "Node          : $(hostname)"
echo "Time          : $(date)"
echo "============================================================"

module load python

srun python pi_requeue.py \
    --darts 1000000000 \
    --seed 42 \
    --report-every 1000000 \
    --checkpoint-every 5000000 \
    --checkpoint-file checkpoints/pi_checkpoint.pkl

echo
echo "Batch script finished at $(date)"
