#!/bin/bash

#SBATCH --job-name=pi-c-requeue
#SBATCH --partition=serial_requeue
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

#SBATCH --output=pi_c_requeue_%j.out
#SBATCH --error=pi_c_requeue_%j.err

#SBATCH --open-mode=append
#SBATCH --requeue

set -euo pipefail

module load gcc

mkdir -p checkpoints

echo "============================================================"
echo "Starting C requeue example"
echo "============================================================"
echo "Job ID        : ${SLURM_JOB_ID}"
echo "Restart count : ${SLURM_RESTART_COUNT:-0}"
echo "Node          : $(hostname)"
echo "Time          : $(date)"
echo "============================================================"

srun ./pi_requeue \
    --darts 1000000000 \
    --seed 42 \
    --report-every 1000000 \
    --checkpoint-every 5000000 \
    --checkpoint-file checkpoints/pi_requeue.bin

echo
echo "Batch script finished: $(date)"
