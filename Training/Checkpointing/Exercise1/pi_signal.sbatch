#!/bin/bash
#SBATCH --job-name=pi-ckpt
#SBATCH --partition=serial_requeue
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

#SBATCH --output=pi_%j.out
#SBATCH --error=pi_%j.err
#SBATCH --open-mode=append

#SBATCH --signal=USR1@120

set -euo pipefail

echo "Job ID       : $SLURM_JOB_ID"
echo "Hostname     : $(hostname)"
echo "Start time   : $(date)"
echo

srun python pi_signal.py \
    --darts 1000000000 \
    --seed 42 \
    --report-every 1000000 \
    --checkpoint-every 10000000 \
    --checkpoint-file pi_checkpoint.pkl

