#!/bin/bash
#SBATCH --job-name=pi-c-signal
#SBATCH --partition=test
#SBATCH --time=00:03:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=pi_c_signal_%j.out
#SBATCH --error=pi_c_signal_%j.err
#SBATCH --signal=USR1@60

set -euo pipefail

module load gcc

mkdir -p checkpoints

echo "Job ID : ${SLURM_JOB_ID}"
echo "Node   : $(hostname)"
echo "Start  : $(date)"

srun ./pi_signal \
    --darts 10000000000 \
    --seed 42 \
    --report-every 1000000 \
    --checkpoint-every 0 \
    --checkpoint-file checkpoints/pi_signal.bin

echo "End: $(date)"
