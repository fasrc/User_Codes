#!/bin/bash
#SBATCH --job-name=pi-c-signal-restart
#SBATCH --partition=rc-testing
#SBATCH --time=00:03:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=pi_c_signal_restart_%j.out
#SBATCH --error=pi_c_signal_restart_%j.err
#SBATCH --signal=USR1@60

set -euo pipefail

module load gcc

mkdir -p checkpoints

echo "Job ID : ${SLURM_JOB_ID}"
echo "Node   : $(hostname)"
echo "Start  : $(date)"

# Restart from the checkpoint written by pi_signal.sbatch. The parameters
# must match the original run; --resume loads the saved state.
srun ./pi_signal \
    --darts 10000000000 \
    --seed 42 \
    --report-every 1000000 \
    --checkpoint-every 0 \
    --sleep 1 \
    --checkpoint-file checkpoints/pi_signal.bin \
    --resume

echo "End: $(date)"
