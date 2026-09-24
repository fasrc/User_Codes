#!/bin/bash
#SBATCH --job-name=pi-signal-restart
#SBATCH --partition=rc-testing
#SBATCH --time=00:03:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=pi_signal_restart_%j.out
#SBATCH --error=pi_signal_restart_%j.err
#SBATCH --signal=USR1@60

module load python

# Restart from the checkpoint written by pi_signal.sbatch. The parameters
# must match the original run; --resume loads the saved state.
srun python pi_signal.py \
    --darts 10000000000 \
    --seed 42 \
    --report-every 1000000 \
    --checkpoint-every 0 \
    --checkpoint-file checkpoints/pi_signal.pkl \
    --resume
