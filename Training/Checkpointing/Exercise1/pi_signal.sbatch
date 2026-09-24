#!/bin/bash
#SBATCH --job-name=pi-signal
#SBATCH --partition=rc-testing
#SBATCH --time=00:03:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=pi_signal_%j.out
#SBATCH --error=pi_signal_%j.err
#SBATCH --signal=USR1@60

module load python

# New run. SLURM sends SIGUSR1 60 s before the time limit (after ~2 min);
# pi_signal.py then saves a checkpoint and exits cleanly.
srun python pi_signal.py \
    --darts 10000000000 \
    --seed 42 \
    --report-every 1000000 \
    --checkpoint-every 0 \
    --checkpoint-file checkpoints/pi_signal.pkl
