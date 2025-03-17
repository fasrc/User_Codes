#!/bin/bash
#SBATCH -J speedup_env       # job name
#SBATCH -o speedup_env.out   # standard output file
#SBATCH -e speedup_env.err   # standard error file
#SBATCH --cpus-per-task=1    # number of cores 
#SBATCH --partition=test     # partition
#SBATCH --time=0-01:00       # time in D-HH:MM
#SBATCH --mem=10G            # memory in GB

# --- Load required modules ---
module load python/3.12.8-fasrc01

# --- Create conda environment ---
mamba create -n speedup_env python=3.12 pip wheel numpy matplotlib -y
