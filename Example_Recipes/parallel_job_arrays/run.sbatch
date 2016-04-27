#!/bin/bash
#SBATCH -J parallel_sum
#SBATCH -o output_%a.out
#SBATCH -e error_%a.err
#SBATCH -p meade
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 30
#SBATCH --mem=4000
#SBATCH --array=1-3

# Create a local work directory
mkdir -p /scratch/$USER/$SLURM_ARRAY_TASK_ID

# Run MATLAB
matlab -nodesktop -nosplash -r "parallel_sum('input_${SLURM_ARRAY_TASK_ID}');exit"

# Cleanup local work directory
rm -rf /scratch/$USER/$SLURM_ARRAY_TASK_ID
