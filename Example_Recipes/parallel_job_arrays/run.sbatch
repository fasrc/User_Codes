#!/bin/bash
#SBATCH -J parallel_sum
#SBATCH -o output_%a.out
#SBATCH -e error_%a.err
#SBATCH -p shared
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 30
#SBATCH --mem=4000
#SBATCH --array=1-3

# Load required software modules
module load matlab/R2018b-fasrc01

# Create a local work directory
mkdir -p /scratch/$USER/$SLURM_ARRAY_TASK_ID

# Run MATLAB
srun -n 1 -c 4 matlab -nodesktop -nosplash -r "parallel_sum('input_${SLURM_ARRAY_TASK_ID}');exit"

# Cleanup local work directory
rm -rf /scratch/$USER/$SLURM_ARRAY_TASK_ID
