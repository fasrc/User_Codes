#!/bin/bash
#SBATCH -J parallel_monte_carlo
#SBATCH -o parallel_monte_carlo.out
#SBATCH -e parallel_monte_carlo.err
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -t 30
#SBATCH -p test
#SBATCH --mem=12GB

# Load required software modules
module load matlab/R2022a-fasrc01

# Run program
srun -n 1 -c $SLURM_CPUS_PER_TASK matlab -nosplash -nodesktop -r "parallel_monte_carlo($SLURM_CPUS_PER_TASK);exit"

