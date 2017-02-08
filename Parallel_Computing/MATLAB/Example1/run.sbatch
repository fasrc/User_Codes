#!/bin/bash
#SBATCH -J parallel_monte_carlo
#SBATCH -o parallel_monte_carlo.out
#SBATCH -e parallel_monte_carlo.err
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -t 30
#SBATCH -p general
#SBATCH --mem=32000

# Load required software modules
source new-modules.sh
module load matlab/R2016a-fasrc01

# Run program
srun -n 1 -c 8 matlab-default -nosplash -nodesktop -r "parallel_monte_carlo($SLURM_CPUS_PER_TASK);exit"

