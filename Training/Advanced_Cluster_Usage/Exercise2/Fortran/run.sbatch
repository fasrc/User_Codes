#!/bin/bash
#SBATCH -J planczos
#SBATCH -o planczos.out
#SBATCH -e planczos.err
#SBATCH -p test
#SBATCH -t 30
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem-per-cpu=5G

# Load required modules
module load intel/24.2.1-fasrc01

# Run program
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun -c $SLURM_CPUS_PER_TASK ./planczos.x
