#!/bin/bash
#SBATCH -J pi_monte_carlo
#SBATCH -o pi_monte_carlo.out
#SBATCH -e pi_monte_carlo.err
#SBATCH -p test
#SBATCH -t 30
#SBATCH -n 8
#SBATCH --mem-per-cpu=4000

# Load required modules
module load intel/24.0.1-fasrc01 openmpi/5.0.2-fasrc01

# Run program
srun -n $SLURM_NTASKS --mpi=pmix ./pi_monte_carlo.x
