#!/bin/bash
#SBATCH -J pi_monte_carlo
#SBATCH -o pi_monte_carlo.out
#SBATCH -e pi_monte_carlo.err
#SBATCH -p test
#SBATCH -t 30
#SBATCH -n 1
#SBATCH --mem-per-cpu=1G

# Load required modules
module load intel/24.2.1-fasrc01 openmpi/5.0.5-fasrc01

# Run program
srun -n $SLURM_NTASKS --mpi=pmix ./pi_monte_carlo.x
