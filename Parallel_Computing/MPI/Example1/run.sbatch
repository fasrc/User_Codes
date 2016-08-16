#!/bin/bash
#SBATCH -J pi_monte_carlo
#SBATCH -o pi_monte_carlo.out
#SBATCH -e pi_monte_carlo.err
#SBATCH -p general
#SBATCH -t 30
#SBATCH -n 8
#SBATCH --mem-per-cpu=4000

# Load required modules
source new-modules.sh
module load intel/15.0.0-fasrc01
module load openmpi/1.8.3-fasrc02

# Run program
srun -n 8 --mpi=pmi2 ./pi_monte_carlo.x
