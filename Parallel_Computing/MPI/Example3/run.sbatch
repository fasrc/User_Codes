#!/bin/bash
#SBATCH -J planczos
#SBATCH -o planczos.out
#SBATCH -e planczos.err
#SBATCH -p rocky
#SBATCH -t 30
#SBATCH -n 8
#SBATCH --mem-per-cpu=4000

# Load required modules
module load intel/23.0.0-fasrc01 openmpi/4.1.4-fasrc01

# Run program
srun -n 8 --mpi=pmix ./planczos.x
