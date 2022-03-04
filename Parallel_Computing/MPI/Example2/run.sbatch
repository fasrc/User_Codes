#!/bin/bash
#SBATCH -J ptrap
#SBATCH -o ptrap.out
#SBATCH -e ptrap.err
#SBATCH -p test
#SBATCH -t 30
#SBATCH -n 8
#SBATCH --mem-per-cpu=4000

# Load required modules
module load intel/21.2.0-fasrc01
module load openmpi/4.1.1-fasrc01

# Run program
srun -n 8 --mpi=pmix ./ptrap.x
