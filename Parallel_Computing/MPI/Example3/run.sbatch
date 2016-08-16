#!/bin/bash
#SBATCH -J planczos
#SBATCH -o planczos.out
#SBATCH -e planczos.err
#SBATCH -p general
#SBATCH -t 30
#SBATCH -n 8
#SBATCH --mem-per-cpu=4000

# Load required modules
source new-modules.sh
module load intel/15.0.0-fasrc01
module load openmpi/1.8.3-fasrc02

# Run program
srun -n 8 --mpi=pmi2 ./planczos.x
