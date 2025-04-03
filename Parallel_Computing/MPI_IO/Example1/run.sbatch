#!/bin/bash
#SBATCH -J mpi_IO_test1
#SBATCH -o mpi_IO_test1.out
#SBATCH -e mpi_IO_test1.err
#SBATCH -p test
#SBATCH -t 30
#SBATCH -n 8
#SBATCH --mem-per-cpu=4000

# --- Load required modules ---
module load intel/25.0.1-fasrc01
module load openmpi/5.0.5-fasrc01

# --- Run program ---
srun -n 8 --mpi=pmix ./mpi_IO_test1.x
