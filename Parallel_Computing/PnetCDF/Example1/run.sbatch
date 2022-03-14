#!/bin/bash
#SBATCH -J parallel_netcdf
#SBATCH -o parallel_netcdf.out
#SBATCH -e parallel_netcdf.err
#SBATCH -p test
#SBATCH -t 0-00:30
#SBATCH -n 8
#SBATCH --mem-per-cpu=4000

# Load required modules
module load intel/21.2.0-fasrc01
module load impi/2021.2.0-fasrc01
module load pnetcdf/1.12.1-fasrc02

# Run program
srun -n 8 --mpi=pmi2 ./transpose.x
