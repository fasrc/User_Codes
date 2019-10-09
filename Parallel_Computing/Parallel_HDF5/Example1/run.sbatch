#!/bin/bash
#SBATCH -J parallel_hdf5
#SBATCH -o parallel_hdf5.out
#SBATCH -e parallel_hdf5.err
#SBATCH -p general
#SBATCH -t 30
#SBATCH -n 8
#SBATCH --mem-per-cpu=4000

# Load required modules
module load intel/17.0.4-fasrc01 openmpi/3.1.1-fasrc01 hdf5/1.10.1-fasrc01
# Run program
srun -n 8 --mpi=pmi2 ./parallel_hdf5.x
