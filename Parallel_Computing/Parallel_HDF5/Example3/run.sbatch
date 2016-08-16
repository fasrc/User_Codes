#!/bin/bash
#SBATCH -J parallel_hdf5_3d
#SBATCH -o parallel_hdf5_3d.out
#SBATCH -e parallel_hdf5_3d.err
#SBATCH -p general
#SBATCH -t 30
#SBATCH -n 8
#SBATCH --mem-per-cpu=4000

# Load required modules
source new-modules.sh
module load intel/15.0.0-fasrc01
module load openmpi/1.10.0-fasrc01
module load hdf5/1.8.12-fasrc07

# Run program
srun -n 8 --mpi=pmi2 ./parallel_hdf5_3d.x
