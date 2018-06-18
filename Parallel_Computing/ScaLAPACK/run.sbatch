#!/bin/bash
#SBATCH -J scalapack_test
#SBATCH -o scalapack_test.out
#SBATCH -e scalapack_test.err
#SBATCH -p shared
#SBATCH -t 0-00:30
#SBATCH -n 6
#SBATCH --mem-per-cpu=4000

#ulimit -l unlimited

# Load required modules
module load intel/17.0.4-fasrc01
module load openmpi/2.1.0-fasrc02 

# Run program
srun -n 6 --mpi=pmi2 ./psgesv.x
