#!/bin/bash
#SBATCH -J mpi4py_test
#SBATCH -o mpi4py_test.out
#SBATCH -e mpi4py_test.err
#SBATCH -p test
#SBATCH -n 4
#SBATCH -t 30
#SBATCH --mem-per-cpu=4000

# Set up environment
module load gcc/9.3.0-fasrc01 openmpi/4.0.2-fasrc01 tau/2.29-fasrc01
source activate mxnet151_cu101

# Run program
srun -n 4 --mpi=pmi2 tau_python mpi4py_test.py
