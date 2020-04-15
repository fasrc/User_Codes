#!/bin/bash
#SBATCH -J mpi4py_test
#SBATCH -o mpi4py_test.out
#SBATCH -e mpi4py_test.err
#SBATCH -p gpu_test
#SBATCH -n 16
#SBATCH -t 30
#SBATCH --mem-per-cpu=4000

# Set up environment
module load python/3.7.7-fasrc01 
source activate python3_env1

# Run program
srun -n 16 --mpi=pmi2 python mpi4py_test.py
