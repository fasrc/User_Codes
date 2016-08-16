#!/bin/bash
#SBATCH -J mpi4py_test
#SBATCH -o mpi4py_test.out
#SBATCH -e mpi4py_test.err
#SBATCH -p general
#SBATCH -n 16
#SBATCH -t 30
#SBATCH --mem-per-cpu=4000

source new-modules.sh
module load python/2.7.6-fasrc01
srun -n 16 --mpi=pmi2 python mpi4py_test.py
