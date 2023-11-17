#!/bin/bash
#SBATCH -J mpi4py_test
#SBATCH -o mpi4py_test.out
#SBATCH -e mpi4py_test.err
#SBATCH -p test
#SBATCH -n 16
#SBATCH -t 30
#SBATCH --mem-per-cpu=4000

# Set up environment
module load python/3.10.12-fasrc01
module load gcc/12.2.0-fasrc01
module load openmpi/4.1.5-fasrc03
source activate python3_env2

# Run program
srun -n 16 --mpi=pmix python mpi4py_test.py
