#!/bin/bash
#SBATCH -p test
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=12000
#SBATCH -J saxpy_test
#SBATCH -o saxpy_test.out
#SBATCH -e saxpy_test.err
#SBATCH -t 30

# Load required modules
module load python/3.10.12-fasrc01
source activate numba_env

# Run the program
srun -n 1 -c 1 python saxpy.py
