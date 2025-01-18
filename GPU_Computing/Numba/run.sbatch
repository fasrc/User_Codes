#!/bin/bash
#SBATCH -p test
#SBATCH -c 1
#SBATCH --mem=12000
#SBATCH -J saxpy_test
#SBATCH -o saxpy_test.out
#SBATCH -e saxpy_test.err
#SBATCH -t 30

# Load required modules
module load python/3.12.5-fasrc01
source activate numba_env

# Run the program
srun -c 1 python saxpy.py
