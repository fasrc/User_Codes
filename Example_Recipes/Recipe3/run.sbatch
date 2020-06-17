#!/bin/bash
#SBATCH -J test
#SBATCH -o test.out
#SBATCH -e test.err
#SBATCH -p shared
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 0-00:30
#SBATCH --mem=4000

# Set up software environment
module load python/3.7.7-fasrc01
module load gcc/9.3.0-fasrc01

# Run program
srun -n 1 -c 1 python drive_sum.py
