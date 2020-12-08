#!/bin/bash
#SBATCH -J test
#SBATCH -o test.out
#SBATCH -e test.err
#SBATCH -p test
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 0-00:30
#SBATCH --mem=4000

# Set up software environment
module load python/3.7.7-fasrc01
module load gurobi/9.0.2-fasrc01
export PYTHONPATH=/n/sw/gurobi902/linux64/lib/python3.7

# Run program
srun -n 1 -c 1 python gurobi_test.py
