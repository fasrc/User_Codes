#!/bin/bash
#SBATCH -J gurobitest    # job name
#SBATCH -o test.out      # standard output file
#SBATCH -e test.err      # standard error file
#SBATCH -p test          # partition
#SBATCH -c 1             # number of cores
#SBATCH -t 0-00:30       # time in D-HH:MM
#SBATCH --mem=4000       # memory in MB

# Set up software environment
module load python
module load gurobi
source activate gurobi_env

# Run program
srun -c $SLURM_CPUS_PER_TASK python gurobi_test.py
