#!/bin/bash
#SBATCH -J mc_pi              # job name
#SBATCH -o mc_pi.out          # standard output file
#SBATCH -e mc_pi.err          # standard error file
#SBATCH -N 1                  # number of nodes
#SBATCH -c 1                  # number of cores
#SBATCH -p serial_requeue     # partition
#SBATCH -t 0-00:30            # time in D-HH:MM
#SBATCH --mem=4000            # memory in MB

# Load required modules
module load python

# Run program
srun -c 1  python mc_pi.py
