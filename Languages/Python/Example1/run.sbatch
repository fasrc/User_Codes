#!/bin/bash
#SBATCH -J mc_pi                    # job name
#SBATCH -o mc_pi.out                # standard output file
#SBATCH -e mc_pi.err                # standard error file
#SBATCH --nodes=1                   # number of nodes
#SBATCH -cpus-per-task=1            # number of cores
#SBATCH --partition=serial_requeue  # partition
#SBATCH --time=0-00:30              # time in D-HH:MM
#SBATCH --mem=4000                  # memory in MB

# Load required modules
module load python

# Run program
python mc_pi.py
