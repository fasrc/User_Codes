#!/bin/bash
#SBATCH -J count_down         # job name
#SBATCH -o count_down.out     # standard output file
#SBATCH -e count_down.err     # standard error file
#SBATCH -p shared             # partition
#SBATCH -c 1                  # number of cores
#SBATCH -t 0-00:30            # time in D-HH:MM
#SBATCH --mem=4000            # memory in MB

# Load required software modules
module load R

# Run program
R CMD BATCH --no-save --no-restore count_down.R 

