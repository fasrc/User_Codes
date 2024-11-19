#!/bin/bash
#SBATCH -J multinode
#SBATCH -o test.out
#SBATCH -e test.err
#SBATCH -p test
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 30
#SBATCH --mem-per-cpu=4000

# Load required software modules
module load R/4.4.1-fasrc01

# Set up R library
export R_LIBS_USER=$HOME/apps/R/4.4.1

# Run program
Rscript --vanilla future_hybrid.R > future_hybrid.Rout 2>&1 

