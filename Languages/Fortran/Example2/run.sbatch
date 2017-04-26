#!/usr/bin/env bash
#SBATCH -J sum_test
#SBATCH -o sum_job.out
#SBATCH -e sum_job.err
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 0-00:30
#SBATCH --mem=4000

# Load required software modules
source new-modules.sh
module load intel/17.0.2-fasrc01

# Run the program
./sum.x
