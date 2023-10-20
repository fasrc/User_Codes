#!/usr/bin/env bash
#SBATCH -J sum_test
#SBATCH -o sum_job.out
#SBATCH -e sum_job.err
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -p test
#SBATCH -t 0-00:30
#SBATCH --mem=4G

# Load required software modules
module load intel/23.2.0-fasrc01

# Run the program
./sum.x
