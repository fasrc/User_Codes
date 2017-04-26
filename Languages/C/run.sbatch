#!/usr/bin/env bash
#SBATCH -J c_test_job
#SBATCH -o c_test_job.out
#SBATCH -e c_test_job.err
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 0-00:30
#SBATCH --mem=4000

# Load required software modules
source new-modules.sh
module load intel/17.0.2-fasrc01

# Run program
./sum.x 100
