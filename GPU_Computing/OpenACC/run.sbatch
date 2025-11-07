#!/bin/bash
#SBATCH -p gpu_test
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH --mem=12000
#SBATCH -J openacc_test
#SBATCH -o openacc_test.out
#SBATCH -e openacc_test.err
#SBATCH -t 30

# Load required modules
module load nvhpc/24.11-fasrc01

# Run the executable
./example_acc.x
