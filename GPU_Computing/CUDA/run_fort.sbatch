#!/bin/bash
#SBATCH -p gpu_test
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH --mem=12000
#SBATCH -J cuda_test
#SBATCH -o cuda_test.out
#SBATCH -e cuda_test.err
#SBATCH -t 30

# Load required modules
module load nvhpc/24.11-fasrc01

# Run the executable
./saxpy.x
