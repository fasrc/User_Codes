#!/bin/bash
#SBATCH -p gpu_test
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH --mem=12000
#SBATCH -J cuda_test
#SBATCH -o cuda_test.out
#SBATCH -e cuda_test.err
#SBATCH -t 30

# Load required modules
module load cuda/12.2.0-fasrc01

# Run the executable
./saxpy.x
