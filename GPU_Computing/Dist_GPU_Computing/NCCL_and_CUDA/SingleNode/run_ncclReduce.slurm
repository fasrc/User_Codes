#!/bin/bash
#SBATCH --job-name=ncclReduce
#SBATCH --output=ncclReduce.out
#SBATCH --error=ncclReduce.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00

# Load the required software modules
module load nvhpc/24.11-fasrc01 

# Run the executable
./ncclReduce.x


