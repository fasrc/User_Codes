#!/bin/bash
#SBATCH -c 1
#SBATCH -N 1
#SBATCH -t 0-00:30
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH -o pytorch_%j.out 
#SBATCH -e pytorch_%j.err 

# Load software modules and source conda environment
module load python/3.10.12-fasrc01
source activate pt2.1.0_cuda12.1

# Run program
srun -c 1 --gres=gpu:1 python check_gpu.py 
