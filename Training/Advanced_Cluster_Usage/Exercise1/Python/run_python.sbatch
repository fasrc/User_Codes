#!/bin/bash
#SBATCH -p test
#SBATCH -J mem_test
#SBATCH -o mem_test.out
#SBATCH -e mem_test.err
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 30
#SBATCH --mem=21G 

# Load required modules
module load python/3.12.8-fasrc01 

# Run the code
srun -n 1 -c 1 python mem_test.py
