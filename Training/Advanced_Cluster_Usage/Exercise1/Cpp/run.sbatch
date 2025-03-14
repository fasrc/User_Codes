#!/bin/bash
#SBATCH -p test
#SBATCH -J mem_test
#SBATCH -o mem_test.out
#SBATCH -e mem_test.err
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 30
#SBATCH --mem=20G 

# Load required modules
module load gcc/14.2.0-fasrc01

# Run the code
./mem_test.x
