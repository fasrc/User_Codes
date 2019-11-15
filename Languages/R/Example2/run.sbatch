#!/bin/bash
#SBATCH -J pso_test
#SBATCH -o pso_test.out
#SBATCH -e pso_test.err
#SBATCH -p shared
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 0-00:30
#SBATCH --mem=4000

# Load required software modules
module load R/3.5.1-fasrc01

# Run program
srun -n 1 -c 1 R CMD BATCH --no-save --no-restore pso_rosenbrock.R 
