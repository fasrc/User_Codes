#!/bin/bash
#SBATCH -J omp_dot
#SBATCH -o omp_dot.out
#SBATCH -e omp_dot.err
#SBATCH -p general
#SBATCH -t 30
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=4000

# Set up environment
source new-modules.sh
export OMP_NUM_THREADS=4

# Run program
./omp_dor.x
