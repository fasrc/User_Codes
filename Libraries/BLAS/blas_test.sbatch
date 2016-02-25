#!/bin/bash
#SBATCH -J blas_test
#SBATCH -o blas_test.out
#SBATCH -e blas_test.err
#SBATCH -p serial_requeue
#SBATCH -t 30
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=4000

# Load required modules
source new-modules.sh

# Run program
./blas_test.x
