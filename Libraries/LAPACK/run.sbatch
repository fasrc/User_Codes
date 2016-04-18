#!/bin/bash
#SBATCH -J lapack_test
#SBATCH -o lapack_test.out
#SBATCH -e lapack_test.err
#SBATCH -p serial_requeue
#SBATCH -t 30
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=4000

# Load required modules
source new-modules.sh

# Run program
./lapack_test.x
