#!/bin/bash
#SBATCH -J openblas_test
#SBATCH -o openblas_test.out
#SBATCH -e openblas_test.err
#SBATCH -p serial_requeue
#SBATCH -t 30
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=4000

# Load required modules
source new-modules.sh
module load intel/15.0.0-fasrc01
module load OpenBLAS/0.2.14-fasrc01

# Run program
./openblas_test.x
