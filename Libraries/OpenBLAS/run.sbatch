#!/bin/bash
#SBATCH -J openblas_test
#SBATCH -o openblas_test.out
#SBATCH -e openblas_test.err
#SBATCH -p shared
#SBATCH -t 30
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=4000

# --- Load required modules ---
module load gcc/9.2.0-fasrc01
module load OpenBLAS/0.3.7-fasrc02

# --- Run program ---
./openblas_test.x
