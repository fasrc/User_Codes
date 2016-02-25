#!/bin/bash
#SBATCH -J gsl_int_test
#SBATCH -o gsl_int_test.out
#SBATCH -e gsl_int_test.err
#SBATCH -p serial_requeue
#SBATCH -t 30
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=4000

# Load required modules
source new-modules.sh
module load gsl/1.16-fasrc02

# Run program
./gsl_int_test.x
