#!/bin/bash
#SBATCH -J prec_test
#SBATCH -o prec_test.out
#SBATCH -e prec_test.err
#SBATCH -p serial_requeue
#SBATCH -t 30
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=4000

# Load required modules
source new-modules.sh
module load gcc/4.8.2-fasrc01
module load armadillo/5.100.2-fasrc01

# Run program
./mpint_test_v2.x
