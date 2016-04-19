#!/bin/bash
#SBATCH -J armadillo_test
#SBATCH -o armadillo_test.out
#SBATCH -e armadillo_test.err
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
./armadillo_test.x
