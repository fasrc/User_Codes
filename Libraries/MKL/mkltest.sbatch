#!/bin/bash
#SBATCH -J mkltest
#SBATCH -o mkltest.out
#SBATCH -e mkltest.err
#SBATCH -p serial_requeue
#SBATCH -t 30
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=4000

# Load required modules
source new-modules.sh
module load intel/15.0.0-fasrc01
module load intel-mkl/11.0.0.079-fasrc02

# Run program
./mkltest.x
