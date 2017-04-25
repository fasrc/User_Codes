#!/usr/bin/env bash
#SBATCH -J cpp_test
#SBATCH -o cpp_test.out
#SBATCH -e cpp_test.err
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -p serial_requeue
#SBATCH -t 0-00:30
#SBATCH --mem=4000

# Load required modules
source new-modules.sh
module load intel/17.0.2-fasrc01

# Run program
./cpp_test.x
