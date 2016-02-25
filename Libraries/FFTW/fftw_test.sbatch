#!/bin/bash
#SBATCH -J fftw_test
#SBATCH -o fftw_test.out
#SBATCH -e fftw_test.err
#SBATCH -p serial_requeue
#SBATCH -t 30
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=4000

# Load required modules
source new-modules.sh
module load gcc/4.8.2-fasrc01
module load openmpi/1.8.3-fasrc02
module load fftw/3.3.4-fasrc06

# Run program
./fftw_test.x
