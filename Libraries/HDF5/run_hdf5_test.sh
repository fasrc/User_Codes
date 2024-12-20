#!/bin/bash
#SBATCH -J hdf5_test
#SBATCH -o hdf5_test.out
#SBATCH -e hdf5_test.err
#SBATCH -p serial_requeue
#SBATCH -t 30
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=4000

# Load required modules
source new-modules.sh
module load hdf5/1.8.12-fasrc08

# Run program
./hdf5_test.x
