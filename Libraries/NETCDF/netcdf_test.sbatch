#!/bin/bash
#SBATCH -J netcdf_test
#SBATCH -o netcdf_test.out
#SBATCH -e netcdf_test.err
#SBATCH -p serial_requeue
#SBATCH -t 30
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=4000

# Load required modules
source new-modules.sh
module load gcc/4.8.2-fasrc01
module load openmpi/1.10.0-fasrc01
module load netcdf/4.1.3-fasrc03

# Run program
./netcdf_test.x
