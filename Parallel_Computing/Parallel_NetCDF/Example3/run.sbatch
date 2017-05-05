#!/bin/bash
#SBATCH -J parallel_netcdf
#SBATCH -o parallel_netcdf.out
#SBATCH -e parallel_netcdf.err
#SBATCH -p general
#SBATCH -t 0-00:30
#SBATCH -n 10
#SBATCH --mem-per-cpu=4000

# Load required modules
source new-modules.sh
module load intel/17.0.2-fasrc01
module load impi/2017.2.174-fasrc01
module load netcdf-fortran/4.4.4-fasrc03

# Run program
srun -n 10 --mpi=pmi2 ./simple_xy_par_wr.x
