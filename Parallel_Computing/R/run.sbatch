#!/bin/bash
#SBATCH -J mpi_test
#SBATCH -o mpi_test.out
#SBATCH -e mpi_test.err
#SBATCH -p general
#SBATCH -n 8
#SBATCH -t 30
#SBATCH --mem-per-cpu=4000

# Load required software modules
source new-modules.sh
module load intel/15.0.0-fasrc01
module load mvapich2/2.0-fasrc03
module load R/3.2.2-fasrc02

# Run program
srun -n 8 --mpi=pmi2 R CMD BATCH --no-save --no-restore mpi_test.R

# NOTE: When using OpenMPI, please replace the mpirun command line above with
# srun -n 8 --mpi=pmi2 --mca mpi_warn_on_fork 0 R CMD BATCH --no-save --no-restore mpi_test.R
