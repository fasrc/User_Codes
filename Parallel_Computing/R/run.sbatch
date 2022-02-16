#!/bin/bash
#SBATCH -J mpi_test
#SBATCH -o mpi_test.o
#SBATCH -e mpi_test.e
#SBATCH -p shared
#SBATCH -n 8
#SBATCH -t 30
#SBATCH --mem-per-cpu=4000

# Load required software modules 
module load gcc/9.3.0-fasrc01 openmpi/4.0.5-fasrc01 R/4.0.5-fasrc03

# Set up Rmpi package
export R_LIBS_USER=$HOME/apps/R_4.0.5_fasrc03:$R_LIBS_USER
export R_PROFILE=$HOME/apps/R_4.0.5_fasrc03/Rmpi/Rprofile

# Run program
export OMPI_MCA_mpi_warn_on_fork=0
srun -n 8 --mpi=pmix R CMD BATCH --no-save --no-restore mpi_test.R mpi_test.out
