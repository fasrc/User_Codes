#!/bin/bash
#SBATCH -J mpi_test
#SBATCH -o mpi_test.o
#SBATCH -e mpi_test.e
#SBATCH -p shared
#SBATCH -n 8
#SBATCH -t 30
#SBATCH --mem-per-cpu=4000

# Load required software modules 
module load R/3.6.1-fasrc02
module load gcc/8.2.0-fasrc01 openmpi/4.0.1-fasrc01

# Set up Rmpi package
export R_LIBS_USER=$HOME/software/R/3.6.1:$R_LIBS_USER
export R_PROFILE=$HOME/software/R/3.6.1/Rmpi/Rprofile

# Run program
export OMPI_MCA_mpi_warn_on_fork=0
srun -n 8 --mpi=pmix R CMD BATCH --no-save --no-restore mpi_test.R mpi_test.out
