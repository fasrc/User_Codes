#!/bin/bash
#SBATCH -J mpi_test
#SBATCH -o mpi_test.out
#SBATCH -e mpi_test.err
#SBATCH -p shared
#SBATCH -n 8
#SBATCH -t 30
#SBATCH --mem-per-cpu=4000

# Load required software modules
module load gcc/7.1.0-fasrc01 
module load openmpi/3.1.3-fasrc01 
module load R/3.5.1-fasrc02

# Set up Rmpi package
export R_LIBS_USER=$HOME/software/R/3.5.1:$R_LIBS_USER
export R_PROFILE=$HOME/software/R/3.5.1/Rmpi/Rprofile

# Run program
export OMPI_MCA_mpi_warn_on_fork=0
srun -n 8 --mpi=pmix R CMD BATCH --no-save --no-restore mpi_test.R
