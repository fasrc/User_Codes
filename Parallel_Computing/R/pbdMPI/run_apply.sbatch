#!/bin/bash
#SBATCH --job-name=pbdMPI_apply
#SBATCH --output=apply_%j.out
#SBATCH --error=apply_%j.err
#SBATCH --partition=test
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=3
#SBATCH --mem-per-cpu=4000
#SBATCH -t 0-00:10               # Runtime in D-HH:MM, minimum of 10 minutes

# load modules
module load gcc/9.3.0-fasrc01 openmpi/4.0.5-fasrc01 R/4.0.5-fasrc02

# set R lirbary path
export R_LIBS_USER=$HOME/apps/R/4.0.5_pbdDocs:$R_LIBS_USER

# choose one of the mpirun statements:
# run R program and keep output and error messages in pbdApply.Rout
mpirun Rscript --vanilla pbdApply.R > pbdApply.Rout 2>&1 

# run R program and keep output in pbdApply.Rout and error messages in error file
#mpirun Rscript --vanilla pbdApply.R > pbdApply.Rout 

