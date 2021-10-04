#!/bin/bash -l
#SBATCH -J mpi_hello
#SBATCH -o mpi_hello.out
#SBATCH -e mpi_hello.err
#SBATCH -t 0-00:10
#SBATCH -p test
#SBATCH -n 4
#SBATCH -N 1
#SBATCH --mem-per-cpu=1000

PRO=mpi_hello

# Load required software modules
module load gcc/10.2.0-fasrc01 openmpi/4.1.1-fasrc01

# Run program
srun -n $SLURM_NTASKS --mpi=pmix ./${PRO}.x > ${PRO}.dat

