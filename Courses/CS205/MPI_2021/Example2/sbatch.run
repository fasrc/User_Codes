#!/bin/bash -l
#SBATCH -J mpi_dot
#SBATCH -o mpi_dot.out
#SBATCH -e mpi_dot.err
#SBATCH -t 0-00:10
#SBATCH -n 4
#SBATCH -N 1
#SBATCH --mem-per-cpu=1000

PRO=mpi_dot

# Load required software modules
module load gcc/9.3.0-fasrc01 openmpi/4.0.5-fasrc01

# Run program
srun -n $SLURM_NTASKS --mpi=pmix ./${PRO}.x | sort -k7 -n > ${PRO}.dat

