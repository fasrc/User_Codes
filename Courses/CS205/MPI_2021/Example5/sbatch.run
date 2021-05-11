#!/bin/bash -l
#SBATCH -J planczos
#SBATCH -o planczos.out
#SBATCH -e planczos.err
#SBATCH -t 20
#SBATCH -n 8
#SBATCH -N 1
#SBATCH --mem-per-cpu=4000

# Set up environment
PRO=planczos

# Load required software modules
module load gcc/9.3.0-fasrc01 openmpi/4.0.5-fasrc01

# Run program
srun -n $SLURM_NTASKS --mpi=pmix ./${PRO}.x > ${PRO}.dat

