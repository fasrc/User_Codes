#!/bin/bash -l
#SBATCH -J mmult
#SBATCH -o mmult.out
#SBATCH -e mmult.err
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -p test
#SBATCH -t 0-00:30
#SBATCH --mem-per-cpu=4000

PRO=mmult

# Load required modules
module load gcc/10.2.0-fasrc01 openmpi/4.1.1-fasrc01

# Run program
srun -n $SLURM_NTASKS --mpi=pmix ./${PRO}.x > ${PRO}.dat

