#!/bin/bash
#SBATCH -J mmult
#SBATCH -o mmult.out
#SBATCH -e mmult.err
#SBATCH -p shared
#SBATCH -n 4
#SBATCH -t 0-00:30
#SBATCH --mem-per-cpu=1000

# Set up environment
module load gcc/8.2.0-fasrc01
module load openmpi/3.1.1-fasrc01

# Run program
srun -n $SLURM_NTASKS --mpi=pmix ./mmult.x
