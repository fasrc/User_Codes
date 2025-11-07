#!/bin/bash
#SBATCH -J planczos
#SBATCH -o planczos.out
#SBATCH -e planczos.err
#SBATCH -p test
#SBATCH -t 30
#SBATCH -n 8
#SBATCH --mem-per-cpu=4G

# Load required modules
module load intel/24.2.1-fasrc01 openmpi/5.0.5-fasrc01

# Run program
srun -n $SLURM_NTASKS --mpi=pmix ./planczos.x
