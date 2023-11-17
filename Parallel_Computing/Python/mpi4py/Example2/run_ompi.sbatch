#!/bin/bash
#SBATCH -J optimize_mpi
#SBATCH -o optimize_mpi.out
#SBATCH -e optimize_mpi.err
#SBATCH -p test
#SBATCH -n 8
#SBATCH -t 30
#SBATCH --mem-per-cpu=4000

# Set up environment
module load python/3.10.12-fasrc01
module load gcc/12.2.0-fasrc01
module load openmpi/4.1.5-fasrc03
source activate python3_env2

# Run the program
srun -n 8 --mpi=pmix python optimize_mpi.py
