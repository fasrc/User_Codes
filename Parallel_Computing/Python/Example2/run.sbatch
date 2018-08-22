#!/bin/bash
#SBATCH -J optimize_mpi
#SBATCH -o optimize_mpi.out
#SBATCH -e optimize_mpi.err
#SBATCH -p shared
#SBATCH -n 8
#SBATCH -t 30
#SBATCH --mem-per-cpu=4000

# Set up environment
module load python/3.6.3-fasrc02
module load gcc/8.2.0-fasrc01
module load openmpi/3.1.1-fasrc01
source activate python3_env1

# Run the program
srun -n 8 --mpi=pmi2 python optimize_mpi.py
