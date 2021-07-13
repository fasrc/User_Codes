#!/bin/bash
#SBATCH -p test
#SBATCH -n 8
#SBATCH -J mpi_test
#SBATCH -o mpi_test.out
#SBATCH -e mpi_test.err
#SBATCH -t 30
#SBATCH --mem-per-cpu=1000

# --- Set up environment ---
module load python/3.8.5-fasrc01
source activate python3_env1

# --- Run the MPI application in the container ---
srun -n 8 --mpi=pmi2 singularity exec mpich_test.simg /usr/bin/mpitest.x

