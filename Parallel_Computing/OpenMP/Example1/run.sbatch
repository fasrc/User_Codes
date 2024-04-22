#!/bin/bash
#SBATCH -J omp_hello
#SBATCH -o omp_hello.out
#SBATCH -e omp_hello.err
#SBATCH -p test
#SBATCH -t 0-00:30
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=4000

# Set up environment
PRO=omp_hello

# Load required software modules
module load gcc/13.2.0-fasrc01

# Run program
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun -c $SLURM_CPUS_PER_TASK ./${PRO}.x > ${PRO}.dat
