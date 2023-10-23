#!/bin/bash
#SBATCH -J omp_diag
#SBATCH -o omp_diag.out
#SBATCH -e omp_diag.err
#SBATCH -p test
#SBATCH -t 30
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=4G

# Set up environment
module load intel/23.2.0-fasrc01
module load intel-mkl/23.2.0-fasrc01 

# Run program
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun -c $SLURM_CPUS_PER_TASK ./omp_diag.x
