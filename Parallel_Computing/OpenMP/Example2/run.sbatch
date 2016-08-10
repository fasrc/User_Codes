#!/bin/bash
#SBATCH -J omp_dot
#SBATCH -o omp_dot.out
#SBATCH -e omp_dot.err
#SBATCH -p general
#SBATCH -t 30
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=4000

# Set up environment
source new-modules.sh
module load intel/15.0.0-fasrc01
module load intel-mkl/11.0.0.079-fasrc02 

# Run program
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun -c $SLURM_CPUS_PER_TASK ./omp_diag.x
