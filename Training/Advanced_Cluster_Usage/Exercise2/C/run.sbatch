#!/bin/bash
#SBATCH -p test
#SBATCH -J omp_mem_test
#SBATCH -o omp_mem_test.out
#SBATCH -e omp_mem_test.err
#SBATCH -N 1
#SBATCH -c 2
#SBATCH -t 30
#SBATCH --mem-per-cpu=10G 

# Load required modules
module load gcc/14.2.0-fasrc01
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Run the code
srun -c ${SLURM_CPUS_PER_TASK} ./omp_mem_test.x
