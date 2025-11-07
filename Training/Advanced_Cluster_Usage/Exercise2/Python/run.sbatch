#!/bin/bash
#SBATCH -p test
#SBATCH -J mp_mem_test
#SBATCH -o mp_mem_test.out
#SBATCH -e mp_mem_test.err
#SBATCH -N 1
#SBATCH -c 2
#SBATCH -t 30
#SBATCH --mem-per-cpu=11G 

# --- Load required modules ---
module load python/3.12.8-fasrc01

# --- Run the code ---
srun -c ${SLURM_CPUS_PER_TASK} python mp_mem_test.py
