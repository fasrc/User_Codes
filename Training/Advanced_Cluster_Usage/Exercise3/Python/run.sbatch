#!/bin/bash
#SBATCH -J mp_pi
#SBATCH -o mp_pi.out
#SBATCH -e mp_pi.err
#SBATCH -t 0-00:30
#SBATCH -p test
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem=4G

PRO=mp_pi

# --- Load required software modules ---
module load python/3.12.8-fasrc01 
unset OMP_NUM_THREADS

# --- Run program with 1, 2, 4, 8, 16, 32, and 64 OpenMP threads ---
echo "Number of threads: ${SLURM_CPUS_PER_TASK}"
srun -c ${SLURM_CPUS_PER_TASK} python ${PRO}.py 30000000 ${SLURM_CPUS_PER_TASK} > ${PRO}.dat
