#!/bin/bash
#SBATCH -J omp_dot
#SBATCH -o omp_dot.out
#SBATCH -e omp_dot.err
#SBATCH -t 0-00:30
#SBATCH -p test
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=4000

PRO=omp_dot

# Run program
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun -c $SLURM_CPUS_PER_TASK singularity exec omp_dot.simg /opt/bin/omp_dot.x | sort > ${PRO}.dat
