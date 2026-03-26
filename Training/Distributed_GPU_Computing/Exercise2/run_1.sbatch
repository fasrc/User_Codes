#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=8G
#SBATCH -J mpi_and_cuda
#SBATCH -t 1:00:00
#SBATCH -p gpu
#SBATCH -o output_1.out
#SBATCH -e error_1.err

export UCX_TLS=^gdr_copy
export UCX_LOG_LEVEL=error
module load gcc/12.2.0-fasrc01 
module load openmpi/5.0.5-fasrc02

srun -n 1 --mpi=pmix ./mpi_cuda_pi_mc.x 10000000000000 123
