#!/bin/bash
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=8G
#SBATCH -J mpi_and_cuda
#SBATCH -t 1:00:00
#SBATCH -p gpu
#SBATCH -o output.out
#SBATCH -e error.err


export UCX_TLS=^gdr_copy
module load gcc/12.2.0-fasrc01 
module load openmpi/5.0.5-fasrc02

srun -n 8 --mpi=pmix ./mpi_cuda.x
