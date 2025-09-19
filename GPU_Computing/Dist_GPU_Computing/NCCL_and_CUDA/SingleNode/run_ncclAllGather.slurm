#!/bin/bash
#SBATCH --job-name=ncclAllGather
#SBATCH --output=ncclAllGather.out
#SBATCH --error=ncclAllGather.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00

# Run the executable inside the container with GPU support
#singularity exec --nv nccl_cuda_12.9.sif ./ncclAllGather.x
singularity exec --nv /n/home06/pkrastev/netscratch/Singularity/NCCL/cuda-dl-base_25.06-cuda12.9-devel-ubuntu24.04.sif ./ncclAllGather.x

