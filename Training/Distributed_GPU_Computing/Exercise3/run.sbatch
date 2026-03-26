#!/bin/bash
#SBATCH --job-name=nccl_pi_mc
#SBATCH --output=output.out
#SBATCH --error=error.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00

# Load the required software modules
module load nvhpc/24.11-fasrc01

# Restrict CUDA to only the GPUs allocated by SLURM
export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS

# Optionally: improve NVLink/PCIe communication for NCCL
export NCCL_DEBUG=WARN

# Run the executable
./nccl_pi_mc.x 10000000000
