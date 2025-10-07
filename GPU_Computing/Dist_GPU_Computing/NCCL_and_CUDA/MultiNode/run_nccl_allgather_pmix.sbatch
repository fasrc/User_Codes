#!/bin/bash
#SBATCH -J nccl_allgather_pmix
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -t 30
#SBATCH -p gpu
#SBATCH -o nccl_allgather_pmix.out
#SBATCH -e nccl_allgather_pmix.err
#SBATCH --mem-per-cpu=4G

# --- Helpful NCCL debug ---
#export NCCL_DEBUG=INFO
# export NCCL_SOCKET_IFNAME=eth0                    # or bond0, enp*, etc.
export NCCL_IB_HCA=mlx5_0                           # adjust if using Infiniband
# export NCCL_P2P_DISABLE=0

export UCX_LOG_LEVEL=error

module load nvhpc/24.11-fasrc01                      # NCCL and CUDA
module load gcc/12.2.0-fasrc01 openmpi/4.1.5-fasrc03 # MPI

srun -n 8 --mpi=pmix ./ncclAllGather_pmix.x 

