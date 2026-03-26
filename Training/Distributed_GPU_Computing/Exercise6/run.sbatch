#!/bin/bash
#SBATCH --job-name=mnist_ddp
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=4
#SBATCH --time=00:20:00
#SBATCH --mem=16G
#SBATCH --partition=gpu

if [ -z "$1" ]; then
    echo "Usage: sbatch slurm_ddp.sh train_mnist_ddp.py"
    exit 1
fi
SCRIPT="$1"

module load python/3.13.12-fasrc01
conda activate /n/holylfs06/LABS/rc_admin/Lab/pkrastev/conda/pt2.5.0_cuda12.4

MASTER_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$(getent ahostsv4 "$MASTER_HOST" | awk '{print $1; exit}')
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 10000))
export WORLD_SIZE=$SLURM_NTASKS
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_SOCKET_IFNAME=^lo,docker
export GLOO_SOCKET_IFNAME=^lo,docker

echo "Job ID         : $SLURM_JOB_ID"
echo "Node list      : $SLURM_JOB_NODELIST"
echo "Num nodes      : $SLURM_NNODES"
echo "Num tasks      : $SLURM_NTASKS"
echo "Tasks per node : $SLURM_NTASKS_PER_NODE"
echo "MASTER_ADDR    : $MASTER_ADDR"
echo "MASTER_PORT    : $MASTER_PORT"
echo "WORLD_SIZE     : $WORLD_SIZE"

srun --ntasks=$SLURM_NTASKS \
     --ntasks-per-node=$SLURM_NTASKS_PER_NODE \
     --gpus-per-task=1 \
     python -u "$SCRIPT"
