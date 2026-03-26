#!/bin/bash
#SBATCH --job-name=mlp_multi_gpu
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=00:20:00
#SBATCH --mem=64G
#SBATCH --partition=gpu

if [ -z "$1" ]; then
    echo "Usage: sbatch slurm_ddp_tp_fsdp.sh mlp_ddp.py"
    exit 1
fi

echo "Running: $1"

module load python/3.13.12-fasrc01
conda activate /n/holylfs06/LABS/rc_admin/Lab/pkrastev/conda/pt2.5.0_cuda12.4

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=39591
export RANK=$SLURM_PROCID
export WORLD_SIZE=$SLURM_NTASKS

echo "Job ID         : $SLURM_JOB_ID"
echo "Node list      : $SLURM_JOB_NODELIST"
echo "Num nodes      : $SLURM_NNODES"
echo "Num tasks      : $SLURM_NTASKS"
echo "Tasks per node : $SLURM_NTASKS_PER_NODE"
echo "GPUs per node  : $SLURM_GPUS_ON_NODE"
echo "MASTER_ADDR    : $MASTER_ADDR"
echo "MASTER_PORT    : $MASTER_PORT"

srun --ntasks=$SLURM_NTASKS \
     --ntasks-per-node=1 \
     --gpus-per-task=1 \
     python -u "$1"
