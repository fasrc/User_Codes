#! /bin/bash
#SBATCH --job-name=mlp_multi_gpu
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1

#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=64G
#SBATCH --partition=gpu

# Check if the first argument ($1) is provided
if [ -z "$1" ]; then
    echo "Pass in the mlp_ddp.py, mlp_tensor_parallel.py or mlp_fsdp.py to run"
    echo "example: sbatch slurm_ddp_tp_fsdp.sh mlp_ddp.py"
else
    echo "Running: $1"
fi

module load python/3.12.11-fasrc02
source activate /n/holylfs06/LABS/rc_admin/Lab/pkrastev/conda/pt2.5.0_cuda12.4

export MASTER_ADDR=$(scontrol show hostnames | head -n 1)
export MASTER_PORT=39591
export RANK=$SLURM_PROCID
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))

srun --ntasks-per-node=$SLURM_NTASKS_PER_NODE \
    python -u "$1"
