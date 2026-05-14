#!/bin/bash
#SBATCH --job-name=mnist_ddp
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --time=00:20:00
#SBATCH --mem=32G
#SBATCH --partition=gpu

# Fail fast on errors, undefined variables, and broken pipes.
set -euo pipefail

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
if [ -z "${1:-}" ]; then
    echo "Usage: sbatch run.sbatch train_mnist_ddp.py [--script-args ...]"
    exit 1
fi
SCRIPT="$1"
shift  # remaining args ("$@") will be forwarded to the Python script

# ---------------------------------------------------------------------------
# Conda environment
# ---------------------------------------------------------------------------
# FASRC-recommended pattern: load the module that provides conda, then
# source conda's hook explicitly so `conda activate` works in non-interactive
# SLURM shells.
#CONDA_ENV=/n/holylfs06/LABS/rc_admin/Lab/pkrastev/conda/pt2.5.0_cuda12.4
CONDA_ENV=/n/netscratch/rc_admin/Everyone/dist-gpu-training/pt2.9.1_cuda12.9

module load python/3.13.12-fasrc01
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# Sanity check: confirm we're using the right Python.
PYTHON_BIN=$(command -v python)
echo "Python binary  : $PYTHON_BIN"
if [[ "$PYTHON_BIN" != "$CONDA_ENV"/* ]]; then
    echo "ERROR: python is not from the expected conda env ($CONDA_ENV)" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Distributed setup
# ---------------------------------------------------------------------------
MASTER_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_ADDR=$(getent ahostsv4 "$MASTER_HOST" | awk '{print $1; exit}')
if [ -z "$MASTER_ADDR" ]; then
    echo "ERROR: could not resolve $MASTER_HOST to an IPv4 address" >&2
    exit 1
fi
export MASTER_ADDR
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 20000))
export WORLD_SIZE=$SLURM_NTASKS

# Threading / interconnect.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_SOCKET_IFNAME=^lo,docker
export GLOO_SOCKET_IFNAME=^lo,docker
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_FAMILY=AF_INET

# Uncomment while debugging hangs or unexpected NCCL behaviour:
# export NCCL_DEBUG=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export TORCH_NCCL_BLOCKING_WAIT=1
# export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# ---------------------------------------------------------------------------
# Job summary
# ---------------------------------------------------------------------------
echo "Job ID         : $SLURM_JOB_ID"
echo "Node list      : $SLURM_JOB_NODELIST"
echo "Num nodes      : $SLURM_NNODES"
echo "Num tasks      : $SLURM_NTASKS"
echo "Tasks per node : $SLURM_NTASKS_PER_NODE"
echo "CPUs per task  : $SLURM_CPUS_PER_TASK"
echo "MASTER_ADDR    : $MASTER_ADDR"
echo "MASTER_PORT    : $MASTER_PORT"
echo "WORLD_SIZE     : $WORLD_SIZE"
echo "Script         : $SCRIPT"
echo "Script args    : $*"

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
# Notes:
# - We omit --gpus-per-task here. The node-level --gpus-per-node=4 plus
#   --ntasks-per-node=4 already gives one GPU per task on FASRC, and the
#   Python code calls torch.cuda.set_device(LOCAL_RANK) explicitly. Mixing
#   both forms can confuse SLURM's GPU binding on some configurations.
# - --cpu-bind=cores improves NUMA locality on multi-GPU nodes.
srun --ntasks="$SLURM_NTASKS" \
     --ntasks-per-node="$SLURM_NTASKS_PER_NODE" \
     --cpu-bind=cores \
     python -u "$SCRIPT" "$@"
