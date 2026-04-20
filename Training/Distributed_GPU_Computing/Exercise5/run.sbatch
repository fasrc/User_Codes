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

# Fail fast on errors, undefined variables, and broken pipes.
set -euo pipefail

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
if [ -z "${1:-}" ]; then
    echo "Usage: sbatch run.sbatch mlp_ddp.py [--script-args ...]"
    exit 1
fi
SCRIPT="$1"
shift  # remaining args ("$@") are forwarded to the Python script

echo "Running: $SCRIPT"

# ---------------------------------------------------------------------------
# Conda environment
# ---------------------------------------------------------------------------
# FASRC-recommended pattern: load the module that provides conda, then
# source conda's hook explicitly so `conda activate` works in non-interactive
# SLURM shells.
CONDA_ENV=/n/holylfs06/LABS/rc_admin/Lab/pkrastev/conda/pt2.5.0_cuda12.4

module load python/3.13.12-fasrc01
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

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

# (Deliberately NOT exporting RANK here: SLURM_PROCID in the sbatch shell
# doesn't carry the per-task value. srun sets the correct SLURM_PROCID
# inside each task, and the Python code reads it directly.)

# Threading / interconnect.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_SOCKET_IFNAME=^lo,docker
export GLOO_SOCKET_IFNAME=^lo,docker
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_FAMILY=AF_INET

# Uncomment while debugging:
# export NCCL_DEBUG=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export TORCH_CPP_LOG_LEVEL=ERROR

# ---------------------------------------------------------------------------
# Job summary
# ---------------------------------------------------------------------------
echo "Job ID         : $SLURM_JOB_ID"
echo "Node list      : $SLURM_JOB_NODELIST"
echo "Num nodes      : $SLURM_NNODES"
echo "Num tasks      : $SLURM_NTASKS"
echo "Tasks per node : $SLURM_NTASKS_PER_NODE"
echo "CPUs per task  : $SLURM_CPUS_PER_TASK"
echo "GPUs per node  : ${SLURM_GPUS_ON_NODE:-unset}"
echo "MASTER_ADDR    : $MASTER_ADDR"
echo "MASTER_PORT    : $MASTER_PORT"
echo "WORLD_SIZE     : $WORLD_SIZE"
echo "Script         : $SCRIPT"
echo "Script args    : $*"

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
# Omitting --gpus-per-task here: on a 1-GPU-per-node allocation with one
# task per node, each task sees the node's single GPU and the Python code
# calls torch.cuda.set_device(LOCAL_RANK) explicitly.
srun --ntasks="$SLURM_NTASKS" \
     --ntasks-per-node="$SLURM_NTASKS_PER_NODE" \
     --cpu-bind=cores \
     python -u "$SCRIPT" "$@"
