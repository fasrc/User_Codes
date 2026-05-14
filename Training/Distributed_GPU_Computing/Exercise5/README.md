# Exercise 5 — Multi-Node Multi-GPU MLP Training with PyTorch DDP

## Introduction

This exercise demonstrates **data-parallel training** of a small
Multilayer Perceptron (MLP) across multiple nodes and GPUs using
[PyTorch DistributedDataParallel (DDP)](https://pytorch.org/docs/stable/notes/ddp.html).

Each SLURM task owns one GPU on one node. The tasks form a single distributed
process group via **NCCL**, split a synthetic dataset evenly with
`DistributedSampler`, run independent forward and backward passes on their
local data shards, and then **all-reduce gradients** across all ranks so every
GPU applies the same gradient update. The result is that all replicas stay
perfectly synchronized while processing the full dataset in parallel.

---

## Content

| File | Description |
|------|-------------|
| `mlp_ddp.py` | Main training script — DDP setup, model, training loop |
| `random_dataset_gen.py` | Synthetic in-memory dataset (`RandomTensorDataset`) |
| `run.sbatch` | SLURM submission script — 2 nodes, 1 task/node, 1 GPU/node |

---

## Workflow

### 1. Reading the distributed environment from SLURM

The Python script reads all job layout information directly from SLURM
environment variables, so no command-line arguments are needed:

```python
rank          = int(os.environ["SLURM_PROCID"])
world_size    = int(os.environ["WORLD_SIZE"])
gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
cpus_per_task = int(os.environ["SLURM_CPUS_PER_TASK"])
```

`WORLD_SIZE` is set by the SLURM script to `$SLURM_NTASKS` before `srun` is
called. `MASTER_ADDR` and `MASTER_PORT` are also exported there, giving
PyTorch the rendezvous point it needs to bootstrap the process group.

### 2. GPU assignment

Each global rank is mapped to its **local** GPU index:

```python
device = rank - gpus_per_node * (rank // gpus_per_node)
# Equivalent to: device = rank % gpus_per_node
torch.cuda.set_device(device)
```

With 1 GPU per node, every rank maps to local GPU 0 on its own node.

### 3. Initializing the distributed process group

```python
init_process_group(
    backend="nccl",
    rank=rank,
    world_size=world_size,
    device_id=torch.device(f"cuda:{device}"),
)
```

The NCCL backend is used for GPU–GPU communication. Passing `device_id`
associates each rank with its GPU upfront, which suppresses NCCL warnings
about unknown device mappings on the first barrier.

### 4. Model construction and wrapping with DDP

A small three-layer MLP is built and moved to the assigned GPU, then
wrapped in `DistributedDataParallel`:

```python
model = MLP(in_feature=6, hidden_units=4, out_feature=2).to(device)
model = DDP(model, device_ids=[device])
```

On construction, DDP broadcasts the initial parameters from rank 0 to all
other ranks so every replica starts from identical weights. After each
`loss.backward()`, DDP inserts an **all-reduce** over all ranks'
gradients before `optimizer.step()` runs, keeping all replicas in sync.

### 5. Dataset and DistributedSampler

`RandomTensorDataset` generates 1,024 `(input, target)` pairs from N(0, 1)
using a fixed seed, so every rank constructs an **identical** in-memory
dataset. `DistributedSampler` then partitions the integer indices
`[0, num_samples)` into disjoint shards — one per rank — so no two ranks
train on the same sample:

```
rank 0 → samples [0,   511]
rank 1 → samples [512, 1023]
```

The effective global batch size is `batch_size × world_size = 32 × 2 = 64`.

### 6. Training loop

For each epoch:

1. `sampler.set_epoch(epoch)` — ensures a different shuffle order each epoch
   (a no-op for `shuffle=False`, but kept to illustrate the best practice).
2. Each rank loads its local shard, moves tensors to the GPU with
   `non_blocking=True`, runs a forward pass, computes MSE loss, and calls
   `loss.backward()`.
3. DDP performs an **all-reduce** on the gradients in the background.
4. `optimizer.step()` applies the averaged gradient, updating weights
   identically on every rank.

### 7. Rendezvous setup in the SLURM script

Before launching `srun`, the SLURM script resolves the master node's IPv4
address and picks a job-unique port:

```bash
MASTER_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_ADDR=$(getent ahostsv4 "$MASTER_HOST" | awk '{print $1; exit}')
export MASTER_ADDR
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 20000))
export WORLD_SIZE=$SLURM_NTASKS
```

`NCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME` exclude loopback and Docker
interfaces so NCCL uses the actual high-speed interconnect.

### 8. Ordered diagnostic output

To prevent interleaved output from concurrent ranks, the script serializes
prints using `dist.barrier()`:

```python
for r in range(world_size):
    if rank == r:
        print(message, flush=True)
    dist.barrier(device_ids=[device])
```

---

## Running

### On FASRC (Cannon cluster)

The SLURM script takes the Python script as its first argument:

```bash
sbatch run.sbatch mlp_ddp.py
```

This requests **2 nodes**, **1 task per node**, **1 GPU per node**, and
**4 CPUs per task**. Output goes to `mlp_multi_gpu_<JOBID>.out`.

The `run.sbatch` submission script has the below contents:

```bash
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

set -euo pipefail

SCRIPT="$1"; shift

CONDA_ENV=/n/netscratch/rc_admin/Everyone/dist-gpu-training/pt2.9.1_cuda12.9
module load python/3.13.12-fasrc01
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

MASTER_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_ADDR=$(getent ahostsv4 "$MASTER_HOST" | awk '{print $1; exit}')
export MASTER_ADDR
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 20000))
export WORLD_SIZE=$SLURM_NTASKS

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_SOCKET_IFNAME=^lo,docker
export GLOO_SOCKET_IFNAME=^lo,docker
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_FAMILY=AF_INET

srun --ntasks="$SLURM_NTASKS" \
     --ntasks-per-node="$SLURM_NTASKS_PER_NODE" \
     --cpu-bind=cores \
     python -u "$SCRIPT" "$@"
```

### Conda environment

A shared environment with PyTorch 2.9.1 and CUDA 12.9 is available to all
workshop participants at:

```
/n/netscratch/rc_admin/Everyone/dist-gpu-training/pt2.9.1_cuda12.9
```

This is the environment already configured in `run.sbatch` — no setup is
needed to use it.

### Creating a local PyTorch environment

If you prefer to create your own PyTorch environment with GPU support:

```bash
# Start an interactive GPU session
salloc -p gpu -t 0-06:00 --mem=8000 --gres=gpu:1

# Load a Python module
module load python

# Create and activate a conda environment
mamba create -n pt2.9.1_cuda12.9 pip wheel
source activate pt2.9.1_cuda12.9

# Install PyTorch with CUDA 12.9 support
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129
```

Then update the `CONDA_ENV` variable in `run.sbatch` to point to your
environment path.

---

## Example output

```
Running: mlp_ddp.py
Python binary  : /n/netscratch/rc_admin/Everyone/dist-gpu-training/pt2.9.1_cuda12.9/bin/python
Job ID         : 14285
Node list      : holygpu7c[26103-26104]
Num nodes      : 2
Num tasks      : 2
Tasks per node : 1
CPUs per task  : 4
GPUs per node  : 1
MASTER_ADDR    : 10.31.179.36
MASTER_PORT    : 43785
WORLD_SIZE     : 2
Script         : mlp_ddp.py

============================================================
Rank 0 of 2
Host                : holygpu7c26103.rc.fas.harvard.edu
Short host          : holygpu7c26103
Local GPU assigned  : 0
GPUs per node       : 1
Visible CUDA devices: 0
Process group init  : True
============================================================

============================================================
Rank 1 of 2
Host                : holygpu7c26104.rc.fas.harvard.edu
Short host          : holygpu7c26104
Local GPU assigned  : 0
GPUs per node       : 1
Visible CUDA devices: 0
Process group init  : True
============================================================
[Rank 0 | GPU 0] Model Summary
  Input features  : 6
  Hidden units    : 4
  Output features : 2
  Total parameters: 38
    - hidden_layer.weight: 24
    - hidden_layer.bias  : 4
    - output_layer.weight: 8
    - output_layer.bias  : 2
[Rank 1 | GPU 0] Model Summary
  Input features  : 6
  Hidden units    : 4
  Output features : 2
  Total parameters: 38
    - hidden_layer.weight: 24
    - hidden_layer.bias  : 4
    - output_layer.weight: 8
    - output_layer.bias  : 2
[Rank 0 | GPU 0] Data Summary
  Total dataset samples : 1024
  Local batch size      : 32
  Steps on this rank    : 16
  World size            : 2
  Effective global batch: 64
  DataLoader workers    : 2
[Rank 1 | GPU 0] Data Summary
  Total dataset samples : 1024
  Local batch size      : 32
  Steps on this rank    : 16
  World size            : 2
  Effective global batch: 64
  DataLoader workers    : 2
[Rank 0 | GPU 0] Starting Epoch 0
  Steps this epoch: 16
[Rank 1 | GPU 0] Starting Epoch 0
  Steps this epoch: 16
[Rank 0 | GPU 0] Finished Epoch 0
  First local loss: 9.202328e-01
  Final local loss: 1.041560e+00
  Mean local loss : 1.132705e+00
[Rank 1 | GPU 0] Finished Epoch 0
  First local loss: 1.371476e+00
  Final local loss: 1.329003e+00
  Mean local loss : 1.203457e+00
[Rank 0 | GPU 0] Training complete on holygpu7c26103
[Rank 1 | GPU 0] Training complete on holygpu7c26104
```

---

## Key concepts illustrated

- **DistributedDataParallel (DDP)** — each rank holds a full model replica;
  gradients are all-reduced automatically after every backward pass
- **DistributedSampler** — partitions dataset indices into disjoint shards,
  one per rank, so the full dataset is covered exactly once per epoch
- **NCCL backend** — GPU-native collective communication for gradient
  all-reduce across nodes
- **SLURM-driven rendezvous** — `MASTER_ADDR` / `MASTER_PORT` / `WORLD_SIZE`
  are resolved in the batch script and exported before `srun` launches tasks
- **Local GPU assignment** — `rank % gpus_per_node` maps each global rank to
  the correct local device index without any MPI library
- **Deterministic dataset construction** — a fixed seed in `RandomTensorDataset`
  guarantees every rank builds an identical dataset; `DistributedSampler`
  then performs the sharding via index selection, not data regeneration
- **Ordered collective output** — `dist.barrier()` serializes prints across
  concurrent ranks to produce deterministic, non-interleaved stdout
- **Effective batch size scaling** — the gradient all-reduce averages
  contributions from all ranks, so the effective batch size is
  `batch_size × world_size` without any code changes in the training loop

---

## Notes

- `NCCL_SOCKET_IFNAME=^lo,docker` excludes loopback and Docker interfaces
  so NCCL picks up the cluster's high-speed interconnect (InfiniBand or
  Ethernet) automatically.
- `NCCL_SOCKET_FAMILY=AF_INET` forces IPv4, which avoids resolution
  ambiguities on clusters that have both IPv4 and IPv6 addresses.
- The script validates that `SLURM_GPUS_ON_NODE` matches
  `torch.cuda.device_count()` and aborts early if they disagree, preventing
  silent GPU mis-assignment.
- Scaling to more GPUs or more nodes requires only changing `--nodes` and
  `--gpus-per-node` in the SLURM script; the Python code adapts automatically
  via the SLURM environment variables.
- Uncomment `NCCL_DEBUG=INFO` or `TORCH_DISTRIBUTED_DEBUG=DETAIL` in the
  SLURM script to get verbose collective-communication diagnostics when
  troubleshooting hangs or connection failures.
