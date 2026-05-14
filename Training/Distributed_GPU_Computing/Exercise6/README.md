# Exercise 6 — Multi-Node Multi-GPU CNN Training on MNIST with PyTorch DDP

## Introduction

This exercise demonstrates **data-parallel training** of a convolutional neural
network (CNN) on the MNIST handwritten-digit dataset across multiple nodes and
GPUs using
[PyTorch DistributedDataParallel (DDP)](https://pytorch.org/docs/stable/notes/ddp.html).

Eight SLURM tasks — 4 per node, one task per GPU — form a single distributed
process group via **NCCL**. Each rank trains on its own shard of the MNIST
training set, computed by `DistributedSampler`. After each backward pass, DDP
**all-reduces gradients** across all 8 ranks so every GPU applies the same
update. A separate validation loop evaluates the full test set in parallel
after every epoch, and the final model checkpoint is saved by rank 0.

---

## Content

| File | Description |
|------|-------------|
| `train_mnist_ddp.py` | Main training script — DDP setup, CNN model, MNIST loader, train/eval loops |
| `run.sbatch` | SLURM submission script — 2 nodes, 4 tasks/node, 4 GPUs/node |

---

## Workflow

### 1. Reading the distributed environment from SLURM

The Python script maps SLURM environment variables to the PyTorch distributed
conventions used by `dist.init_process_group`:

```python
os.environ["RANK"]       = os.environ["SLURM_PROCID"]
os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
```

`MASTER_ADDR` and `MASTER_PORT` are resolved and exported by the SLURM batch
script before `srun` is called, giving every rank the same rendezvous point.

### 2. GPU assignment

Each rank sets the CUDA device to its **local** rank on the node:

```python
local_rank = int(os.environ["LOCAL_RANK"])   # 0–3 on each node
torch.cuda.set_device(local_rank)
```

With 4 tasks per node and 4 GPUs per node, each task maps to exactly one GPU.

### 3. Initializing the distributed process group via TCPStore

Rather than relying on `init_method="env://"`, the script builds an explicit
`TCPStore` with the already-resolved IPv4 `MASTER_ADDR`. This bypasses
`getaddrinfo` and avoids IPv6-vs-IPv4 ambiguity that can cause hangs on
clusters with dual-stack networking:

```python
store = dist.TCPStore(
    host_name=master_addr,
    port=master_port,
    world_size=world_size,
    is_master=(rank == 0),
    timeout=datetime.timedelta(seconds=init_timeout),
)
dist.init_process_group(
    backend="nccl",
    store=store,
    rank=rank,
    world_size=world_size,
    timeout=datetime.timedelta(seconds=init_timeout),
    device_id=torch.device(f"cuda:{local_rank}"),
)
```

### 4. MNIST data loading — no torchvision required

The script ships its own pure-PyTorch MNIST loader. On rank 0, the four
original IDX-format gzip files are downloaded from a public S3 mirror:

```python
MNIST_URLS = {
    "train_images": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
    ...
}
```

All other ranks wait at a `dist.barrier()` until the download is complete,
then load the same files from disk. Images are cast to `float32`, divided by
255, and normalized to zero mean / unit standard deviation using the standard
MNIST statistics (mean = 0.1307, std = 0.3081).

### 5. Model: simple CNN for MNIST

`MNISTNet` is a two-block CNN followed by a small fully-connected head:

```
Conv2d(1→32, 3×3) → ReLU
Conv2d(32→64, 3×3) → ReLU → MaxPool2d(2) → Dropout(0.25)
Linear(64×14×14 → 256) → ReLU → Dropout(0.5) → Linear(256 → 10)
```

The model is moved to the local GPU, then wrapped in `DistributedDataParallel`:

```python
model = MNISTNet().to(device)
model = DDP(model, device_ids=[local_rank])
```

DDP broadcasts the initial parameters from rank 0 to all other ranks on
construction, guaranteeing identical starting weights everywhere.

### 6. Dataset sharding with DistributedSampler

`DistributedSampler` partitions the 60,000 training samples into 8 disjoint
shards — one per rank. With a per-rank batch size of 64, the effective global
batch size is:

```
64 (per-rank) × 8 (ranks) = 512
```

The validation set (10,000 samples) is similarly sharded across all 8 ranks.
Because `DistributedSampler` pads the dataset so every rank sees the same
number of batches, the evaluation loop reports the padded sample count when it
differs from the true dataset size.

### 7. Linear learning-rate scaling

A base learning rate of 1e-3 is linearly scaled by the world size:

```python
lr = args.lr * world_size   # 1e-3 × 8 = 8e-3
```

This compensates for the larger effective batch size and keeps the gradient
update magnitude roughly constant relative to single-GPU training. The scaling
can be disabled with `--no-lr-scaling`.

### 8. Training loop

For each epoch:

1. `sampler.set_epoch(epoch)` — re-seeds the shuffle so each rank sees a
   different sample ordering every epoch.
2. Each rank runs a forward pass on its local batch, computes cross-entropy
   loss, and calls `loss.backward()`.
3. DDP performs an **all-reduce** on the gradients before `optimizer.step()`.
4. Per-batch loss and running accuracy are accumulated on-device to avoid
   GPU→CPU synchronization on every iteration.
5. At epoch end, three scalar tensors (loss sum, correct count, total count)
   are all-reduced across ranks to compute global train loss and accuracy.

### 9. Evaluation loop

After each epoch, the validation set is evaluated across all ranks:

```python
global_loss_sum = reduce_sum(local_loss_sum, device).item()
global_correct  = reduce_sum(local_correct,  device).item()
global_total    = reduce_sum(local_total,    device).item()
```

Only rank 0 prints the epoch summary; all other ranks are silent.

### 10. Checkpoint saving

After the final epoch, rank 0 saves the model's state dict (unwrapped from
the DDP shell) to disk:

```python
torch.save(model.module.state_dict(), args.checkpoint)
```

### 11. Rendezvous setup in the SLURM script

The batch script resolves the master node's IPv4 address before `srun`:

```bash
MASTER_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_ADDR=$(getent ahostsv4 "$MASTER_HOST" | awk '{print $1; exit}')
export MASTER_ADDR
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 20000))
export WORLD_SIZE=$SLURM_NTASKS
```

The port formula ensures different concurrent jobs on the same node do not
collide (20,000 possible offsets above base port 29500).

---

## Running

### On FASRC (Cannon cluster)

The SLURM script takes the Python script as its first argument:

```bash
sbatch run.sbatch train_mnist_ddp.py
```

This requests **2 nodes**, **4 tasks per node**, **4 GPUs per node**, and
**8 CPUs per task**. Output goes to `mnist_ddp_<JOBID>.out`.

Optional training arguments can be passed after the script name:

```bash
sbatch run.sbatch train_mnist_ddp.py --epochs 20 --batch-size 128
```

The `run.sbatch` submission script has the below contents:

```bash
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
Python binary  : /n/netscratch/rc_admin/Everyone/dist-gpu-training/pt2.9.1_cuda12.9/bin/python
Job ID         : 14286
Node list      : holygpu7c[26103-26104]
Num nodes      : 2
Num tasks      : 8
Tasks per node : 4
CPUs per task  : 8
MASTER_ADDR    : 10.31.179.36
MASTER_PORT    : 43786
WORLD_SIZE     : 8
Script         : train_mnist_ddp.py
Script args    :
Training with 8 GPUs
Per-rank batch size: 64 (effective: 512)
Learning rate: 0.008 (scaled)
Using 4 DataLoader worker(s) per rank
Downloading train_images...
Downloading train_labels...
Downloading test_images...
Downloading test_labels...
  Epoch 1 | batch 0/118 | local loss 2.2919
  Epoch 1 | batch 100/118 | local loss 0.0515
Epoch 1/10 | train loss 0.5426  acc 86.80% | val loss 0.0892  acc 97.26%
  ...
Epoch 10/10 | train loss 0.0776  acc 97.61% | val loss 0.0347  acc 98.79%
Checkpoint saved to mnist_ddp.pt
```

---

## Key concepts illustrated

- **DistributedDataParallel (DDP)** — each rank holds a full model replica;
  gradients are all-reduced automatically after every backward pass
- **DistributedSampler** — partitions MNIST indices into disjoint per-rank
  shards so the full dataset is covered exactly once per epoch
- **NCCL backend** — GPU-native collective communication for gradient
  all-reduce across nodes
- **Explicit TCPStore rendezvous** — building the store with an IPv4 address
  bypasses `getaddrinfo` and avoids IPv6 ambiguity on dual-stack clusters
- **Local GPU assignment** — `LOCAL_RANK` (from `SLURM_LOCALID`) maps each
  global rank to the correct local device index
- **Linear LR scaling** — learning rate is multiplied by world size to
  compensate for the larger effective batch size
- **On-device metric accumulation** — loss and accuracy are summed in GPU
  tensors throughout the epoch and all-reduced once at the end, minimizing
  CPU–GPU synchronization overhead
- **Torchvision-free MNIST** — IDX-format files are downloaded and parsed with
  `gzip` and `struct` from the standard library; the only dependencies are
  NumPy and PyTorch
- **Checkpoint from rank 0** — `model.module.state_dict()` unwraps the DDP
  wrapper before saving, producing a checkpoint compatible with plain
  (non-distributed) inference code

---

## Notes

- `NCCL_SOCKET_IFNAME=^lo,docker` excludes loopback and Docker interfaces
  so NCCL picks up the cluster's high-speed interconnect (InfiniBand or
  Ethernet) automatically.
- `NCCL_SOCKET_FAMILY=AF_INET` forces IPv4, which avoids resolution
  ambiguities on clusters that have both IPv4 and IPv6 addresses.
- `--cpu-bind=cores` in the `srun` call improves NUMA locality by pinning
  each task's threads to cores on the same NUMA domain as its GPU.
- MNIST is downloaded only once by rank 0; a `dist.barrier()` ensures all
  other ranks wait until the files are on disk before reading them.
- The validation `DistributedSampler` may add a few padding samples to make
  the dataset evenly divisible across ranks. The output notes the padded
  count when it exceeds the true test-set size of 10,000.
- Uncomment `NCCL_DEBUG=INFO` or `TORCH_DISTRIBUTED_DEBUG=DETAIL` in the
  SLURM script to get verbose collective-communication diagnostics when
  troubleshooting hangs or connection failures.
- Scaling to more GPUs requires only changing `--nodes` and `--gpus-per-node`
  in the SLURM script; the Python code adapts automatically via the SLURM
  environment variables.
