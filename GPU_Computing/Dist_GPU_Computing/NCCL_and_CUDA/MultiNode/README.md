# Distributed GPU Collectives on Multiple Nodes (NCCL + CUDA)

This repository contains **minimal, production-style examples** for running distributed GPU collectives across multiple nodes using **NCCL** and **CUDA** with three different bootstrapping strategies:

1. **NCCL + CUDA + MPI** (MPI used only for process launch and `ncclUniqueId` broadcast)  
2. **NCCL + CUDA (no MPI) using a shared file** to exchange `ncclUniqueId`  
3. **NCCL + CUDA + PMIx** (MPI-free; PMIx handles rendezvous)

Each section lists the example codes, how to compile them, and example **Slurm** batch scripts to run on **2 nodes × 4 GPUs = 8 ranks**.

---

## Prerequisites

- **CUDA Toolkit** (nvcc, libcudart)
- **NCCL** library
- A cluster with **GPUs visible on each node**
- A job launcher:
  - **MPI** (e.g., OpenMPI, MPICH, or vendor MPI) **or**
  - **Slurm + PMIx** (for the PMIx examples) **or**
  - **Slurm** only (for the shared-file example)
- For the **shared-file** bootstrap: a **shared filesystem** visible to all nodes (e.g., `$SLURM_SUBMIT_DIR`, `$HOME` on a network FS)

> **Device selection** follows the common “one process per GPU” pattern using the **local rank** on each node (e.g., `device = local_rank % cudaDeviceCount()`).

## 1) NCCL + CUDA + MPI

**MPI is used only for**:
- rank/world size discovery
- computing local rank (via `MPI_Comm_split_type`)
- broadcasting `ncclUniqueId` (`MPI_Bcast`)
  
All actual data movement runs through **NCCL**.

### Example Codes

- `ncclAllGather_mpi.cu` — AllGather across 8 ranks (each rank contributes one float)
- `ncclBcast_mpi.cu` — Broadcast from rank 0 to all ranks, then per-rank GPU kernel work
- `ncclReduce_mpi.cu` — Two Reduce ops to rank 0, then a simple device reduction on root
- `ncclReduceScatter_mpi.cu` — Sum-then-scatter with `recvcount=1` (each rank receives one element)

### Build

Load the required software modules:

```bash
module load nvhpc/24.11-fasrc01                      # NCCL and CUDA  
module load gcc/12.2.0-fasrc01 openmpi/4.1.5-fasrc03 # MPI
```

Then use `nvcc` to compile the example codes:

```bash
# AllGather
nvcc -O3 -std=c++17 -o ncclAllGather_mpi.x ncclAllGather_mpi.cu -lnccl -lmpi

# Bcast
nvcc -O3 -std=c++17 -o ncclBcast_mpi.x ncclBcast_mpi.cu -lnccl -lmpi

# Reduce
nvcc -O3 -std=c++17 -o ncclReduce_mpi.x ncclReduce_mpi.cu -lnccl -lmpi

# ReduceScatter
nvcc -O3 -std=c++17 -o ncclReduceScatter_mpi.x ncclReduceScatter_mpi.cu -lnccl -lmpi
```

### Slurm Batch Scripts (MPI)

**AllGather (2 nodes × 4 GPUs)**

```bash
#!/bin/bash
#SBATCH -J nccl_allgather_mpi
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -t 30
#SBATCH -p gpu
#SBATCH -o nccl_allgather_mpi.out
#SBATCH -e nccl_allgather_mpi.err
#SBATCH --mem-per-cpu=4G

# --- Helpful NCCL debug ---
#export NCCL_DEBUG=INFO
# export NCCL_SOCKET_IFNAME=eth0                     # or bond0, enp*, etc.
export NCCL_IB_HCA=mlx5_0                            # adjust if using Infiniband
# export NCCL_P2P_DISABLE=0

export UCX_LOG_LEVEL=error

module load nvhpc/24.11-fasrc01                      # NCCL and CUDA  
module load gcc/12.2.0-fasrc01 openmpi/4.1.5-fasrc03 # MPI

srun -n 8 --mpi=pmix ./ncclAllGather_mpi.x
```

**Bcast**

```bash
#!/bin/bash
#SBATCH -J nccl_bcast_mpi
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -t 30
#SBATCH -p gpu
#SBATCH -o nccl_bcast_mpi.out
#SBATCH -e nccl_bcast_mpi.err
#SBATCH --mem-per-cpu=4G

# --- Helpful NCCL debug ---
#export NCCL_DEBUG=INFO
# export NCCL_SOCKET_IFNAME=eth0                     # or bond0, enp*, etc.
export NCCL_IB_HCA=mlx5_0                            # adjust if using Infiniband
# export NCCL_P2P_DISABLE=0

export UCX_LOG_LEVEL=error

module load nvhpc/24.11-fasrc01                      # NCCL and CUDA  
module load gcc/12.2.0-fasrc01 openmpi/4.1.5-fasrc03 # MPI

srun -n 8 --mpi=pmix ./ncclBcast_mpi.x
```

**Reduce**

```bash
#!/bin/bash
#SBATCH -J nccl_reduce_mpi
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -t 30
#SBATCH -p gpu
#SBATCH -o nccl_reduce_mpi.out
#SBATCH -e nccl_reduce_mpi.err
#SBATCH --mem-per-cpu=4G

# --- Helpful NCCL debug ---
#export NCCL_DEBUG=INFO
# export NCCL_SOCKET_IFNAME=eth0                    # or bond0, enp*, etc.
export NCCL_IB_HCA=mlx5_0                           # adjust if using Infiniband
# export NCCL_P2P_DISABLE=0

export UCX_LOG_LEVEL=error

module load nvhpc/24.11-fasrc01                      # NCCL and CUDA
module load gcc/12.2.0-fasrc01 openmpi/4.1.5-fasrc03 # MPI

srun -n 8 --mpi=pmix ./ncclReduce_mpi.x
```

**ReduceScatter**

```bash
#!/bin/bash
#SBATCH -J nccl_reduce_scatter_mpi
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -t 30
#SBATCH -p gpu
#SBATCH -o nccl_reduce_scatter_mpi.out
#SBATCH -e nccl_reduce_scatter_mpi.err
#SBATCH --mem-per-cpu=4G

# --- Helpful NCCL debug ---
#export NCCL_DEBUG=INFO
#export NCCL_SOCKET_IFNAME=eth0                      # or bond0, enp*, etc.
export NCCL_IB_HCA=mlx5_0                            # adjust if using Infiniband
#export NCCL_P2P_DISABLE=0

export UCX_LOG_LEVEL=error

module load nvhpc/24.11-fasrc01                      # NCCL and CUDA  
module load gcc/12.2.0-fasrc01 openmpi/4.1.5-fasrc03 # MPI

srun -n 8 --mpi=pmix ./ncclReduceScatter_mpi.x
```

> **mpirun alternative:**  
> `mpirun -np 8 -N 4 ./ncclAllGather_mpi.x` (and similarly for other examples)

---

## 2) NCCL + CUDA (No MPI) — **Shared-File Bootstrap**

This approach uses **no MPI**. Rank discovery comes from environment variables (e.g., `SLURM_*` or `WORLD_SIZE`/`RANK`). Rank 0 calls `ncclGetUniqueId()` and writes the UID to a shared file; other ranks spin-wait for that file and read the UID. NCCL handles all inter/intra-node collectives.

### Example Code

- `ncclAllGather_nompi.cu` — AllGather with shared-file rendezvous  
  (ordered printing via a tiny NCCL AllReduce barrier)

### Build

```bash
nvcc -O3 -std=c++17 -o ncclAllGather_nompi.x ncclAllGather_nompi.cu -lnccl
```

### Slurm Batch Script (shared-file)

```bash
#!/bin/bash
#SBATCH -J nccl_allgather_nompi
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -t 30
#SBATCH -p gpu
#SBATCH -o nccl_allgather_nompi.out
#SBATCH -e nccl_allgather_nompi.err
#SBATCH --mem-per-cpu=4G

# --- Helpful NCCL debug ---
#export NCCL_DEBUG=INFO
# export NCCL_SOCKET_IFNAME=eth0                    # or bond0, enp*, etc.
export NCCL_IB_HCA=mlx5_0                           # adjust if using Infiniband
# export NCCL_P2P_DISABLE=0

export UCX_LOG_LEVEL=error

module load nvhpc/24.11-fasrc01                      # NCCL and CUDA
module load gcc/12.2.0-fasrc01 openmpi/4.1.5-fasrc03 # MPI

srun -n 8 --export=ALL ./ncclAllGather_nompi.x
```

**Notes (shared-file)**

- The code writes/reads `nccl_uid.$SLURM_JOB_ID` in `$SLURM_SUBMIT_DIR` (fallback: `$HOME`).
- Ensure the directory is **shared and writable** by all ranks.
- If your site lacks a shared filesystem, consider the **PMIx** or **TCP rendezvous** variant.

---

## 3) NCCL + CUDA + **PMIx** (No MPI)

This approach uses **PMIx** (not MPI) to handle rendezvous and rank metadata. Rank 0 publishes the UID via `PMIx_Put` + `PMIx_Commit`; **all ranks** then call a **collecting fence** (`PMIX_COLLECT_DATA=true`), and non-root ranks retrieve the UID with `PMIx_Get` **targeting the publisher proc** (`{nspace, rank=0}`), which is required on some PMIx stacks.

### Example Code

- `ncclAllGather_pmix.cu` — AllGather with PMIx bootstrap and ordered prints via PMIx fences

### Build

```bash
nvcc -O3 -std=c++17 -o ncclAllGather_pmix.x ncclAllGather_pmix.cu -lnccl -lpmix
```

### Slurm Batch Script (PMIx)

```bash
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

srun -n 8 --mpi=pmix ./ncclAllGather_pmix.x
```

**Notes (PMIx)**

- If `PMIx_Get("nccl_uid")` times out:  
  1) Ensure a **collecting fence** (`PMIX_COLLECT_DATA=true`) runs **after** `PMIx_Put`/`PMIx_Commit`.  
  2) **Query the publisher proc** (rank 0) in `PMIx_Get` rather than your own proc.  
  3) Verify you are launching with the matching PMIx flavor (e.g., `--mpi=pmix_v3` or `pmix_v5`).
- If your site restricts Put/Get, switch to `PMIx_Publish`/`PMIx_Lookup` or use the shared-file/TCP variant.
