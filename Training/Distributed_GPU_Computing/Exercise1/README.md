# Exercise 1 — Computing π with MPI + CUDA on 4 GPUs

## Introduction

This exercise demonstrates the basic pattern of combining **MPI** (inter-process communication)
with **CUDA** (on-device computation) to distribute a numerical workload across
multiple GPUs on a single compute node.

The code computes $\pi$ using the midpoint-rule approximation of the following integral:

$$
\pi = \int^1_0 4 / (1 + x^2) dx
$$

Each MPI rank is assigned a disjoint slice of the integration interval and
evaluates its slice entirely on its own GPU. A final `MPI_Reduce` sums the
per-rank partial results to produce $\pi$.

---

## Content

| File | Description |
|------|-------------|
| `mpi_cuda_pi.cu` | CUDA/MPI source code |
| `Makefile` | Build rules (NVCC + MPI host compiler) |
| `run_4.sbatch` | SLURM submission script — 1 node, 4 MPI ranks, 4 GPUs |

---

## Workflow

### 1. Distributing work across MPI ranks

The total number of integration bins (`NBIN_DEFAULT = 1,000,000,000`, or a
user-supplied value) is divided as evenly as possible across all MPI ranks.
Early ranks absorb any remainder so no bin is left unaccounted for:

```
rank 0 → bins [0,        nbin_local_0)
rank 1 → bins [nbin_local_0, ...)
...
```

Each rank uses `MPI_Comm_split_type(..., MPI_COMM_TYPE_SHARED, ...)` to build
a shared-memory sub-communicator and maps its local rank to a GPU via
`local_rank % num_devices`.

### 2. GPU computation — `compute_pi_partial`

Each rank launches a CUDA kernel across thousands of threads. Every thread
processes a strided subset of the rank's bins using the midpoint rule:

```
x_i = (start_bin + i + 0.5) * step
sum += 4.0 / (1.0 + x_i * x_i)
```

The launch configuration is chosen automatically via
`cudaOccupancyMaxPotentialBlockSize` to maximize GPU occupancy, capped at a
practical ceiling of `32 × SM_count` blocks.

### 3. GPU reduction — `reduce_sum`

Rather than copying thousands of partial sums back to the host, the code
reduces them on the GPU using a two-level warp-shuffle kernel:

1. Each thread lane reduces across its warp via `__shfl_down_sync`.
2. Per-warp results land in shared memory and are reduced by the first warp.
3. The loop repeats on the shrinking output array until a single scalar remains.

Only one `double` is ever transferred from device to host per rank.

### 4. MPI reduction

```c
MPI_Reduce(&pi_local, &pi_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
```

Rank 0 collects the per-rank partial sums and reports the final result along
with the absolute and relative error against the exact value of π.

### 5. Timing

`MPI_Barrier` is called before and after the compute + reduce section so that
the reported wall time measures the parallel work itself, not MPI startup
overhead.

### 6. Ordered diagnostic output

To avoid interleaved output from concurrent `printf` calls, rank 0 prints its
own diagnostics first, then receives and prints each subsequent rank's line in
order using point-to-point `MPI_Send` / `MPI_Recv`.

---

## Compiling

Load the required modules and compile with `make`:

```bash
module load gcc/12.2.0-fasrc01
module load openmpi/5.0.5-fasrc02 # This also loads a CUDA module

make
```

This produces the executable `mpi_cuda_pi.x`. To remove it:

The Makefile uses `nvcc` with `mpicxx` as the host compiler (`-ccbin mpicxx`),
which ensures that MPI headers and libraries are found automatically.

The `Makefile` has the below contents:

```make
# Compiler
NVCC        = nvcc

# Use MPI C++ compiler as host compiler
MPICXX      = mpicxx

# Target executable
TARGET      = mpi_cuda_pi.x

# Source file
SRC         = mpi_cuda_pi.cu

# Compiler flags
NVCC_FLAGS  = -O3 -ccbin $(MPICXX)

# Default target
all: $(TARGET)

# Build rule
$(TARGET): $(SRC)
        $(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SRC)

# Clean rule
clean:
        rm -f $(TARGET) *.x *.err *.out
```

You could also compile the code directly with:

```bash
nvcc -O3 -ccbin mpicxx -o mpi_cuda_pi.x mpi_cuda_pi.cu
```

---

## Running

### On FASRC (Cannon cluster)

Submit the provided SLURM script:

```bash
sbatch run_4.sbatch
```

This requests **1 node, 4 MPI tasks, and 4 GPUs**. Output goes to
`output.out`; errors to `error.err`.

To run with a different number of bins (the default is 10,000), e.g.,

```bash
# Edit the last line of run_4.sbatch, e.g.:
srun -n $SLURM_NTASKS --mpi=pmix ./mpi_cuda_pi.x 5000
```

The `run_4.sbatch` submission script has the below contents:

```bash
#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=8G
#SBATCH -J mpi_and_cuda
#SBATCH -t 1:00:00
#SBATCH -p gpu
#SBATCH -o output.out
#SBATCH -e error.err

export UCX_TLS=^gdr_copy
export UCX_LOG_LEVEL=error
module load gcc/12.2.0-fasrc01 
module load openmpi/5.0.5-fasrc02

srun -n $SLURM_NTASKS --mpi=pmix ./mpi_cuda_pi.x 10000
```

---

## Example output

```
cat output.out 
rank 0/4 | local_rank 0/4 | GPU 0 | bins = 2500 | blocks = 4 | threads/block = 768 | total threads = 3072 | partial pi = 0.979914653245634
rank 1/4 | local_rank 1/4 | GPU 1 | bins = 2500 | blocks = 4 | threads/block = 768 | total threads = 3072 | partial pi = 0.874675783824257
rank 2/4 | local_rank 2/4 | GPU 2 | bins = 2500 | blocks = 4 | threads/block = 768 | total threads = 3072 | partial pi = 0.719413999127246
rank 3/4 | local_rank 3/4 | GPU 3 | bins = 2500 | blocks = 4 | threads/block = 768 | total threads = 3072 | partial pi = 0.567588218225989

========== FINAL RESULTS ==========
Exact PI       = 3.14159265
Computed PI    = 3.14159265
Absolute error = 8.33333846e-10
Relative error = 2.65258402e-08 %
Total bins     = 10000
Wall time      = 0.000135 s
```

---

## Key concepts illustrated

- **MPI + CUDA co-design** — one MPI rank per GPU; each rank owns its device
- **Work decomposition** — even distribution of bins with remainder handling
- **Occupancy-aware launch configuration** — `cudaOccupancyMaxPotentialBlockSize`
- **Warp-shuffle reduction** — fast on-device reduction with `__shfl_down_sync`
- **GPU-only data movement** — only one scalar per rank ever crosses the PCIe bus
- **Ordered collective output** — deterministic stdout from concurrent MPI ranks
- **Convergence limit of numerical integration** — more GPUs partition the
  work faster but do not improve accuracy; only increasing `NBIN` does that

---

## Notes

- `UCX_TLS=^gdr_copy` in the SLURM script disables GPUDirect Copy in UCX.
  This is a conservative setting for nodes where GPUDirect RDMA may not be
  fully configured; remove it if your nodes support it and you want maximum
  MPI bandwidth.
- `--mpi=pmix` tells `srun` to use the PMIx process management interface,
  which is required for OpenMPI 5.x on FASRC.
- The code is single-node only as written. Extending it to multiple nodes
  requires no code changes — just increase `--ntasks-per-node` and `-N` in
  the SLURM script.


