# Exercise 3: Multi-GPU Monte Carlo Pi with NCCL — Single-Node

This exercise demonstrates multi-GPU Monte Carlo π estimation using
**NCCL** (NVIDIA Collective Communications Library) on a single node.
Unlike Exercise 2 (MPI + CUDA), here a single CPU process owns all GPUs
directly and uses NCCL's `ncclAllReduce` to sum per-GPU hit counts across
devices — no MPI required.

## Algorithm

The value of π is estimated by sampling random points uniformly in the unit
square and counting how many fall inside the unit circle:

$$\pi \approx 4 \times \frac{\text{hits inside circle}}{\text{total samples}}$$

<p align="center">
  <img src="mc_pi_white.png" alt="Monte Carlo Pi algorithm" width="480"/>
</p>

Each GPU receives an equal share of the total sample count. Within each GPU, a
CUDA kernel distributes samples across threads, each seeded independently via
SplitMix64. After the kernel, **CUB** (`cub::DeviceReduce::Sum`) reduces the
per-thread hit counts to a single scalar on-device. NCCL's `ncclAllReduce`
then sums the per-GPU scalars so that every GPU holds the global hit count.
The entire pipeline — kernel launch, CUB reduction, and NCCL AllReduce — is
enqueued asynchronously inside a single `ncclGroupStart / ncclGroupEnd` block.

## Implementation Details

| Component | Description |
|-----------|-------------|
| `nccl_pi_mc.cu` | Pure CUDA + NCCL source — no MPI |
| GPU init | `ncclCommInitAll` initializes all GPU communicators in one call |
| Collective | `ncclAllReduce` (sum) — result available on every GPU |
| Reduction | `cub::DeviceReduce::Sum` for the per-GPU thread-count reduction |
| RNG | Linear-congruential generator scrambled with SplitMix64 |
| Timing | CUDA events on GPU 0's stream, covering the full pipeline |

## Contents

| File | Description |
|------|-------------|
| `nccl_pi_mc.cu` | CUDA + NCCL source — Monte Carlo π computation |
| `Makefile` | Build rules |
| `run.sbatch` | SLURM script — 4 GPUs, single task |
| `output.out` | Example output — 4 GPUs, 10 billion samples |
| `mc_pi_white.png` | Algorithm illustration |

## Requirements

### Modules

```bash
module load nvhpc/24.11-fasrc01
```

## Compilation

```bash
make
```

or manually:

```bash
nvcc -o nccl_pi_mc.x nccl_pi_mc.cu -lnccl
```

### Makefile

```makefile
# Compiler
NVCC = nvcc

# Target
TARGET = nccl_pi_mc.x

# Source
SRC = nccl_pi_mc.cu

# Libraries
LIBS = -lnccl

# Default rule
all: $(TARGET)

# Build rule
$(TARGET): $(SRC)
	$(NVCC) -o $@ $^ $(LIBS)

# Clean rule
clean:
	rm -f $(TARGET)
```

## Running on the Cluster (SLURM)

```bash
sbatch run.sbatch
```

The script allocates a single task with 4 GPUs and runs the executable directly
(no `srun --mpi` needed — there is only one process):

```bash
#!/bin/bash
#SBATCH --job-name=nccl_pi_mc
#SBATCH --output=output.out
#SBATCH --error=error.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00

# Load the required software modules
module load nvhpc/24.11-fasrc01

# Restrict CUDA to only the GPUs allocated by SLURM
export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS

# Optionally: improve NVLink/PCIe communication for NCCL
export NCCL_DEBUG=WARN

# Run the executable
./nccl_pi_mc.x 10000000000
```

The optional argument is the total sample count (default: 1,000,000,000).

## Expected Output (4 GPUs, 10 billion samples)

```
NCCL version 2.18.5+cuda12.2
GPU 0/4 | samples=2500000000 | local hits=1963486240
GPU 1/4 | samples=2500000000 | local hits=1963515235
GPU 2/4 | samples=2500000000 | local hits=1963455274
GPU 3/4 | samples=2500000000 | local hits=1963486136
--------------------------------------------------
Exact PI       = 3.14159265
Estimated PI   = 3.14157715
Absolute error = 0.00001550
Relative error = 0.00049337 %
Total hits     = 7853942885
Total samples  = 10000000000
Total time     = 30.987 ms
```
