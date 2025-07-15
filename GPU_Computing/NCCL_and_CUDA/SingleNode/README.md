# Multi-GPU Computing with NCCL -- Single Node

NVIDIA Collective Communications Library (NCCL) and CUDA can be used together to efficiently implement multi-GPU communication and computation on a single node. NCCL provides optimized primitives for collective operations like `all-reduce`, `broadcast`, `reduce`, and `gather`, specifically tuned for NVIDIA GPUs and interconnects such as NVLink. When used in conjunction with CUDA, developers can offload computation to GPUs and use NCCL to synchronize and exchange data between them without needing to go through the CPU or host memory.

Here we include a collection of examples illustrating the use of NCCL on the FAS Cannon cluster. These examples demonstrate basic NCCL functionality for single-node setups using multiple GPUs, covering initialization with NCCL communicators, memory allocation and data transfers with CUDA, and synchronization across devices.

## Contents

* `ncclAllGather.cu`: NCCL+CUDA source code illustrating the `ncclAllGather` operation  
* `ncclBcast.cu`: NCCL+CUDA source code illustrating the `ncclBcast` operation
* `ncclReduce.cu`: NCCL+CUDA source code illustrating the `ncclReduce` operation
* `ncclReduceScatter.cu`: NCCL+CUDA source code illustrating the `ncclReduceScatter` operation

>Note: For details, look at the header of the source code.

## Compile

All examples in this directory can be compiled using a `Singularity` container based on the NVIDIA CUDA 12.9 development image, which includes the required CUDA toolkit and NCCL libraries:

```bash
singularity pull nccl_cuda_12.9.sif docker://nvcr.io/nvidia/cuda-dl-base:25.06-cuda12.9-devel-ubuntu24.04
```

From the folder containing your `.cu` files (e.g., `ncclAllGather.cu`, `ncclBcast.cu`, etc.), run the following:

```bash
singularity exec --nv nccl_cuda_12.9.sif nvcc -o ncclAllGather.x ncclAllGather.cu -lnccl -Wno-deprecated-gpu-targets
singularity exec --nv nccl_cuda_12.9.sif nvcc -o ncclBcast.x     ncclBcast.cu     -lnccl -Wno-deprecated-gpu-targets
singularity exec --nv nccl_cuda_12.9.sif nvcc -o ncclReduce.x    ncclReduce.cu    -lnccl -Wno-deprecated-gpu-targets
singularity exec --nv nccl_cuda_12.9.sif nvcc -o ncclReduceScatter.x ncclReduceScatter.cu -lnccl -Wno-deprecated-gpu-targets
```

Each of these commands will produce a corresponding `.x` executable, e.g., `ncclAllGather.x`, ready to run on a GPU node.

## Example Batch-Job Submission Script: `run_ncclAllGather.slurm`

```bash
#!/bin/bash
#SBATCH --job-name=ncclAllGather
#SBATCH --output=ncclAllGather.out
#SBATCH --error=ncclAllGather.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00

# Run the executable inside the container with GPU support
singularity exec --nv nccl_cuda_12.9.sif ./ncclAllGather.x
```

Run with:

```bash
sbatch run_ncclAllGather.slurm
```
## Example Output: `ncclAllGather.out`

```bash
10.00	20.00	30.00	40.00	
This is device 0
10.00	20.00	30.00	40.00	
This is device 1
10.00	20.00	30.00	40.00	
This is device 2
10.00	20.00	30.00	40.00	
This is device 3
10.00	20.00	30.00	40.00
```


