# MPI and CUDA 

This example implements a **high-performance parallel estimator for π (pi)** using a hybrid approach that combines **MPI** for distributed processing and **CUDA** for GPU acceleration. It uses the **midpoint rule** for numerical integration of the function:

$$
f(x) = \frac{4}{1 + x^2}
$$

over the interval $[0, 1]$, which corresponds to the integral form of π.


## Build Instructions

You can load the required software modules

```bash
module load gcc/12.2.0-fasrc01 
module load openmpi/5.0.5-fasrc02
```
and compile the program using the included `Makefile`:

```bash
make
```

Alternatively, manually compile with:

```bash
nvcc -Xcompiler -fopenmp mpi_cuda.cu -o mpi_cuda.x -I${MPI_INCLUDE} -L${MPI_LIB} -lmpi -lgomp
```

### Example Makefile

```make
# Compiler
NVCC = nvcc

# Output binary name
TARGET = mpi_cuda.x

# Source file
SRC = mpi_cuda.cu

# Compiler flags
CXXFLAGS = -Xcompiler -fopenmp

# Include and library paths (customizable)
INCLUDES = -I$(MPI_INCLUDE)
LIBS     = -L$(MPI_LIB) -lmpi -lgomp

# Default target
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CXXFLAGS) $(SRC) -o $(TARGET) $(INCLUDES) $(LIBS)

# Clean target
clean:
	rm -f $(TARGET)

# Phony targets
.PHONY: all clean
```

### Source code `mpi_cuda.cu`

```c
#include <stdio.h>
#include <mpi.h>
#include <cuda.h>

#define NBIN  1000000000   // Number of bins
#define NUM_BLOCK   13     // Number of thread blocks
#define NUM_THREAD 192     // Number of threads per block

// CUDA Kernel
__global__ void cal_pi(float *sum, int nbin, float step, float offset, int nthreads, int nblocks) {
    int i;
    float x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (i = idx; i < nbin; i += nthreads * nblocks) {
        x = offset + (i + 0.5f) * step;
        sum[idx] += 4.0f / (1.0f + x * x);
    }
}

int main(int argc, char **argv) {
    int myid, nproc, nbin, tid, num_devices, device_id;
    float step, offset, pi = 0.0f, pig;
    dim3 dimGrid(NUM_BLOCK, 1, 1);
    dim3 dimBlock(NUM_THREAD, 1, 1);
    float *sumHost, *sumDev;
    int dev_used;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    // Determine number of bins and step size
    nbin = NBIN / nproc;
    step = 1.0f / (float)(nbin * nproc);
    offset = myid * step * nbin;

    // Determine number of GPUs
    cudaGetDeviceCount(&num_devices);
    device_id = myid % num_devices;
    cudaSetDevice(device_id);

    // Allocate memory
    size_t size = NUM_BLOCK * NUM_THREAD * sizeof(float);
    sumHost = (float *)malloc(size);
    cudaMalloc((void **)&sumDev, size);
    cudaMemset(sumDev, 0, size);

    // Launch CUDA kernel
    cal_pi<<<dimGrid, dimBlock>>>(sumDev, nbin, step, offset, NUM_THREAD, NUM_BLOCK);

    // Copy results and reduce
    cudaMemcpy(sumHost, sumDev, size, cudaMemcpyDeviceToHost);
    for (tid = 0; tid < NUM_THREAD * NUM_BLOCK; tid++) {
        pi += sumHost[tid];
    }
    pi *= step;

    // Cleanup
    free(sumHost);
    cudaFree(sumDev);

    cudaGetDevice(&dev_used);
    printf("myid = %d: device used = %d; partial pi = %f\n", myid, dev_used, pi);

    // MPI reduction
    MPI_Allreduce(&pi, &pig, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    if (myid == 0)
        printf("PI = %f\n", pig);

    MPI_Finalize();
    return 0;
}
```

## Run Instructions

Use the provided batch-job submission script `run.sbatch` to send the job to the queue:

```bash
sbatch run.sbatch
```

### Batch-job submission script

```bash
#!/bin/bash
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=8G
#SBATCH -J mpi_and_cuda
#SBATCH -t 1:00:00
#SBATCH -p gpu
#SBATCH -o output.out
#SBATCH -e error.err


export UCX_TLS=^gdr_copy
module load gcc/12.2.0-fasrc01 
module load openmpi/5.0.5-fasrc02

srun -n 8 --mpi=pmix ./mpi_cuda.x
```

## Configuration

You can change the number of Monte Carlo samples (integration bins) by modifying the `NBIN` constant in the source code:

```c
#define NBIN 1000000000  // Total number of bins
```

To improve precision, you can increase this number or switch from `float` to `double`.

## Output Example

```
myid = 0: device used = 0; partial pi = 0.785398
myid = 1: device used = 1; partial pi = 0.785398
myid = 2: device used = 2; partial pi = 0.785398
myid = 3: device used = 3; partial pi = 0.785398
PI = 3.141593
```
