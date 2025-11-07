# Using CUDA on FASRC Cluster

<img src="cuda-logo.jpeg" alt="cuda-logo" width="200"/>

[NVIDIA CUDA](https://developer.nvidia.com/cuda-toolkit) is a parallel computing platform and programming model developed by NVIDIA for general-purpose computing on their GPUs (Graphics Processing Units). CUDA is designed to take advantage of the parallelism inherent in GPUs and provides an easy-to-use programming interface for developers to write parallel programs that can perform complex computations faster than traditional CPU-based implementations.

CUDA provides a C/C++ programming interface and a compiler that can generate optimized GPU code. It also provides a set of libraries that can be used for various applications, including linear algebra, image processing, and signal processing. CUDA applications are typically written in a host-device model, where the CPU (host) manages the overall program flow, and the GPU (device) executes the computationally intensive tasks.

CUDA has been widely adopted in scientific and engineering applications where the computational demands are high. It has enabled researchers and engineers to tackle problems that were previously impossible due to computational constraints. For example, CUDA has been used in a variety of fields, including physics, chemistry, biology, and finance, to perform simulations, data analysis, and machine learning.

In addition to its programming interface and libraries, NVIDIA also provides a variety of tools to aid in CUDA development, including profilers and debuggers. These tools can help developers optimize their code and identify performance bottlenecks, which can be critical in achieving the best performance from a GPU-accelerated application. 

## **Example 1:** SAXPY in CUDA C

### CUDA C Source Code

```c
#include <stdio.h>

__global__ void saxpy(int n, float a, float *x, float *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

int main()
{
    int n = 1000000;
    float a = 2.0;
    float *x, *y, *d_x, *d_y;
    size_t size = n * sizeof(float);

    // Allocate memory on host
    x = (float*)malloc(size);
    y = (float*)malloc(size);

    // Initialize input vectors
    for (int i = 0; i < n; i++) {
        x[i] = 1.0;
        y[i] = 2.0;
    }

    // Allocate memory on device
    cudaMalloc(&d_x, size);
    cudaMalloc(&d_y, size);

    // Copy input vectors from host to device
    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);

    // Launch kernel on device
    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;
    saxpy<<<num_blocks, block_size>>>(n, a, d_x, d_y);

    // Copy result from device to host
    cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);

    // Free memory on device
    cudaFree(d_x);
    cudaFree(d_y);

    // Print result
    printf("Result: y[0] = %f\n", y[0]);

    // Free memory on host
    free(x);
    free(y);

    return 0;
}
```

### Compile the code

If the above code is named <code>saxpy.cu</code> it can be compiled as, e.g.,

```bash
module load cuda/12.9.1-fasrc01
nvcc -o saxpy.x saxpy.cu
```

### Example batch-job submission script

```bash
#!/bin/bash
#SBATCH -p gpu_test
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH --mem=12000
#SBATCH -J cuda_test
#SBATCH -o cuda_test.out
#SBATCH -e cuda_test.err
#SBATCH -t 30

# Load required modules
module load cuda/12.9.1-fasrc01

# Run the executable
./saxpy.x
```
Assuming the batch-job submission script is named <code>run.sbatch</code>, the jobs is sent to the queue, as usual, with:

```bash
sbatch run.sbatch
```
### Example Output

```bash
$ cat cuda_test.out 
Result: y[0] = 4.000000
```

## **Example 2:** SAXPY in CUDA Fortran

### CUDA Fortran Source Code

```fortran
module mathOps
contains
  attributes(global) subroutine saxpy(x, y, a)
    implicit none
    real :: x(:), y(:)
    real, value :: a
    integer :: i, n
    n = size(x)
    i = blockDim%x * (blockIdx%x - 1) + threadIdx%x
    if (i <= n) y(i) = y(i) + a*x(i)
  end subroutine saxpy
end module mathOps

program testSaxpy
  use mathOps
  use cudafor
  implicit none
  integer, parameter :: N = 1024000
  real :: x(N), y(N), a
  real, device :: x_d(N), y_d(N)
  type(dim3) :: grid, tBlock

  tBlock = dim3(256,1,1)
  grid = dim3(ceiling(real(N)/tBlock%x),1,1)

  call random_number(x)
  call random_number(y)
  call random_number(a)

  !x = 1.0; y = 2.0; a = 2.0
  x_d = x
  y_d = y
  call saxpy<<<grid, tBlock>>>(x_d, y_d, a)
  y = y_d

  print *, "Size of arrays: ", N
  print *, 'Grid             : ', grid
  print *, 'Threads per block: ', tBlock

  print *, "Constant a:", a
  print *, 'Average values ', sum(abs(x))/N, sum(abs(y))/N
end program testSaxpy
```
### Compile the code

If the above code is named <code>saxpy.cuf</code> it can be compiled as, e.g.,

```bash
module load nvhpc/24.11-fasrc01
nvfortran -o saxpy.x saxpy.cuf
```

### Example batch-job submission script

```bash
#!/bin/bash
#SBATCH -p gpu_test
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH --mem=12000
#SBATCH -J cuda_test
#SBATCH -o cuda_test.out
#SBATCH -e cuda_test.err
#SBATCH -t 30

# Load required modules
module load nvhpc/24.11-fasrc01

# Run the executable
./saxpy.x
```
Assuming the batch-job submission script is named <code>run_fort.sbatch</code>, the jobs is sent to the queue, as usual, with:

```bash
sbatch run_fort.sbatch
```
### Example Output

```bash
$ cat cuda_test.out 
 Size of arrays:       1024000
 Grid             :          4000            1            1
 Threads per block:           256            1            1
 Constant a:   0.1387444    
 Average values    0.4998179       0.5694433 
```

## References

* [About CUDA](https://developer.nvidia.com/about-cuda)
* [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
