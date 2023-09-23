# Using GPU-Accelerated Libraries

<img src="gpu-libs-logo.png" alt="gpu-libs-logo" width="400"/>

[NVIDIA provides a suite of libraries optimized for GPU-accelerated computing](https://developer.nvidia.com/gpu-accelerated-libraries), which can be used to speed up a variety of scientific computations. Some of the most commonly used libraries include CUDA, cuDNN, and cuBLAS. CUDA is a parallel computing platform and programming model that provides direct access to the GPU's processing power, while cuDNN and cuBLAS are libraries for deep learning and linear algebra operations, respectively. These libraries can be used in conjunction with popular programming languages such as C++, Python, and MATLAB, making it easy to take advantage of GPU acceleration without requiring extensive knowledge of GPU programming. 

## Example: cuBLAS

The SAXPY operation is a fundamental linear algebra operation that calculates the linear combination of two vectors,

$$
z = \alpha x + y,
$$

where $x$, $y$, and $z$ are vectors, and $\alpha$ is a scalar. It is a widely used operation in many scientific and engineering applications, including machine learning and numerical simulations. In this context, it is often desirable to accelerate SAXPY computations using GPUs to take advantage of their parallel computing power.

To illustrate the SAXPY operation in CUDA, we will provide a C example using the BLAS library and a cuBLAS example. The C example will demonstrate the basic concept of SAXPY, and the cuBLAS example will show how to leverage the CUDA library to perform the operation on the GPU. In this illustrative example the vector dimension is rather small (vector dimension = 5), but in realistic applications it is usually in order of millions, or billions.

### Serial C Code

```c
#include <stdio.h>
#include <gsl/gsl_cblas.h>

int main()
{
    const int n = 5;
    const float alpha = 2.0;
    float x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    float y[] = {2.0, 4.0, 6.0, 8.0, 10.0};

    // Perform SAXPY operation
    cblas_saxpy(n, alpha, x, 1, y, 1);

    // Print final values
    printf("SAXPY result: ");
    for (int i = 0; i < n; i++) {
        printf("%f ", y[i]);
    }
    printf("\n");

    return 0;
}
```
**Compiling:**

If the above code is named, e.g., <code>saxpy_blas.c</code>, it can be compiled with:

```bash
# Load the required modules
module load gcc/13.2.0-fasrc01
# Compile the code
gcc -o saxpy_blas.x saxpy_blas.c -lgslcblas
```


### GPU / cuBLAS C Code

```c
#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main(){
    const int n = 5;
    const float alpha = 2.0;
    float x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    float y[] = {2.0, 4.0, 6.0, 8.0, 10.0};

    // Initialize cuBLAS context
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Allocate memory on device
    float *d_x, *d_y;
    cudaMalloc(&d_x, n*sizeof(float));
    cudaMalloc(&d_y, n*sizeof(float));

    // Copy data to device
    cudaMemcpy(d_x, x, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n*sizeof(float), cudaMemcpyHostToDevice);

    // Perform SAXPY operation
    cublasSaxpy(handle, n, &alpha, d_x, 1, d_y, 1);

    // Copy data back to host
    cudaMemcpy(y, d_y, n*sizeof(float), cudaMemcpyDeviceToHost);

    // Destroy cuBLAS context
    cublasDestroy(handle);

    // Free memory on device
    cudaFree(d_x);
    cudaFree(d_y);

    // Print final values
    printf("SAXPY result: ");
    for (int i = 0; i < n; i++) {
        printf("%f ", y[i]);
    }
    printf("\n");

    return 0;
}
```

**Compiling:**

If the above code is named, e.g., <code>saxpy_cublas.c</code>, it can be compiled with:

```bash
module load cuda/12.2.0-fasrc01 gcc/13.2.0-fasrc01
gcc -o saxpy_cublas.x saxpy_cublas.c -lcudart -lcublas 
```

**Executing:**

Below is an example batch-job submission script, <code>run.sbatch</code>:

```bash
#!/bin/bash
#SBATCH -p gpu_test
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH --mem=12000
#SBATCH -J saxpy_example
#SBATCH -o saxpy_example.out
#SBATCH -e saxpy_example.err
#SBATCH -t 30

# Load required modules
module load gcc/13.2.0-fasrc01 cuda/12.2.0-fasrc01

# Run the executable
./saxpy_cublas.x
```

The jobs is sent to the queue with:

```bash
sbatch run.sbatch
```
**Example output:**

```bash
cat saxpy_example.out
SAXPY result: 4.000000 8.000000 12.000000 16.000000 20.000000 
```

## References:

* [cuBLAS](https://developer.nvidia.com/cublas)
* [N Ways to SAXPY](https://developer.nvidia.com/blog/n-ways-to-saxpy-demonstrating-the-breadth-of-gpu-programming-options/)
* [NVIDIA HPC SDK](https://developer.nvidia.com/hpc-sdk)