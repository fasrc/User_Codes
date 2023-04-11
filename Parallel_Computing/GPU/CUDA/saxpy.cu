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
