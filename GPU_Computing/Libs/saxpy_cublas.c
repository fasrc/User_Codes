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
    cudaMalloc((void**)&d_x, n * sizeof(float)); // Cast to void**
    cudaMalloc((void**)&d_y, n * sizeof(float)); // Cast to void**

    // Copy data to device
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);

    // Perform SAXPY operation
    cublasSaxpy(handle, n, &alpha, d_x, 1, d_y, 1);

    // Copy data back to host
    cudaMemcpy(y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);

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

