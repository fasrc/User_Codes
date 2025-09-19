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

