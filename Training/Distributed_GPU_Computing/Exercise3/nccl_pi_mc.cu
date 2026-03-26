#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cub/device/device_reduce.cuh>
#include <nccl.h>

#define NSAMPLES_DEFAULT 1000000000LL

#define CUDA_CHECK(call) \
do { cudaError_t e=(call); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} } while(0)

#define NCCL_CHECK(call) \
do { ncclResult_t r=(call); if(r!=ncclSuccess){fprintf(stderr,"NCCL %s:%d %s\n",__FILE__,__LINE__,ncclGetErrorString(r));exit(1);} } while(0)

__device__ __host__ static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

__device__ static inline double uniform_01(uint64_t &state) {
    state = state * 2862933555777941757ULL + 3037000493ULL;
    return (double)(splitmix64(state) >> 11) * (1.0 / 9007199254740992.0);
}

__global__ void monte_carlo_pi_kernel(unsigned long long *counts,
                                      long long nsamples, uint64_t seed) {
    long long tid    = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x  * blockDim.x;
    uint64_t state = splitmix64(seed + (uint64_t)tid * 6364136223846793005ULL);
    unsigned long long hits = 0;
    for (long long i = tid; i < nsamples; i += stride) {
        double x = uniform_01(state), y = uniform_01(state);
        if (x*x + y*y <= 1.0) hits++;
    }
    counts[tid] = hits;
}

int main(int argc, char **argv) {
    int nGPUs;
    CUDA_CHECK(cudaGetDeviceCount(&nGPUs));

    long long nsamples_total = (argc > 1) ? atoll(argv[1]) : NSAMPLES_DEFAULT;

    std::vector<int>          devs(nGPUs);
    std::vector<ncclComm_t>   comms(nGPUs);
    std::vector<cudaStream_t> streams(nGPUs);

    for (int i = 0; i < nGPUs; i++) devs[i] = i;
    NCCL_CHECK(ncclCommInitAll(comms.data(), nGPUs, devs.data()));

    // Single pair of events on GPU 0 for total wall time
    cudaEvent_t ev_total_start, ev_total_stop;
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaEventCreate(&ev_total_start));
    CUDA_CHECK(cudaEventCreate(&ev_total_stop));

    const int    blocks   = 128, threads = 256;
    const long long nthreads = (long long)blocks * threads;

    long long base_per_gpu = nsamples_total / nGPUs;
    long long remainder    = nsamples_total % nGPUs;

    std::vector<unsigned long long*> d_counts(nGPUs), d_local(nGPUs), d_global(nGPUs);
    std::vector<void*>               d_cub_tmp(nGPUs, nullptr);
    std::vector<size_t>              cub_tmp_bytes(nGPUs, 0);

    for (int i = 0; i < nGPUs; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
        CUDA_CHECK(cudaMalloc(&d_counts[i], nthreads * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMalloc(&d_local[i],  sizeof(unsigned long long)));
        CUDA_CHECK(cudaMalloc(&d_global[i], sizeof(unsigned long long)));
        cub::DeviceReduce::Sum(nullptr, cub_tmp_bytes[i],
                               d_counts[i], d_local[i], (int)nthreads, streams[i]);
        CUDA_CHECK(cudaMalloc(&d_cub_tmp[i], cub_tmp_bytes[i]));
    }

    // Record total start on GPU 0's stream before any work is enqueued
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaEventRecord(ev_total_start, streams[0]));

    ncclGroupStart();
    for (int i = 0; i < nGPUs; i++) {
        long long my_samples = base_per_gpu + (i == 0 ? remainder : 0LL);
        uint64_t  gpu_seed   = splitmix64(123456789ULL + (uint64_t)i * 0xA4D4ADFE8B8A3E1FULL);

        CUDA_CHECK(cudaSetDevice(i));

        monte_carlo_pi_kernel<<<blocks, threads, 0, streams[i]>>>(
            d_counts[i], my_samples, gpu_seed);

        cub::DeviceReduce::Sum(d_cub_tmp[i], cub_tmp_bytes[i],
                               d_counts[i], d_local[i], (int)nthreads, streams[i]);

        NCCL_CHECK(ncclAllReduce(d_local[i], d_global[i], 1,
                                 ncclUint64, ncclSum, comms[i], streams[i]));
    }
    ncclGroupEnd();

    // Record total stop on GPU 0's stream — after NCCL has flushed its ops
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaEventRecord(ev_total_stop, streams[0]));

    // Sync all streams and print per-GPU diagnostics
    unsigned long long h_local, h_global = 0;
    for (int i = 0; i < nGPUs; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));

        long long my_samples = base_per_gpu + (i == 0 ? remainder : 0LL);
        CUDA_CHECK(cudaMemcpy(&h_local, d_local[i],
                              sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        printf("GPU %d/%d | samples=%lld | local hits=%llu\n",
               i, nGPUs, my_samples, h_local);

        if (i == 0)
            CUDA_CHECK(cudaMemcpy(&h_global, d_global[i],
                                  sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    }

    // Total elapsed time (GPU 0's stream covers the full pipeline)
    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventSynchronize(ev_total_stop));
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, ev_total_start, ev_total_stop));

    constexpr double PI_EXACT    = M_PI;
    double pi_estimate   = 4.0 * (double)h_global / (double)nsamples_total;
    double abs_error     = fabs(pi_estimate - PI_EXACT);
    double rel_error_pct = (abs_error / PI_EXACT) * 100.0;

    printf("--------------------------------------------------\n");
    printf("Exact PI       = %.8f\n", PI_EXACT);
    printf("Estimated PI   = %.8f\n", pi_estimate);
    printf("Absolute error = %.8f\n", abs_error);
    printf("Relative error = %.8f %%\n", rel_error_pct);
    printf("Total hits     = %llu\n",   h_global);
    printf("Total samples  = %lld\n",   nsamples_total);
    printf("Total time     = %.3f ms\n", total_ms);

    // Cleanup
    for (int i = 0; i < nGPUs; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        ncclCommDestroy(comms[i]);
        cudaStreamDestroy(streams[i]);
        cudaFree(d_counts[i]);
        cudaFree(d_local[i]);
        cudaFree(d_global[i]);
        cudaFree(d_cub_tmp[i]);
    }
    cudaEventDestroy(ev_total_start);
    cudaEventDestroy(ev_total_stop);

    return 0;
}
