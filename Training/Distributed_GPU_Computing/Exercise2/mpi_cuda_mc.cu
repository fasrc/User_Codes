#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>
#include <cuda_runtime.h>

//
// Default total number of Monte Carlo samples across all MPI ranks.
//
#define NSAMPLES_DEFAULT 1000000000LL

//
// CUDA error checking.
//
#define CUDA_CHECK(call)                                                       \
do {                                                                           \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                           \
                __FILE__, __LINE__, cudaGetErrorString(err));                  \
        MPI_Abort(MPI_COMM_WORLD, 1);                                          \
    }                                                                          \
} while (0)

//
// Simple 64-bit SplitMix hash to decorrelate seeds.
// Good for turning rank/thread/sample identifiers into randomized states.
//
__device__ __host__ static inline uint64_t splitmix64(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    x = x ^ (x >> 31);
    return x;
}

//
// Fast stateless RNG step.
// This advances a 64-bit state and returns a new 64-bit pseudo-random integer.
//
__device__ static inline uint64_t rng_next(uint64_t &state)
{
    state = state * 2862933555777941757ULL + 3037000493ULL;
    return splitmix64(state);
}

//
// Convert a 64-bit random integer into a uniform double in [0,1).
// We use the top 53 bits so it fits the precision of IEEE double.
//
__device__ static inline double uniform_01(uint64_t &state)
{
    uint64_t r = rng_next(state);
    return (double)(r >> 11) * (1.0 / 9007199254740992.0); // 2^53
}

//
// Kernel 1:
// Each thread generates a strided subset of Monte Carlo samples.
// It counts how many random (x,y) points fall inside the quarter circle.
//
// A point is inside if:
//   x^2 + y^2 <= 1
//
// Output:
//   counts[tid] = number of hits produced by this thread
//
__global__ void monte_carlo_pi_kernel(unsigned long long *counts,
                                      long long nsamples_local,
                                      uint64_t base_seed)
{
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    //
    // Give each thread a different RNG state.
    //
    uint64_t state = splitmix64(base_seed ^ (uint64_t)tid);

    unsigned long long local_hits = 0ULL;

    //
    // Strided loop over this rank's samples.
    //
    for (long long i = tid; i < nsamples_local; i += stride) {
        double x = uniform_01(state);
        double y = uniform_01(state);

        double r2 = x * x + y * y;
        if (r2 <= 1.0) {
            local_hits++;
        }
    }

    counts[tid] = local_hits;
}

//
// Kernel 2:
// Parallel reduction for unsigned long long counts.
// Reduces input[] into one partial sum per block.
//
__global__ void reduce_ull_sum(const unsigned long long *input,
                               unsigned long long *output,
                               long long n)
{
    extern __shared__ unsigned long long sdata[];

    unsigned int tid = threadIdx.x;
    long long global_idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;

    unsigned long long x = 0ULL;
    if (global_idx < n) {
        x = input[global_idx];
    }

    sdata[tid] = x;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

//
// Choose a dynamic launch configuration for the Monte Carlo kernel.
//
static void choose_compute_launch_config(long long nsamples_local,
                                         int device_id,
                                         int *blocks,
                                         int *threads)
{
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    int min_grid_size = 0;
    int block_size = 0;

    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
        &min_grid_size,
        &block_size,
        monte_carlo_pi_kernel,
        0,
        0));

    if (block_size <= 0) {
        block_size = 256;
    }

    //
    // Heuristic:
    // target a few thousand samples per thread on average.
    //
    const long long target_samples_per_thread = 4096;
    long long wanted_threads =
        (nsamples_local + target_samples_per_thread - 1) / target_samples_per_thread;

    if (wanted_threads < block_size) {
        wanted_threads = block_size;
    }

    long long block_count = (wanted_threads + block_size - 1) / block_size;

    int sm_count = prop.multiProcessorCount;
    long long max_useful_blocks = (long long)sm_count * 32;

    if (block_count < min_grid_size) {
        block_count = min_grid_size;
    }
    if (block_count > max_useful_blocks) {
        block_count = max_useful_blocks;
    }

    long long max_blocks_from_work = (nsamples_local + block_size - 1) / block_size;
    if (max_blocks_from_work > 0 && block_count > max_blocks_from_work) {
        block_count = max_blocks_from_work;
    }

    if (block_count < 1) {
        block_count = 1;
    }

    *threads = block_size;
    *blocks = (int)block_count;
}

//
// Repeatedly reduce a device array of unsigned long long values
// until only one scalar remains. Copy that scalar to host.
//
static unsigned long long gpu_reduce_to_scalar(unsigned long long *d_input,
                                               long long n,
                                               int reduce_threads)
{
    unsigned long long *d_in = d_input;
    unsigned long long *d_out = NULL;
    unsigned long long result = 0ULL;

    while (n > 1) {
        long long blocks = (n + reduce_threads - 1) / reduce_threads;
        size_t out_bytes = (size_t)blocks * sizeof(unsigned long long);

        CUDA_CHECK(cudaMalloc((void **)&d_out, out_bytes));

        reduce_ull_sum<<<(int)blocks, reduce_threads,
                         reduce_threads * sizeof(unsigned long long)>>>(
            d_in, d_out, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        if (d_in != d_input) {
            CUDA_CHECK(cudaFree(d_in));
        }

        d_in = d_out;
        d_out = NULL;
        n = blocks;
    }

    CUDA_CHECK(cudaMemcpy(&result, d_in, sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));

    if (d_in != d_input) {
        CUDA_CHECK(cudaFree(d_in));
    }

    return result;
}

int main(int argc, char **argv)
{
    int myid, nproc;

    //
    // Total and local Monte Carlo samples.
    //
    long long nsamples_total = NSAMPLES_DEFAULT;
    long long nsamples_local, remainder;

    //
    // GPU / MPI-local placement info.
    //
    int num_devices = 0;
    int local_rank = 0;
    int local_size = 0;
    int device_id = 0;

    //
    // CUDA launch configuration.
    //
    int compute_blocks = 0;
    int compute_threads = 0;
    long long total_compute_threads = 0;

    //
    // Per-thread counts on device.
    //
    unsigned long long *d_counts = NULL;

    //
    // Local/global hit counts and final pi estimate.
    //
    unsigned long long hits_local = 0ULL;
    unsigned long long hits_global = 0ULL;
    double pi_estimate = 0.0;

    //
    // Seed base for RNG. Can be overridden from command line.
    //
    uint64_t base_seed = 123456789ULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    //
    // Optional command line:
    //   argv[1] = total number of samples
    //   argv[2] = seed
    //
    if (argc > 1) {
        nsamples_total = atoll(argv[1]);
        if (nsamples_total <= 0) {
            if (myid == 0) {
                fprintf(stderr, "Invalid number of samples: %s\n", argv[1]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (argc > 2) {
        base_seed = (uint64_t)strtoull(argv[2], NULL, 10);
    }

    //
    // Split communicator by shared-memory node so local_rank is per-node.
    //
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD,
                        MPI_COMM_TYPE_SHARED,
                        0,
                        MPI_INFO_NULL,
                        &local_comm);

    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &local_size);

    //
    // Pick GPU by local rank.
    //
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    if (num_devices <= 0) {
        fprintf(stderr, "Rank %d: no CUDA devices found on this node\n", myid);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    device_id = local_rank % num_devices;
    CUDA_CHECK(cudaSetDevice(device_id));

    //
    // Split samples across MPI ranks, including remainder.
    //
    nsamples_local = nsamples_total / nproc;
    remainder = nsamples_total % nproc;

    if (myid < remainder) {
        nsamples_local += 1;
    }

    //
    // Choose dynamic CUDA launch size.
    //
    choose_compute_launch_config(nsamples_local, device_id,
                                 &compute_blocks, &compute_threads);

    total_compute_threads = (long long)compute_blocks * compute_threads;

    //
    // Allocate one hit counter per CUDA thread.
    //
    CUDA_CHECK(cudaMalloc((void **)&d_counts,
                          (size_t)total_compute_threads * sizeof(unsigned long long)));

    //
    // Make the seed rank-dependent so different MPI ranks do not reuse streams.
    //
    uint64_t rank_seed = splitmix64(base_seed ^ (uint64_t)myid);

    //
    // Launch Monte Carlo kernel.
    //
    monte_carlo_pi_kernel<<<compute_blocks, compute_threads>>>(
        d_counts,
        nsamples_local,
        rank_seed
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    //
    // Reduce per-thread hit counts to one local count on the GPU.
    //
    hits_local = gpu_reduce_to_scalar(d_counts, total_compute_threads, 256);

    //
    // Sum hit counts across MPI ranks.
    //
    MPI_Allreduce(&hits_local, &hits_global, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    //
    // Final Monte Carlo pi estimate.
    //
    pi_estimate = 4.0 * (double)hits_global / (double)nsamples_total;

    //
    // Print per-rank diagnostics.
    //
    printf("rank %d/%d | local_rank %d/%d | GPU %d | local samples = %lld | "
           "blocks = %d | threads/block = %d | total threads = %lld | "
           "local hits = %llu\n",
           myid, nproc,
           local_rank, local_size,
           device_id,
           nsamples_local,
           compute_blocks,
           compute_threads,
           total_compute_threads,
           (unsigned long long)hits_local);

    //
    // Print final result once.
    //
    if (myid == 0) {
        printf("Monte Carlo PI = %.15f\n", pi_estimate);
        printf("Total hits      = %llu\n", (unsigned long long)hits_global);
        printf("Total samples   = %lld\n", nsamples_total);
    }

    CUDA_CHECK(cudaFree(d_counts));
    MPI_Comm_free(&local_comm);
    MPI_Finalize();

    return 0;
}
