#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <mpi.h>
#include <cuda_runtime.h>

#define NSAMPLES_DEFAULT 1000000000LL

#define CUDA_CHECK(call)                                                       \
do {                                                                           \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                          \
                __FILE__, __LINE__, cudaGetErrorString(err));                  \
        MPI_Abort(MPI_COMM_WORLD, 1);                                          \
    }                                                                          \
} while (0)

// ---------------------------------------------------------------------------
// SplitMix64 hash — used to generate well-scrambled seeds.
// ---------------------------------------------------------------------------
__device__ __host__ static inline uint64_t splitmix64(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

__device__ static inline uint64_t rng_next(uint64_t &state)
{
    state = state * 2862933555777941757ULL + 3037000493ULL;
    return splitmix64(state);
}

__device__ static inline double uniform_01(uint64_t &state)
{
    return (double)(rng_next(state) >> 11) * (1.0 / 9007199254740992.0);
}

// ---------------------------------------------------------------------------
// Kernel 1: Monte Carlo sampling.
// Each thread independently counts hits in its strided subset of samples.
//
// Seeding strategy: each thread gets a unique seed derived from the rank seed
// using a large-prime multiplicative spacing, giving strong decorrelation
// between threads regardless of block/grid geometry.
// ---------------------------------------------------------------------------
__global__ void monte_carlo_pi_kernel(unsigned long long *counts,
                                      long long nsamples_local,
                                      uint64_t rank_seed)
{
    long long tid    = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x  * blockDim.x;

    uint64_t state = splitmix64(rank_seed + (uint64_t)tid * 6364136223846793005ULL);

    unsigned long long local_hits = 0ULL;
    for (long long i = tid; i < nsamples_local; i += stride) {
        double x = uniform_01(state);
        double y = uniform_01(state);
        if (x * x + y * y <= 1.0) local_hits++;
    }
    counts[tid] = local_hits;
}

// ---------------------------------------------------------------------------
// Kernel 2: parallel reduction using warp shuffles.
//
// Replaces the shared-memory tree reduction, which silently produces wrong
// results when blockDim.x is not a power of two — a real risk when block
// size comes from cudaOccupancyMaxPotentialBlockSize.
// ---------------------------------------------------------------------------
__global__ void reduce_ull_sum(const unsigned long long *input,
                                unsigned long long *output, long long n)
{
    unsigned long long val = 0ULL;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) val = input[idx];

    // Step 1: warp-level reduction via shuffle
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);

    // Step 2: one value per warp into shared memory
    __shared__ unsigned long long sdata[32];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    if (lane == 0) sdata[wid] = val;
    __syncthreads();

    // Step 3: first warp reduces the per-warp values
    if (wid == 0) {
        val = (lane < (blockDim.x >> 5)) ? sdata[lane] : 0ULL;
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        if (lane == 0) output[blockIdx.x] = val;
    }
}

// ---------------------------------------------------------------------------
// Occupancy-aware launch configuration for the MC kernel.
// ---------------------------------------------------------------------------
static void choose_compute_launch_config(long long nsamples_local,
                                         int device_id,
                                         int *blocks,
                                         int *threads)
{
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    int min_grid_size = 0, block_size = 0;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
        &min_grid_size, &block_size, monte_carlo_pi_kernel, 0, 0));
    if (block_size <= 0) block_size = 256;

    const long long target_samples_per_thread = 4096;
    long long wanted_threads =
        (nsamples_local + target_samples_per_thread - 1) / target_samples_per_thread;
    if (wanted_threads < block_size) wanted_threads = block_size;

    long long block_count = (wanted_threads + block_size - 1) / block_size;

    long long max_useful_blocks = (long long)prop.multiProcessorCount * 32;
    if (block_count < min_grid_size)     block_count = min_grid_size;
    if (block_count > max_useful_blocks) block_count = max_useful_blocks;

    long long max_from_work = (nsamples_local + block_size - 1) / block_size;
    if (max_from_work > 0 && block_count > max_from_work)
        block_count = max_from_work;
    if (block_count < 1) block_count = 1;

    *threads = block_size;
    *blocks  = (int)block_count;
}

// ---------------------------------------------------------------------------
// Iteratively reduce a device array of unsigned long long values to a single
// scalar on the GPU. Only one value is ever copied back to the host.
// ---------------------------------------------------------------------------
static unsigned long long gpu_reduce_to_scalar(unsigned long long *d_input,
                                                long long n,
                                                int reduce_threads)
{
    unsigned long long *d_in  = d_input;
    unsigned long long *d_out = NULL;

    while (n > 1) {
        long long blocks = (n + reduce_threads - 1) / reduce_threads;
        CUDA_CHECK(cudaMalloc((void **)&d_out,
                              (size_t)blocks * sizeof(unsigned long long)));

        reduce_ull_sum<<<(int)blocks, reduce_threads>>>(d_in, d_out, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        if (d_in != d_input) CUDA_CHECK(cudaFree(d_in));
        d_in  = d_out;
        d_out = NULL;
        n     = blocks;
    }

    unsigned long long result = 0ULL;
    CUDA_CHECK(cudaMemcpy(&result, d_in, sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
    if (d_in != d_input) CUDA_CHECK(cudaFree(d_in));
    return result;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int myid, nproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    long long nsamples_total = NSAMPLES_DEFAULT;
    uint64_t  base_seed      = 123456789ULL;

    if (argc > 1) {
        nsamples_total = atoll(argv[1]);
        if (nsamples_total <= 0) {
            if (myid == 0)
                fprintf(stderr, "Invalid number of samples: %s\n", argv[1]);
            MPI_Finalize();
            return 1;
        }
    }
    if (argc > 2)
        base_seed = (uint64_t)strtoull(argv[2], NULL, 10);

    // Shared-memory communicator for local rank → GPU mapping
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,
                        0, MPI_INFO_NULL, &local_comm);
    int local_rank, local_size;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &local_size);

    int num_devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    if (num_devices <= 0) {
        fprintf(stderr, "Rank %d: no CUDA devices found\n", myid);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int device_id = local_rank % num_devices;
    CUDA_CHECK(cudaSetDevice(device_id));

    // Distribute samples across ranks.
    // Ranks 0..(remainder-1) each get one extra sample.
    // No start_bin offset is needed — MC samples are position-independent,
    // unlike the integration code where global bin index determines x.
    long long nsamples_local = nsamples_total / nproc
                             + (myid < (nsamples_total % nproc) ? 1 : 0);

    int compute_blocks, compute_threads;
    choose_compute_launch_config(nsamples_local, device_id,
                                 &compute_blocks, &compute_threads);
    long long total_compute_threads = (long long)compute_blocks * compute_threads;

    unsigned long long *d_counts = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_counts,
                          (size_t)total_compute_threads
                          * sizeof(unsigned long long)));

    // Large-prime multiplicative spacing gives strong decorrelation between
    // ranks regardless of how close their myid values are numerically.
    uint64_t rank_seed = splitmix64(base_seed
                         + (uint64_t)myid * 0xA4D4ADFE8B8A3E1FULL);

    // ------------------------------------------------------------------
    // Timed section
    // ------------------------------------------------------------------
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    monte_carlo_pi_kernel<<<compute_blocks, compute_threads>>>(
        d_counts, nsamples_local, rank_seed);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long long hits_local =
        gpu_reduce_to_scalar(d_counts, total_compute_threads, 256);

    // MPI_Reduce: only rank 0 needs hits_global to compute pi
    unsigned long long hits_global = 0ULL;
    MPI_Reduce(&hits_local, &hits_global, 1,
               MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double tlocal = MPI_Wtime() - t0;

    // True parallel wall time = slowest rank
    double tmax = 0.0;
    MPI_Reduce(&tlocal, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // ------------------------------------------------------------------
    // Ordered per-rank diagnostics
    // ------------------------------------------------------------------
    char diag[512];
    int  diag_len = snprintf(diag, sizeof(diag),
        "rank %d/%d | local_rank %d/%d | GPU %d | samples = %lld | "
        "blocks = %d | threads/block = %d | total threads = %lld | "
        "hits = %llu | time = %.6f s",
        myid, nproc, local_rank, local_size, device_id,
        nsamples_local, compute_blocks, compute_threads,
        total_compute_threads, hits_local, tlocal);

    if (myid == 0) {
        printf("%s\n", diag);
        fflush(stdout);
        char recv_buf[512];
        for (int src = 1; src < nproc; src++) {
            MPI_Recv(recv_buf, sizeof(recv_buf), MPI_CHAR,
                     src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("%s\n", recv_buf);
            fflush(stdout);
        }
    } else {
        MPI_Send(diag, diag_len + 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    }

    // ------------------------------------------------------------------
    // Final results — rank 0 only
    // ------------------------------------------------------------------
    if (myid == 0) {
        const double pi_exact  = 3.14159265358979323846;
        double pi_estimate     = 4.0 * (double)hits_global
                                     / (double)nsamples_total;
        double abs_error       = fabs(pi_estimate - pi_exact);
        double rel_error_pct   = (abs_error / pi_exact) * 100.0;

        printf("\n========== FINAL RESULTS ==========\n");
        printf("Exact PI       = %.8f\n", pi_exact);
        printf("Estimated PI   = %.8f\n", pi_estimate);
        printf("Absolute error = %.8e\n", abs_error);
        printf("Relative error = %.8e %%\n", rel_error_pct);
        printf("Total hits     = %llu\n",   hits_global);
        printf("Total samples  = %lld\n",   nsamples_total);
        printf("Wall time      = %.6f s\n",  tmax);
        fflush(stdout);
    }

    CUDA_CHECK(cudaFree(d_counts));
    MPI_Comm_free(&local_comm);
    MPI_Finalize();
    return 0;
}
