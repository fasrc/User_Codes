#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <vector>
#include <mpi.h>
#include <cuda_runtime.h>
#include <cub/device/device_reduce.cuh>
#include <nccl.h>

#define NSAMPLES_DEFAULT 1000000000LL

#define CUDA_CHECK(call)                                                       \
do {                                                                           \
    cudaError_t e = (call);                                                    \
    if (e != cudaSuccess) {                                                    \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                          \
                __FILE__, __LINE__, cudaGetErrorString(e));                    \
        MPI_Abort(MPI_COMM_WORLD, 1);                                          \
    }                                                                          \
} while (0)

#define NCCL_CHECK(call)                                                       \
do {                                                                           \
    ncclResult_t r = (call);                                                   \
    if (r != ncclSuccess) {                                                    \
        fprintf(stderr, "NCCL error at %s:%d: %s\n",                          \
                __FILE__, __LINE__, ncclGetErrorString(r));                    \
        MPI_Abort(MPI_COMM_WORLD, 1);                                          \
    }                                                                          \
} while (0)

// ---------------------------------------------------------------------------
// RNG — splitmix64 + LCG, identical to the single-node NCCL version
// ---------------------------------------------------------------------------
__device__ __host__ static inline uint64_t splitmix64(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

__device__ static inline double uniform_01(uint64_t &state)
{
    state = state * 2862933555777941757ULL + 3037000493ULL;
    return (double)(splitmix64(state) >> 11) * (1.0 / 9007199254740992.0);
}

// ---------------------------------------------------------------------------
// Kernel: Monte Carlo sampling — unchanged from the single-node version.
// Large-prime thread spacing gives strong per-thread RNG decorrelation.
// ---------------------------------------------------------------------------
__global__ void monte_carlo_pi_kernel(unsigned long long *counts,
                                      long long nsamples_local,
                                      uint64_t rank_seed)
{
    long long tid    = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x  * blockDim.x;

    uint64_t state = splitmix64(rank_seed
                     + (uint64_t)tid * 6364136223846793005ULL);

    unsigned long long hits = 0ULL;
    for (long long i = tid; i < nsamples_local; i += stride) {
        double x = uniform_01(state);
        double y = uniform_01(state);
        if (x * x + y * y <= 1.0) hits++;
    }
    counts[tid] = hits;
}

// ---------------------------------------------------------------------------
// Occupancy-aware launch configuration
// ---------------------------------------------------------------------------
static void choose_launch_config(long long nsamples_local, int device_id,
                                 int *blocks, int *threads)
{
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    int min_grid = 0, block_size = 0;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
        &min_grid, &block_size, monte_carlo_pi_kernel, 0, 0));
    if (block_size <= 0) block_size = 256;

    const long long tgt = 4096;
    long long wanted = (nsamples_local + tgt - 1) / tgt;
    if (wanted < block_size) wanted = block_size;

    long long nblocks = (wanted + block_size - 1) / block_size;
    long long max_useful = (long long)prop.multiProcessorCount * 32;
    if (nblocks < min_grid)    nblocks = min_grid;
    if (nblocks > max_useful)  nblocks = max_useful;

    long long max_work = (nsamples_local + block_size - 1) / block_size;
    if (max_work > 0 && nblocks > max_work) nblocks = max_work;
    if (nblocks < 1) nblocks = 1;

    *threads = block_size;
    *blocks  = (int)nblocks;
}

int main(int argc, char **argv)
{
    // -----------------------------------------------------------------------
    // MPI initialisation
    // -----------------------------------------------------------------------
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
                fprintf(stderr, "Invalid sample count: %s\n", argv[1]);
            MPI_Finalize(); return 1;
        }
    }
    if (argc > 2)
        base_seed = (uint64_t)strtoull(argv[2], NULL, 10);

    // -----------------------------------------------------------------------
    // GPU assignment: use shared-memory communicator to get local rank,
    // then map local_rank -> GPU round-robin
    // -----------------------------------------------------------------------
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

    // -----------------------------------------------------------------------
    // NCCL bootstrapping via MPI
    //
    // This replaces ncclCommInitAll from the single-node version.
    // Rank 0 generates a unique rendezvous token; MPI_Bcast distributes it
    // to all ranks (potentially across nodes). Every rank then calls
    // ncclCommInitRank to join the communicator with a consistent
    // (unique_id, nranks, global_rank) triple.
    //
    // NCCL then handles all GPU-to-GPU transfers — NVLink on-node,
    // InfiniBand (or RoCE) across nodes — transparently.
    // -----------------------------------------------------------------------
    ncclUniqueId nccl_id;
    if (myid == 0) NCCL_CHECK(ncclGetUniqueId(&nccl_id));
    MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);

    ncclComm_t    nccl_comm;
    cudaStream_t  stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    NCCL_CHECK(ncclCommInitRank(&nccl_comm, nproc, nccl_id, myid));

    // -----------------------------------------------------------------------
    // Work distribution — same remainder logic as the MPI MC version
    // -----------------------------------------------------------------------
    long long nsamples_local = nsamples_total / nproc
                             + (myid < (nsamples_total % nproc) ? 1 : 0);

    int compute_blocks, compute_threads;
    choose_launch_config(nsamples_local, device_id,
                         &compute_blocks, &compute_threads);
    long long total_threads = (long long)compute_blocks * compute_threads;

    // -----------------------------------------------------------------------
    // Device memory
    // -----------------------------------------------------------------------
    unsigned long long *d_counts = NULL;
    unsigned long long *d_hits   = NULL;   // local reduced scalar
    unsigned long long *d_global = NULL;   // global reduced scalar (AllReduce)
    void               *d_tmp    = NULL;
    size_t              tmp_bytes = 0;

    CUDA_CHECK(cudaMalloc(&d_counts,
               (size_t)total_threads * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_hits,   sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_global, sizeof(unsigned long long)));

    // Query CUB temp storage size
    cub::DeviceReduce::Sum(nullptr, tmp_bytes,
                           d_counts, d_hits, (int)total_threads, stream);
    CUDA_CHECK(cudaMalloc(&d_tmp, tmp_bytes));

    // Per-rank RNG seed: large-prime spacing for strong decorrelation
    uint64_t rank_seed = splitmix64(base_seed
                         + (uint64_t)myid * 0xA4D4ADFE8B8A3E1FULL);

    // -----------------------------------------------------------------------
    // Timed section
    // -----------------------------------------------------------------------
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // 1. MC kernel
    monte_carlo_pi_kernel<<<compute_blocks, compute_threads, 0, stream>>>(
        d_counts, nsamples_local, rank_seed);
    CUDA_CHECK(cudaGetLastError());

    // 2. CUB parallel reduction: per-thread counts -> single local hit total
    cub::DeviceReduce::Sum(d_tmp, tmp_bytes,
                           d_counts, d_hits, (int)total_threads, stream);

    // 3. NCCL AllReduce: sum local hits across all GPUs on all nodes
    //    NCCL uses NVLink intra-node, InfiniBand/RoCE inter-node
    NCCL_CHECK(ncclAllReduce(d_hits, d_global, 1,
                             ncclUint64, ncclSum, nccl_comm, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    MPI_Barrier(MPI_COMM_WORLD);
    double tlocal = MPI_Wtime() - t0;

    // True wall time = slowest rank
    double tmax = 0.0;
    MPI_Reduce(&tlocal, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // -----------------------------------------------------------------------
    // Read back results
    // -----------------------------------------------------------------------
    unsigned long long h_hits_local  = 0ULL;
    unsigned long long h_hits_global = 0ULL;
    CUDA_CHECK(cudaMemcpy(&h_hits_local,  d_hits,
               sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_hits_global, d_global,
               sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    // -----------------------------------------------------------------------
    // Ordered per-rank diagnostics
    // -----------------------------------------------------------------------
    char diag[512];
    int  diag_len = snprintf(diag, sizeof(diag),
        "rank %d/%d | local_rank %d/%d | GPU %d | samples = %lld | "
        "blocks = %d | threads/block = %d | total threads = %lld | "
        "hits = %llu | time = %.6f s",
        myid, nproc, local_rank, local_size, device_id,
        nsamples_local, compute_blocks, compute_threads,
        total_threads, h_hits_local, tlocal);

    if (myid == 0) {
        printf("%s\n", diag); fflush(stdout);
        char buf[512];
        for (int src = 1; src < nproc; src++) {
            MPI_Recv(buf, sizeof(buf), MPI_CHAR,
                     src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("%s\n", buf); fflush(stdout);
        }
    } else {
        MPI_Send(diag, diag_len + 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    }

    // -----------------------------------------------------------------------
    // Final results — rank 0 only
    // -----------------------------------------------------------------------
    if (myid == 0) {
        const double pi_exact    = 3.14159265358979323846;
        double pi_estimate       = 4.0 * (double)h_hits_global
                                       / (double)nsamples_total;
        double abs_error         = fabs(pi_estimate - pi_exact);
        double rel_error_pct     = (abs_error / pi_exact) * 100.0;

        printf("\n========== FINAL RESULTS ==========\n");
        printf("Exact PI       = %.8f\n", pi_exact);
        printf("Estimated PI   = %.8f\n", pi_estimate);
        printf("Absolute error = %.8e\n", abs_error);
        printf("Relative error = %.8e %%\n", rel_error_pct);
        printf("Total hits     = %llu\n",   h_hits_global);
        printf("Total samples  = %lld\n",   nsamples_total);
        printf("Wall time      = %.6f s\n",  tmax);
        fflush(stdout);
    }

    // -----------------------------------------------------------------------
    // Cleanup
    // -----------------------------------------------------------------------
    cudaFree(d_counts);
    cudaFree(d_hits);
    cudaFree(d_global);
    cudaFree(d_tmp);
    ncclCommDestroy(nccl_comm);
    cudaStreamDestroy(stream);
    MPI_Comm_free(&local_comm);
    MPI_Finalize();
    return 0;
}
