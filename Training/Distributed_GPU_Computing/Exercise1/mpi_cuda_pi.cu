#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>
#include <cuda_runtime.h>
#include <math.h>

#define NBIN_DEFAULT 1000000000LL

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
// Kernel 1: each thread accumulates a partial sum over a strided set of bins.
//
//   pi = integral_0^1  4 / (1 + x^2)  dx
//
// approximated with the midpoint rule:
//
//   pi ~ sum_i  [ 4 / (1 + x_i^2) ] * step,   x_i = (i + 0.5) * step
//
// 'start_bin' is the global bin offset for this MPI rank.
// ---------------------------------------------------------------------------
__global__ void compute_pi_partial(double *partial_sums,
                                   long long nbin_local,
                                   long long start_bin,
                                   double step)
{
    long long tid    = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x  * blockDim.x;

    double local_sum = 0.0;
    for (long long i = tid; i < nbin_local; i += stride) {
        double x = ((double)(start_bin + i) + 0.5) * step;
        local_sum += 4.0 / (1.0 + x * x);
    }
    partial_sums[tid] = local_sum;
}

// ---------------------------------------------------------------------------
// Kernel 2: parallel reduction using warp shuffles.
//
// Unlike the classic shared-memory tree reduction, this does NOT require
// blockDim.x to be a power of two, making it safe with any block size
// returned by cudaOccupancyMaxPotentialBlockSize.
//
// Each block reduces its chunk of 'input' to one value written to 'output'.
// ---------------------------------------------------------------------------
__global__ void reduce_sum(const double *input, double *output, long long n)
{
    double val = 0.0;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) val = input[idx];

    // Step 1: warp-level reduction via shuffle
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);

    // Step 2: one value per warp lands in shared memory
    __shared__ double sdata[32];   // max 32 warps per block
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    if (lane == 0) sdata[wid] = val;
    __syncthreads();

    // Step 3: first warp reduces the per-warp values
    if (wid == 0) {
        val = (lane < (blockDim.x >> 5)) ? sdata[lane] : 0.0;
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        if (lane == 0) output[blockIdx.x] = val;
    }
}

// ---------------------------------------------------------------------------
// Choose an occupancy-aware launch configuration for compute_pi_partial.
// ---------------------------------------------------------------------------
static void choose_compute_launch_config(long long nbin_local,
                                         int device_id,
                                         int *blocks,
                                         int *threads)
{
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    int min_grid_size = 0, block_size = 0;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
        &min_grid_size, &block_size, compute_pi_partial, 0, 0));
    if (block_size <= 0) block_size = 256;

    const long long target_iters_per_thread = 4096;
    long long wanted_threads =
        (nbin_local + target_iters_per_thread - 1) / target_iters_per_thread;
    if (wanted_threads < block_size) wanted_threads = block_size;

    long long block_count = (wanted_threads + block_size - 1) / block_size;

    long long max_useful_blocks = (long long)prop.multiProcessorCount * 32;
    if (block_count < min_grid_size)    block_count = min_grid_size;
    if (block_count > max_useful_blocks) block_count = max_useful_blocks;

    long long max_from_work = (nbin_local + block_size - 1) / block_size;
    if (max_from_work > 0 && block_count > max_from_work)
        block_count = max_from_work;
    if (block_count < 1) block_count = 1;

    *threads = block_size;
    *blocks  = (int)block_count;
}

// ---------------------------------------------------------------------------
// Iteratively reduce a device array down to a single scalar on the GPU.
// Only one double is ever copied back to the host.
// ---------------------------------------------------------------------------
static double gpu_reduce_to_scalar(double *d_input,
                                   long long n,
                                   int reduce_threads)
{
    double *d_in  = d_input;
    double *d_out = NULL;

    while (n > 1) {
        long long blocks = (n + reduce_threads - 1) / reduce_threads;

        CUDA_CHECK(cudaMalloc((void **)&d_out, (size_t)blocks * sizeof(double)));

        reduce_sum<<<(int)blocks, reduce_threads>>>(d_in, d_out, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        if (d_in != d_input) CUDA_CHECK(cudaFree(d_in));

        d_in  = d_out;
        d_out = NULL;
        n     = blocks;
    }

    double result = 0.0;
    CUDA_CHECK(cudaMemcpy(&result, d_in, sizeof(double), cudaMemcpyDeviceToHost));
    if (d_in != d_input) CUDA_CHECK(cudaFree(d_in));

    return result;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int myid, nproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    // Optional bin count override
    long long nbin_total = NBIN_DEFAULT;
    if (argc > 1) {
        nbin_total = atoll(argv[1]);
        if (nbin_total <= 0) {
            if (myid == 0)
                fprintf(stderr, "Invalid number of bins: %s\n", argv[1]);
            MPI_Finalize();
            return 1;
        }
    }

    // Build a shared-memory communicator to identify local rank → GPU mapping
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

    // Distribute bins across ranks; early ranks absorb the remainder
    long long base       = nbin_total / nproc;
    long long remainder  = nbin_total % nproc;
    long long nbin_local, start_bin;

    if (myid < remainder) {
        nbin_local = base + 1;
        start_bin  = (long long)myid * nbin_local;
    } else {
        nbin_local = base;
        start_bin  = remainder * (base + 1) + (long long)(myid - remainder) * base;
    }

    double step = 1.0 / (double)nbin_total;

    int compute_blocks, compute_threads;
    choose_compute_launch_config(nbin_local, device_id,
                                 &compute_blocks, &compute_threads);
    long long total_compute_threads = (long long)compute_blocks * compute_threads;

    double *d_partial = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_partial,
                          (size_t)total_compute_threads * sizeof(double)));

    // ------------------------------------------------------------------
    // Timed section: compute kernel + GPU reduction + MPI reduction
    // MPI_Barrier ensures all ranks start the clock at the same moment.
    // ------------------------------------------------------------------
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    compute_pi_partial<<<compute_blocks, compute_threads>>>(
        d_partial, nbin_local, start_bin, step);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    double raw_sum_local = gpu_reduce_to_scalar(d_partial,
                                                total_compute_threads, 256);
    double pi_local  = raw_sum_local * step;

    // MPI_Reduce: only rank 0 receives the global sum
    double pi_global = 0.0;
    MPI_Reduce(&pi_local, &pi_global, 1, MPI_DOUBLE, MPI_SUM, 0,
               MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end  = MPI_Wtime();
    double t_wall = t_end - t_start;

    // ------------------------------------------------------------------
    // Ordered per-rank diagnostics: rank 0 prints first, then receives
    // and prints each subsequent rank's line in order.
    // This avoids interleaved output from concurrent printf calls.
    // ------------------------------------------------------------------
    char diag[512];
    int  diag_len;

    diag_len = snprintf(diag, sizeof(diag),
        "rank %d/%d | local_rank %d/%d | GPU %d | bins = %lld | "
        "blocks = %d | threads/block = %d | total threads = %lld | "
        "partial pi = %.15f",
        myid, nproc, local_rank, local_size, device_id,
        nbin_local, compute_blocks, compute_threads,
        total_compute_threads, pi_local);

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
        const double pi_exact    = 3.14159265358979323846;
        double abs_error         = fabs(pi_global - pi_exact);
        double rel_error_pct     = (abs_error / pi_exact) * 100.0;

        printf("\n========== FINAL RESULTS ==========\n");
        printf("Exact PI       = %.8f\n", pi_exact);
        printf("Computed PI    = %.8f\n", pi_global);
        printf("Absolute error = %.8e\n", abs_error);
        printf("Relative error = %.8e %%\n", rel_error_pct);
        printf("Total bins     = %lld\n",   nbin_total);
        printf("Wall time      = %.6f s\n",  t_wall);
        fflush(stdout);
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_partial));
    MPI_Comm_free(&local_comm);
    MPI_Finalize();
    return 0;
}
