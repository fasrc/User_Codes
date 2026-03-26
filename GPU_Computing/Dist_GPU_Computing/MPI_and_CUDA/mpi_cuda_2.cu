#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>
#include <cuda_runtime.h>

//
// Total number of integration bins across all MPI ranks.
// You can also override this from the command line.
//
#define NBIN_DEFAULT 1000000000LL

//
// Simple CUDA error-checking macro.
// If any CUDA call fails, abort the whole MPI job.
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
// Kernel 1:
// Each thread computes a partial sum of the integral over a strided set of bins.
//
// The integral is:
//
//   pi = integral_0^1 4 / (1 + x^2) dx
//
// which we approximate with the midpoint rule:
//
//   pi ~ sum_i [ 4 / (1 + x_i^2) ] * step
//
// where x_i = (i + 0.5) * step
//
// Each MPI rank is assigned a subset of global bins.
// 'start_bin' tells this rank where its portion begins in the global bin index space.
//
// Output:
//   partial_sums[tid] = this thread's accumulated sum (without multiplying by step yet)
//
__global__ void compute_pi_partial(double *partial_sums,
                                   long long nbin_local,
                                   long long start_bin,
                                   double step)
{
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    double local_sum = 0.0;

    //
    // Strided loop:
    // thread tid handles bins:
    //   tid, tid + stride, tid + 2*stride, ...
    //
    for (long long i = tid; i < nbin_local; i += stride) {
        long long global_bin = start_bin + i;
        double x = ((double)global_bin + 0.5) * step;
        local_sum += 4.0 / (1.0 + x * x);
    }

    partial_sums[tid] = local_sum;
}

//
// Kernel 2:
// Standard parallel reduction kernel.
// Reduces 'n' input values down to one partial sum per block.
//
// input  -> array of values to reduce
// output -> one reduced value per block
//
// This uses shared memory for speed.
//
__global__ void reduce_sum(const double *input, double *output, long long n)
{
    extern __shared__ double sdata[];

    unsigned int tid = threadIdx.x;
    long long global_idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;

    //
    // Load one element per thread into shared memory.
    // If out of bounds, contribute 0.
    //
    double x = 0.0;
    if (global_idx < n) {
        x = input[global_idx];
    }
    sdata[tid] = x;
    __syncthreads();

    //
    // Tree reduction in shared memory:
    // after each step, half as many threads remain active.
    //
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    //
    // Thread 0 writes this block's result.
    //
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

//
// Choose a dynamic launch configuration for the main compute kernel.
//
// Goals:
// - choose a block size that CUDA says is reasonable for this kernel
// - choose enough blocks to occupy the GPU
// - avoid launching far more threads than useful for the amount of work
//
static void choose_compute_launch_config(long long nbin_local,
                                         int device_id,
                                         int *blocks,
                                         int *threads)
{
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    int min_grid_size = 0;
    int block_size = 0;

    //
    // Ask CUDA for an occupancy-friendly block size for this kernel.
    //
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
        &min_grid_size,
        &block_size,
        compute_pi_partial,
        0,
        0));

    if (block_size <= 0) {
        block_size = 256;
    }

    //
    // Heuristic:
    // target a few thousand iterations per thread on average.
    // If there are more local bins, we use more total threads.
    //
    const long long target_iters_per_thread = 4096;
    long long wanted_threads =
        (nbin_local + target_iters_per_thread - 1) / target_iters_per_thread;

    //
    // At least one full block.
    //
    if (wanted_threads < block_size) {
        wanted_threads = block_size;
    }

    long long block_count = (wanted_threads + block_size - 1) / block_size;

    //
    // Also keep enough blocks to occupy the GPU reasonably well.
    //
    int sm_count = prop.multiProcessorCount;
    long long max_useful_blocks = (long long)sm_count * 32;

    if (block_count < min_grid_size) {
        block_count = min_grid_size;
    }
    if (block_count > max_useful_blocks) {
        block_count = max_useful_blocks;
    }

    //
    // Do not launch more blocks than there is work for.
    //
    long long max_blocks_from_work = (nbin_local + block_size - 1) / block_size;
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
// Repeatedly reduce a device array down to a single value on the GPU.
// Returns the result copied back to the host.
//
// This avoids copying a large array of per-thread partial sums back to the CPU.
// Only one final double is copied back.
//
static double gpu_reduce_to_scalar(double *d_input,
                                   long long n,
                                   int reduce_threads)
{
    double *d_in = d_input;
    double *d_out = NULL;
    double result = 0.0;

    //
    // Repeatedly launch reduction kernels until only one value remains.
    //
    while (n > 1) {
        long long blocks = (n + reduce_threads - 1) / reduce_threads;
        size_t out_bytes = (size_t)blocks * sizeof(double);

        CUDA_CHECK(cudaMalloc((void **)&d_out, out_bytes));

        reduce_sum<<<(int)blocks, reduce_threads,
                     reduce_threads * sizeof(double)>>>(d_in, d_out, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        //
        // Free the previous input buffer if it was an intermediate buffer
        // allocated inside this routine.
        //
        if (d_in != d_input) {
            CUDA_CHECK(cudaFree(d_in));
        }

        d_in = d_out;
        d_out = NULL;
        n = blocks;
    }

    //
    // Copy the final scalar back to the host.
    //
    CUDA_CHECK(cudaMemcpy(&result, d_in, sizeof(double), cudaMemcpyDeviceToHost));

    //
    // If the final buffer is not the original input, free it.
    //
    if (d_in != d_input) {
        CUDA_CHECK(cudaFree(d_in));
    }

    return result;
}

int main(int argc, char **argv)
{
    int myid, nproc;

    //
    // Variables for splitting work across MPI ranks.
    //
    long long nbin_total = NBIN_DEFAULT;
    long long nbin_local, remainder, start_bin;

    //
    // Integration step size and results.
    //
    double step;
    double pi_local = 0.0;
    double pi_global = 0.0;

    //
    // GPU / launch configuration info.
    //
    int num_devices = 0;
    int local_rank = 0;
    int local_size = 0;
    int device_id = 0;
    int compute_blocks = 0;
    int compute_threads = 0;
    long long total_compute_threads = 0;

    //
    // Device buffer for per-thread partial sums.
    //
    double *d_partial = NULL;

    //
    // Initialize MPI.
    //
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    //
    // Optional command-line override for total number of bins.
    // Example:
    //   mpirun -np 4 ./mpi_pi_optimized 2000000000
    //
    if (argc > 1) {
        nbin_total = atoll(argv[1]);
        if (nbin_total <= 0) {
            if (myid == 0) {
                fprintf(stderr, "Invalid number of bins: %s\n", argv[1]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    //
    // Create a communicator containing ranks on the same shared-memory node.
    // This helps us identify a rank's local position on a node.
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
    // Determine how many CUDA devices exist on this node.
    //
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    if (num_devices <= 0) {
        fprintf(stderr, "Rank %d: no CUDA devices found on this node\n", myid);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    //
    // Map one local MPI rank to one GPU by local rank.
    //
    // If local_size <= num_devices, each local rank gets a unique GPU.
    // If local_size > num_devices, ranks will wrap around and share GPUs.
    //
    device_id = local_rank % num_devices;
    CUDA_CHECK(cudaSetDevice(device_id));

    //
    // Split bins across MPI ranks, including remainder bins.
    //
    // Example:
    //   nbin_total = 10, nproc = 3
    //
    // ranks 0..(remainder-1) get one extra bin.
    //
    nbin_local = nbin_total / nproc;
    remainder = nbin_total % nproc;

    if (myid < remainder) {
        nbin_local += 1;
        start_bin = myid * nbin_local;
    } else {
        start_bin = remainder * (nbin_total / nproc + 1)
                  + (myid - remainder) * (nbin_total / nproc);
    }

    //
    // Global step size over the full [0,1] interval.
    //
    step = 1.0 / (double)nbin_total;

    //
    // Dynamically choose CUDA launch geometry for the compute kernel.
    //
    choose_compute_launch_config(nbin_local, device_id,
                                 &compute_blocks, &compute_threads);

    total_compute_threads = (long long)compute_blocks * compute_threads;

    //
    // Allocate one partial-sum slot per compute thread.
    //
    CUDA_CHECK(cudaMalloc((void **)&d_partial,
                          (size_t)total_compute_threads * sizeof(double)));

    //
    // Launch kernel 1:
    // each thread computes a partial sum over its strided subset of bins.
    //
    compute_pi_partial<<<compute_blocks, compute_threads>>>(
        d_partial,
        nbin_local,
        start_bin,
        step
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    //
    // Reduce all thread partial sums down to one scalar on the GPU.
    //
    // We use 256 threads/block for reduction, which is a common good choice.
    //
    double raw_sum_local = gpu_reduce_to_scalar(d_partial,
                                                total_compute_threads,
                                                256);

    //
    // Multiply by step to get this MPI rank's contribution to pi.
    //
    pi_local = raw_sum_local * step;

    //
    // Sum all MPI ranks' partial pi values into pi_global.
    //
    MPI_Allreduce(&pi_local, &pi_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    //
    // Print per-rank diagnostics.
    //
    printf("rank %d/%d | local_rank %d/%d | GPU %d | local bins = %lld | "
           "compute blocks = %d | threads/block = %d | total threads = %lld | "
           "partial pi = %.15f\n",
           myid, nproc,
           local_rank, local_size,
           device_id,
           nbin_local,
           compute_blocks,
           compute_threads,
           total_compute_threads,
           pi_local);

    //
    // Print final result once.
    //
    if (myid == 0) {
        printf("Final PI = %.15f\n", pi_global);
    }

    //
    // Cleanup.
    //
    CUDA_CHECK(cudaFree(d_partial));
    MPI_Comm_free(&local_comm);
    MPI_Finalize();

    return 0;
}
