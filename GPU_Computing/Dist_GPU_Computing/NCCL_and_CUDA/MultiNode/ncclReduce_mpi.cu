/*
 * =============================================================================
 *  NCCL + CUDA + MPI Reduce (Multi-Node, One GPU per Rank)
 * =============================================================================
 *
 *  Overview
 *  --------
 *  Demonstrates bootstrapping a multi-node NCCL communicator with MPI and
 *  performing a pair of `ncclReduce` operations (sum to root). Each MPI rank
 *  controls exactly one GPU (selected via its per-node local rank). Every rank
 *  initializes two length-8 vectors on its GPU:
 *
 *      x = [1, 1, 1, 1, 1, 1, 1, 1]
 *      y = [2, 2, 2, 2, 2, 2, 2, 2]
 *
 *  NCCL reduces (sums) x and y across all ranks to rank 0. A simple CUDA kernel
 *  on rank 0 then computes a block-wise reduction of the received vector and
 *  prints a “dot” value from the device (for 8 ranks, this prints 128.00).
 *  Non-root ranks print a matching “0.00” line to keep the output aligned with
 *  single-node examples.
 *
 *  What this shows
 *  ---------------
 *  - Using MPI *only* for:
 *      • process/rank discovery
 *      • computing per-node local rank to select the CUDA device
 *      • broadcasting a single `ncclUniqueId` to all ranks
 *  - Using NCCL for the inter/intra-node collectives (two `ncclReduce` calls)
 *  - One-process-per-GPU pattern that scales across nodes
 *  - Ordered, non-interleaved printing by iterating ranks 0..N-1
 *
 *  Data flow
 *  ---------
 *    Host (all ranks) -> Device (all ranks)
 *      └─ ncclReduce(x, root=0) and ncclReduce(y, root=0)
 *            └─ Rank 0 launches CUDA kernel to reduce and print a value
 *               Other ranks print a placeholder line
 *
 *  Requirements
 *  ------------
 *  - CUDA Toolkit
 *  - NCCL library
 *  - MPI implementation (for launch + bootstrap)
 *  - A cluster with ≥ 1 GPU per MPI rank (example assumes 2 nodes × 4 GPUs)
 *
 *  Build
 *  -----
 *  nvcc -O3 -std=c++17 -o ncclReduce_mpi.x ncclReduce_mpi.cu -lnccl -lmpi
 *
 *  Expected output (shape)
 *  -----------------------
 *  Rank 0 prints the initial host vectors once (x then y).
 *  Then, for ranks 0..N-1 in order:
 *      This is rank R, device D
 *          dot(x,y) = 128.00     (on rank 0 for N=8; others print 0.00)
 *
 *  Notes
 *  -----
 *  - Only the root’s Sx_d/Sy_d contain valid reduced data after `ncclReduce`.
 *  - Device selection: `device = local_rank % cudaDeviceCount()`.
 *  - The kernel uses a single block with `blockDim = data_size` (8) and shared
 *    memory to do a simple sum reduction; adjust if you change `data_size`.
 * =============================================================================
 */

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <nccl.h>
#include <mpi.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(cmd) do {                                  \
  cudaError_t e = cmd;                                        \
  if (e != cudaSuccess) {                                     \
    fprintf(stderr, "CUDA error %s:%d: %s\n",                 \
            __FILE__, __LINE__, cudaGetErrorString(e));       \
    MPI_Abort(MPI_COMM_WORLD, 1);                             \
  }                                                           \
} while(0)

#define CHECK_NCCL(cmd) do {                                  \
  ncclResult_t r = cmd;                                       \
  if (r != ncclSuccess) {                                     \
    fprintf(stderr, "NCCL error %s:%d: %s\n",                 \
            __FILE__, __LINE__, ncclGetErrorString(r));       \
    MPI_Abort(MPI_COMM_WORLD, 1);                             \
  }                                                           \
} while(0)

__global__ void Dev_dot(double *x, double *y, int n) {
   __shared__ double tmp[512];
   int i = threadIdx.x;
   int t = blockDim.x * blockIdx.x + threadIdx.x;

   if (t < n) tmp[i] = x[t]; else tmp[i] = 0.0;
   __syncthreads();

   for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (i < stride) tmp[i] += tmp[i + stride];
      __syncthreads();
   }

   if (threadIdx.x == 0) {
      y[blockIdx.x] = tmp[0];
      printf("\tdot(x,y) = %1.2f\n", y[blockIdx.x]);
   }
}

static void print_vector_host(const double* in, int n) {
  for (int i = 0; i < n; ++i) printf("%1.2f\t", in[i]);
  printf("\n");
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int world_rank = -1, world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Determine local rank per node (for device selection)
  MPI_Comm local_comm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
  int local_rank = -1;
  MPI_Comm_rank(local_comm, &local_rank);

  int ndev = 0;
  CHECK_CUDA(cudaGetDeviceCount(&ndev));
  if (ndev < 1) {
    if (world_rank == 0) fprintf(stderr, "No CUDA devices found.\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  int device = local_rank % ndev;
  CHECK_CUDA(cudaSetDevice(device));

  // Bootstrap NCCL with MPI
  ncclUniqueId id;
  if (world_rank == 0) {
    CHECK_NCCL(ncclGetUniqueId(&id));
  }
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

  ncclComm_t comm;
  CHECK_NCCL(ncclCommInitRank(&comm, world_size, id, world_rank));

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  const int data_size = 8; // same as your original
  std::vector<double> hx(data_size, 1.0), hy(data_size, 2.0);

  double *x_d = nullptr, *y_d = nullptr, *Sx_d = nullptr, *Sy_d = nullptr;
  CHECK_CUDA(cudaMalloc(&x_d,  data_size * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&y_d,  data_size * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&Sx_d, data_size * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&Sy_d, data_size * sizeof(double)));

  CHECK_CUDA(cudaMemcpyAsync(x_d, hx.data(), data_size * sizeof(double), cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaMemcpyAsync(y_d, hy.data(), data_size * sizeof(double), cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  // Only once: show the initial vectors (like your original top-of-output)
  if (world_rank == 0) {
    print_vector_host(hx.data(), data_size);
    print_vector_host(hy.data(), data_size);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // Global reduce to root=0
  CHECK_NCCL(ncclGroupStart());
  CHECK_NCCL(ncclReduce((const void*)x_d, (void*)Sx_d,
                        data_size, ncclDouble, ncclSum, 0, comm, stream));
  CHECK_NCCL(ncclReduce((const void*)y_d, (void*)Sy_d,
                        data_size, ncclDouble, ncclSum, 0, comm, stream));
  CHECK_NCCL(ncclGroupEnd());

  CHECK_CUDA(cudaStreamSynchronize(stream));
  MPI_Barrier(MPI_COMM_WORLD);

  // Print blocks in rank order so the output matches your sample style
  for (int r = 0; r < world_size; ++r) {
    if (world_rank == r) {
      // Leading blank line to match your formatting
      printf("\n This is rank %d, device %d\n", world_rank, device);
      fflush(stdout);

      if (world_rank == 0) {
        // Launch kernel only on rank 0; its printf comes from the device
        Dev_dot<<<1, data_size, 0, stream>>>(Sy_d, Sx_d, data_size);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaStreamSynchronize(stream));
      } else {
        // For non-root ranks, mimic your "0.00" line from host
        printf("\tdot(x,y) = 0.00\n");
        fflush(stdout);
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // Cleanup
  CHECK_CUDA(cudaFree(x_d));
  CHECK_CUDA(cudaFree(y_d));
  CHECK_CUDA(cudaFree(Sx_d));
  CHECK_CUDA(cudaFree(Sy_d));
  CHECK_CUDA(cudaStreamDestroy(stream));
  ncclCommDestroy(comm);
  MPI_Comm_free(&local_comm);
  MPI_Finalize();
  return 0;
}

