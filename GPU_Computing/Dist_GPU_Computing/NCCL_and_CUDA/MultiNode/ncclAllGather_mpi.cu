/*
 * =============================================================================
 *  NCCL + CUDA + MPI AllGather (Multi-Node, One GPU per Rank)
 * =============================================================================
 *
 *  Overview
 *  --------
 *  Demonstrates how to bootstrap NCCL collectives with MPI across multiple
 *  nodes (e.g., 2 nodes × 4 GPUs = 8 ranks). Each MPI rank controls exactly
 *  one GPU (selected via local rank), contributes a single float equal to
 *  10*(rank+1), and participates in an `ncclAllGather` so that every rank
 *  receives the full vector:
 *
 *      [10.0, 20.0, 30.0, ..., 10.0 * world_size]
 *
 *  After the collective, each rank prints the gathered vector from its GPU.
 *  The output is printed in rank order to avoid interleaving.
 *
 *  What this shows
 *  ---------------
 *  - Using MPI only for bootstrap:
 *      * Discover world size / ranks and local rank per node
 *      * Broadcast a single `ncclUniqueId` to all ranks
 *  - Using NCCL for the actual inter/intra-node communication (AllGather)
 *  - One process per GPU pattern for multi-GPU, multi-node jobs
 *
 *  Requirements
 *  ------------
 *  - CUDA Toolkit
 *  - NCCL library
 *  - MPI implementation with PMIx/SLURM or mpirun support
 *  - A cluster with ≥ 1 GPU per MPI rank (example assumes 8 GPUs total)
 *
 *  Build
 *  -----
 *  nvcc -O3 -std=c++17 -o ncclAllGather_mpi.x ncclAllGather_mpi.cu -lnccl -lmpi
 *
 *  Expected Output (shape)
 *  -----------------------
 *  Rank 0 prints the initial seed line once:
 *      10.00 20.00 30.00 ... (up to 10*world_size)
 *  Then each rank, in order, prints:
 *      This is rank R, device D
 *          10.00 20.00 30.00 ... (same gathered vector)
 *
 *  Notes
 *  -----
 *  - Device selection uses the per-node local rank (via MPI split-type),
 *    mapping local_rank % num_devices.
 *  - The code is generic: it works for any world_size ≥ 1, not just 8.
 *  - MPI is used only for process management and the `ncclUniqueId` broadcast;
 *    NCCL handles all device-to-device communication across nodes.
 * =============================================================================
 */

#include <stdio.h>
#include <vector>
#include <mpi.h>
#include <cuda_runtime.h>
#include <nccl.h>

#define CHECK_CUDA(cmd) do {                                  \
  cudaError_t e = (cmd);                                      \
  if (e != cudaSuccess) {                                     \
    fprintf(stderr, "CUDA error %s:%d: %s\n",                 \
            __FILE__, __LINE__, cudaGetErrorString(e));       \
    MPI_Abort(MPI_COMM_WORLD, 1);                             \
  }                                                           \
} while(0)

#define CHECK_NCCL(cmd) do {                                   \
  ncclResult_t r = (cmd);                                      \
  if (r != ncclSuccess) {                                      \
    fprintf(stderr, "NCCL error %s:%d: %s\n",                  \
            __FILE__, __LINE__, ncclGetErrorString(r));        \
    MPI_Abort(MPI_COMM_WORLD, 1);                              \
  }                                                            \
} while(0)

__global__ void Dev_print(const float* x, int n) {
  int i = threadIdx.x;
  if (i < n) printf("%1.2f\t", x[i]);
}

static void print_line_host(const float* a, int n) {
  for (int i = 0; i < n; ++i) printf("%1.2f\t", a[i]);
  printf("\n");
}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  int world_rank = -1, world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Expect 8 ranks (2 nodes × 4 GPUs), but code works for any world_size >= 1
  const int size = world_size;    // gathered length
  const int sendcount = 1;        // each rank contributes one float

  // Determine local rank for device selection
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
  if (world_rank == 0) CHECK_NCCL(ncclGetUniqueId(&id));
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

  ncclComm_t comm;
  CHECK_NCCL(ncclCommInitRank(&comm, world_size, id, world_rank));

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  // Host "send" buffer: only position [rank] is nonzero = 10*(rank+1)
  std::vector<float> hsend(size, 0.0f);
  hsend[world_rank] = 10.0f * (world_rank + 1);

  // Device buffers
  float *sendbuff_d = nullptr, *recvbuff_d = nullptr;
  CHECK_CUDA(cudaMalloc(&sendbuff_d, size * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&recvbuff_d, size * sizeof(float)));

  CHECK_CUDA(cudaMemcpyAsync(sendbuff_d, hsend.data(),
                             size * sizeof(float), cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  // Print initial seed values once (like your host_x* prints)
  if (world_rank == 0) {
    std::vector<float> init(size);
    for (int i = 0; i < size; ++i) init[i] = 10.0f * (i + 1);
    print_line_host(init.data(), size);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // AllGather: each rank sends one float (the [rank]-th element)
  CHECK_NCCL(ncclAllGather(/*send*/ sendbuff_d + world_rank,
                           /*recv*/ recvbuff_d,
                           sendcount, ncclFloat, comm, stream));

  CHECK_CUDA(cudaStreamSynchronize(stream));
  MPI_Barrier(MPI_COMM_WORLD);

  // Print per-rank blocks in order to avoid interleaving
  for (int r = 0; r < world_size; ++r) {
    if (world_rank == r) {
      printf("\nThis is rank %d, device %d\n", world_rank, device);
      fflush(stdout);
      Dev_print<<<1, size, 0, stream>>>(recvbuff_d, size);
      CHECK_CUDA(cudaGetLastError());
      CHECK_CUDA(cudaDeviceSynchronize()); // ensure device printf flushes before next rank
      printf("\n");
      fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // Cleanup
  CHECK_CUDA(cudaFree(sendbuff_d));
  CHECK_CUDA(cudaFree(recvbuff_d));
  CHECK_CUDA(cudaStreamDestroy(stream));
  ncclCommDestroy(comm);
  MPI_Comm_free(&local_comm);
  MPI_Finalize();
  return 0;
}

