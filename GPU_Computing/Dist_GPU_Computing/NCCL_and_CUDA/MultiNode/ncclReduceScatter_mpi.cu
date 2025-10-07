/*
 * =============================================================================
 *  NCCL + CUDA + MPI ReduceScatter (Multi-Node, One GPU per Rank)
 * =============================================================================
 *
 *  Overview
 *  --------
 *  Demonstrates bootstrapping a multi-node NCCL communicator with MPI and
 *  performing an `ncclReduceScatter` (sum) across ranks where each MPI rank
 *  controls exactly one GPU (selected via per-node local rank). Every rank
 *  prepares a length-`world_size` vector; the sum across ranks is computed and
 *  exactly one element (recvcount = 1) is scattered back to each rank/GPU.
 *
 *  Initialization & Math
 *  ---------------------
 *  On rank r (0-based), the host vector is:
 *      hsend[k] = 10*(r + 1) + 40*k,   for k = 0..world_size-1
 *
 *  After `ReduceScatter` with SUM, rank r receives the r-th element of the
 *  globally reduced vector:
 *      recv[r] = Σ_{q=0..W-1} ( 10*(q+1) + 40*r )
 *               = 10 * W*(W+1)/2 + 40 * W * r
 *      where W = world_size
 *
 *  Example (W = 8): recv[r] = 360 + 320*r → 360, 680, 1000, …, 2600.
 *
 *  What this shows
 *  ---------------
 *  - MPI used only for:
 *      • process/rank discovery and per-node local rank
 *      • broadcasting a single `ncclUniqueId` to all ranks
 *  - NCCL used for the collective (`ncclReduceScatter`) across nodes/GPUs
 *  - One-process-per-GPU pattern with deterministic, ordered printing
 *
 *  Output flow
 *  -----------
 *    Host (all ranks) → Device (all ranks)
 *      └─ ncclReduceScatter(sum, recvcount=1) → each GPU gets one float
 *            └─ CUDA kernel prints the received element from the device
 *
 *  Requirements
 *  ------------
 *  - CUDA Toolkit
 *  - NCCL library
 *  - MPI implementation (for launch + bootstrap)
 *  - A cluster with ≥ 1 GPU per MPI rank (e.g., 2 nodes × 4 GPUs = 8 ranks)
 *
 *  Build
 *  -----
 *  nvcc -O3 -std=c++17 -o ncclReduceScatter_mpi.x ncclReduceScatter_mpi.cu -lnccl -lmpi
 *
 *  Expected output (shape)
 *  -----------------------
 *  - Each rank first prints its host send vector (in rank order).
 *  - Then, for ranks 0..W-1 in order:
 *        This is rank R, device D
 *            <single float printed from the device>   // equals 10*W*(W+1)/2 + 40*W*R
 *
 *  Notes
 *  -----
 *  - Device selection: `device = local_rank % cudaDeviceCount()`.
 *  - Ordered printing is gated by `MPI_Barrier` to avoid interleaving.
 *  - `recvcount = 1` for clarity; increase to receive a block per rank.
 *  - Error macros abort the whole job on first CUDA/NCCL error for simplicity.
 * =============================================================================
 */

#include <cstdio>
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

#define CHECK_NCCL(cmd) do {                                  \
  ncclResult_t r = (cmd);                                     \
  if (r != ncclSuccess) {                                     \
    fprintf(stderr, "NCCL error %s:%d: %s\n",                 \
            __FILE__, __LINE__, ncclGetErrorString(r));       \
    MPI_Abort(MPI_COMM_WORLD, 1);                             \
  }                                                           \
} while(0)

__global__ void Dev_print(const float* x, int n) {
  int i = threadIdx.x;
  if (i < n) printf("%1.2f\t", x[i]);
}

static void print_vector_host(const float* a, int n) {
  for (int i = 0; i < n; ++i) printf("%1.2f\t", a[i]);
  printf("\n");
}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  int world_rank = -1, world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // One GPU per rank: pick device by local rank
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

  // ReduceScatter parameters
  const int size = world_size;   // send vector length per rank
  const int recvcount = 1;       // each rank receives 1 element

  // Host send vector: hsend[k] = 10*(rank+1) + 40*k
  std::vector<float> hsend(size);
  for (int k = 0; k < size; ++k) hsend[k] = 10.0f * (world_rank + 1) + 40.0f * k;

  // Print each rank's host vector (like original example), in order
  for (int r = 0; r < world_size; ++r) {
    if (world_rank == r) {
      print_vector_host(hsend.data(), size);
      fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // Device buffers
  float *sendbuff_d = nullptr, *recvbuff_d = nullptr;
  CHECK_CUDA(cudaMalloc(&sendbuff_d, size * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&recvbuff_d, recvcount * sizeof(float)));

  CHECK_CUDA(cudaMemcpyAsync(sendbuff_d, hsend.data(),
                             size * sizeof(float), cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  // ReduceScatter (sum) across ranks; each rank receives its r-th element of the summed vector
  CHECK_NCCL(ncclReduceScatter((const void*)sendbuff_d,
                               (void*)recvbuff_d,
                               recvcount, ncclFloat, ncclSum,
                               comm, stream));

  CHECK_CUDA(cudaStreamSynchronize(stream));
  MPI_Barrier(MPI_COMM_WORLD);

  // Print per-rank result in order (non-interleaved)
  for (int r = 0; r < world_size; ++r) {
    if (world_rank == r) {
      printf("\nThis is rank %d, device %d\n", world_rank, device);
      fflush(stdout);
      Dev_print<<<1, recvcount, 0, stream>>>(recvbuff_d, recvcount);
      CHECK_CUDA(cudaGetLastError());
      CHECK_CUDA(cudaDeviceSynchronize());
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
