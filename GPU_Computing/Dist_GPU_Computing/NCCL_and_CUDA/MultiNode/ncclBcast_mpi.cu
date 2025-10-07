// nvcc -O3 -std=c++17 -o ncclBcast_mpi.x ncclBcast_mpi.cu -lnccl -lmpi
#include <nccl.h>
#include <mpi.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK_CUDA(cmd) do {                                   \
  cudaError_t e = (cmd);                                       \
  if (e != cudaSuccess) {                                      \
    fprintf(stderr, "CUDA error %s:%d: %s\n",                  \
            __FILE__, __LINE__, cudaGetErrorString(e));        \
    MPI_Abort(MPI_COMM_WORLD, 1);                              \
  }                                                            \
} while(0)

#define CHECK_NCCL(cmd) do {                                   \
  ncclResult_t r = (cmd);                                      \
  if (r != ncclSuccess) {                                      \
    fprintf(stderr, "NCCL error %s:%d: %s\n",                  \
            __FILE__, __LINE__, ncclGetErrorString(r));        \
    MPI_Abort(MPI_COMM_WORLD, 1);                              \
  }                                                            \
} while(0)

__global__ void kernel(int *a, int n)
{
  int index = threadIdx.x;
  if (index < n) {
    a[index] *= 2;
    printf("%d\t", a[index]);
  }
}

static void print_vector_host(const int *in, int n) {
  for (int i = 0; i < n; ++i) printf("%d\t", in[i]);
  printf("\n");
}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  int world_rank = -1, world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Local rank per node (for selecting GPU)
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

  const int data_size = 8;

  // Host data: only rank 0 initializes (like your single-node version)
  std::vector<int> h_data(data_size, 0);
  if (world_rank == 0) {
    srand(42); // deterministic
    for (int i = 0; i < data_size; ++i) {
      h_data[i] = (rand() % 8) * 2; // even ints in [0,14]
    }
    // Print once (like your original)
    print_vector_host(h_data.data(), data_size);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // Device buffers
  int *d_data = nullptr;
  CHECK_CUDA(cudaMalloc(&d_data, data_size * sizeof(int)));

  if (world_rank == 0) {
    CHECK_CUDA(cudaMemcpyAsync(d_data, h_data.data(),
                               data_size * sizeof(int),
                               cudaMemcpyHostToDevice, stream));
  }

  // Broadcast from root=0 to all ranks
  CHECK_CUDA(cudaStreamSynchronize(stream));
  CHECK_NCCL(ncclBcast((void*)d_data, data_size, ncclInt, /*root=*/0, comm, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  MPI_Barrier(MPI_COMM_WORLD);

  // Print per-rank blocks in order to avoid interleaving
  for (int r = 0; r < world_size; ++r) {
    if (world_rank == r) {
      printf("\nThis is rank %d, device %d\n", world_rank, device);
      fflush(stdout);
      kernel<<<1, data_size, 0, stream>>>(d_data, data_size);
      CHECK_CUDA(cudaGetLastError());
      CHECK_CUDA(cudaDeviceSynchronize()); // flush device printf
      printf("\n");
      fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // Cleanup
  CHECK_CUDA(cudaFree(d_data));
  CHECK_CUDA(cudaStreamDestroy(stream));
  ncclCommDestroy(comm);
  MPI_Comm_free(&local_comm);
  MPI_Finalize();
  return 0;
}
