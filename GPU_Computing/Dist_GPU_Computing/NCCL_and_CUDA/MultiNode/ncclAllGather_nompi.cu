/*
 * =============================================================================
 *  NCCL + CUDA AllGather (No MPI) with Shared-File Bootstrap
 * =============================================================================
 *
 *  Overview
 *  --------
 *  Demonstrates a fully MPI-free, multi-node NCCL program using CUDA where each
 *  process controls exactly one GPU (selected via per-node local rank env vars).
 *  Rank 0 creates a `ncclUniqueId` and writes it to a small file on a shared
 *  filesystem; other ranks read that file to join the same NCCL communicator.
 *
 *  Each rank contributes a single float equal to 10*(rank+1). An `ncclAllGather`
 *  is performed so every rank receives the full vector:
 *
 *      [10.0, 20.0, 30.0, ..., 10.0 * world_size]
 *
 *  The gathered vector is printed from the device on each rank, in order, using
 *  a simple NCCL-based barrier to avoid interleaving.
 *
 *  What this shows
 *  ---------------
 *  - Launching multi-node jobs **without MPI** (e.g., via Slurm `srun`)
 *  - Minimal out-of-band bootstrap using a **shared file** for `ncclUniqueId`
 *  - One-process-per-GPU mapping via `LOCAL_RANK`/`SLURM_LOCALID`
 *  - Using **NCCL only** for both intra- and inter-node collectives (AllGather)
 *  - Ordered, readable output using a tiny NCCL AllReduce as a barrier
 *
 *  Requirements
 *  ------------
 *  - CUDA Toolkit
 *  - NCCL library
 *  - A shared filesystem visible to all ranks (for the uid file)
 *  - A launcher that sets rank env vars (e.g., Slurm `srun`)
 *
 *  Build
 *  -----
 *  nvcc -O3 -std=c++17 -o ncclAllGather_nompi.x ncclAllGather_nompi.cu -lnccl
 *
 *  Bootstrap details
 *  -----------------
 *  - Rank/size are taken from environment (e.g., SLURM_NTASKS/PROCID/LOCALID).
 *  - Rank 0 calls `ncclGetUniqueId()` and writes it to:
 *        $SLURM_SUBMIT_DIR/nccl_uid.$SLURM_JOB_ID     (fallback: $HOME/nccl_uid.default)
 *  - Other ranks spin-wait up to ~30s for that file, then read the id and join.
 *
 *  Expected output (shape)
 *  -----------------------
 *  Rank 0 prints the seed once:
 *      10.00 20.00 30.00 ... (up to 10*world_size)
 *  Then, for ranks 0..world_size-1 in order:
 *      This is rank R, device D
 *          10.00 20.00 30.00 ... (same gathered vector)
 *
 *  Notes
 *  -----
 *  - Device selection: `device = local_rank % cudaDeviceCount()`.
 *  - The simple NCCL barrier uses `ncclAllReduce` on a single int and a stream
 *    sync to serialize prints without MPI.
 *  - Ensure the uid path is on a **shared** and **writable** filesystem.
 *  - If your site lacks a shared FS or you prefer not to write files, you can
 *    replace the bootstrap with a small TCP rendezvous or PMIx publish/lookup.
 * =============================================================================
 */

#include <stdio.h>
#include <vector>
#include <string>
#include <unistd.h>
#include <sys/stat.h>
#include <cuda_runtime.h>
#include <nccl.h>

#define CHECK_CUDA(cmd) do {                                   \
  cudaError_t e = (cmd);                                       \
  if (e != cudaSuccess) {                                      \
    fprintf(stderr, "CUDA error %s:%d: %s\n",                  \
            __FILE__, __LINE__, cudaGetErrorString(e));        \
    exit(1);                                                   \
  }                                                            \
} while(0)

#define CHECK_NCCL(cmd) do {                                   \
  ncclResult_t r = (cmd);                                      \
  if (r != ncclSuccess) {                                      \
    fprintf(stderr, "NCCL error %s:%d: %s\n",                  \
            __FILE__, __LINE__, ncclGetErrorString(r));        \
    exit(1);                                                   \
  }                                                            \
} while(0)

static int get_env_int(const char* k, int defv) {
  const char* v = getenv(k);
  return v ? atoi(v) : defv;
}

static void print_line_host(const float* a, int n) {
  for (int i = 0; i < n; ++i) printf("%1.2f\t", a[i]);
  printf("\n");
}

// --- Minimal shared-file OOB for ncclUniqueId ---
static void write_unique_id(const char* path, const ncclUniqueId& id){
  FILE* f = fopen(path, "wb");
  if (!f) { perror("fopen write_unique_id"); exit(1); }
  size_t n = fwrite(&id, 1, sizeof(id), f);
  fclose(f);
  if (n != sizeof(id)) { fprintf(stderr, "short write of unique id\n"); exit(1); }
}

static void read_unique_id_blocking(const char* path, ncclUniqueId& id){
  for (int i=0; i<300; ++i) { // wait up to ~30s
    struct stat st;
    if (stat(path, &st) == 0 && st.st_size == (off_t)sizeof(id)) {
      FILE* f = fopen(path, "rb");
      if (!f) { perror("fopen read_unique_id"); exit(1); }
      size_t n = fread(&id, 1, sizeof(id), f);
      fclose(f);
      if (n != sizeof(id)) { fprintf(stderr, "short read of unique id\n"); exit(1); }
      return;
    }
    usleep(100000); // 100 ms
  }
  fprintf(stderr, "Timed out waiting for unique id file: %s\n", path);
  exit(1);
}

// Simple NCCL-based barrier: allreduce a single int and sync stream
static void nccl_barrier(ncclComm_t comm, cudaStream_t stream, int* d_token){
  CHECK_NCCL(ncclAllReduce(d_token, d_token, 1, ncclInt, ncclSum, comm, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
}

__global__ void Dev_print(const float* x, int n) {
  int i = threadIdx.x;
  if (i < n) printf("%1.2f\t", x[i]);
}

int main() {
  // --- Rank/size from Slurm (fallback to torchrun-style envs) ---
  int world_size = get_env_int("SLURM_NTASKS", get_env_int("WORLD_SIZE", 1));
  int world_rank = get_env_int("SLURM_PROCID", get_env_int("RANK", 0));
  int local_rank = get_env_int("SLURM_LOCALID", get_env_int("LOCAL_RANK", 0));

  if (world_size < 1) { fprintf(stderr, "Invalid world_size\n"); return 1; }

  // --- Select device by local rank ---
  int ndev = 0;
  CHECK_CUDA(cudaGetDeviceCount(&ndev));
  if (ndev < 1) { fprintf(stderr, "No CUDA devices found.\n"); return 1; }
  int device = local_rank % ndev;
  CHECK_CUDA(cudaSetDevice(device));

  // --- Bootstrap NCCL unique id via shared file ---
  const char* jobid = getenv("SLURM_JOB_ID");
  const char* dir   = getenv("SLURM_SUBMIT_DIR");
  if (!dir) dir = getenv("HOME");
  std::string uid_path = std::string(dir ? dir : ".") + "/nccl_uid." + (jobid ? jobid : "default");

  ncclUniqueId id;
  if (world_rank == 0) {
    CHECK_NCCL(ncclGetUniqueId(&id));
    write_unique_id(uid_path.c_str(), id);
  } else {
    read_unique_id_blocking(uid_path.c_str(), id);
  }

  // --- Create NCCL communicator ---
  ncclComm_t comm;
  CHECK_NCCL(ncclCommInitRank(&comm, world_size, id, world_rank));

  // --- Stream & a tiny device token for barriers ---
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));
  int one = 1;
  int* d_token = nullptr;
  CHECK_CUDA(cudaMalloc(&d_token, sizeof(int)));
  CHECK_CUDA(cudaMemcpy(d_token, &one, sizeof(int), cudaMemcpyHostToDevice));

  // --- Problem setup ---
  const int size = world_size;  // gathered length
  const int sendcount = 1;      // each rank contributes one float

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

  // Print initial seed values once (rank 0), e.g. 10..(10*world_size)
  if (world_rank == 0) {
    std::vector<float> init(size);
    for (int i = 0; i < size; ++i) init[i] = 10.0f * (i + 1);
    print_line_host(init.data(), size);
  }

  // --- AllGather across nodes using NCCL ---
  CHECK_NCCL(ncclAllGather(/*send*/ sendbuff_d + world_rank,
                           /*recv*/ recvbuff_d,
                           sendcount, ncclFloat, comm, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  // --- Ordered printing using NCCL barrier ---
  for (int r = 0; r < world_size; ++r) {
    nccl_barrier(comm, stream, d_token);        // before rank r prints
    if (world_rank == r) {
      printf("\nThis is rank %d, device %d\n", world_rank, device);
      fflush(stdout);
      Dev_print<<<1, size, 0, stream>>>(recvbuff_d, size);
      CHECK_CUDA(cudaGetLastError());
      CHECK_CUDA(cudaDeviceSynchronize());      // flush device printf
      printf("\n");
      fflush(stdout);
    }
    nccl_barrier(comm, stream, d_token);        // after rank r prints
  }

  // --- Cleanup ---
  CHECK_CUDA(cudaFree(sendbuff_d));
  CHECK_CUDA(cudaFree(recvbuff_d));
  CHECK_CUDA(cudaFree(d_token));
  CHECK_CUDA(cudaStreamDestroy(stream));
  ncclCommDestroy(comm);

  return 0;
}
