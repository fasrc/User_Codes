// nvcc -O3 -std=c++17 -o nccl_nompi.x nccl_nompi.cu -lnccl
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <unistd.h>         // sleep
#include <sys/stat.h>       // stat
#include <cuda_runtime.h>
#include <nccl.h>

#define CHECK_CUDA(x) do { auto e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)
#define CHECK_NCCL(x) do { auto r=(x); if(r!=ncclSuccess){ \
  fprintf(stderr,"NCCL %s:%d %s\n",__FILE__,__LINE__,ncclGetErrorString(r)); exit(1);} } while(0)

// Get ranks from Slurm without MPI
static int get_env_int(const char* k, int defv){
  const char* v = getenv(k); return v ? atoi(v) : defv;
}

// Simple shared-file OOB for ncclUniqueId (requires a shared FS, e.g. $HOME)
static void write_unique_id(const char* path, const ncclUniqueId& id){
  FILE* f = fopen(path, "wb");
  if (!f) { perror("fopen write_unique_id"); exit(1); }
  size_t n = fwrite(&id, 1, sizeof(id), f);
  fclose(f);
  if (n != sizeof(id)) { fprintf(stderr, "short write of unique id\n"); exit(1); }
}

static void read_unique_id_blocking(const char* path, ncclUniqueId& id){
  // Wait for file from rank 0 (max ~30s)
  for (int i=0; i<300; ++i) {
    struct stat st;
    if (stat(path, &st) == 0 && st.st_size == (off_t)sizeof(id)) {
      FILE* f = fopen(path, "rb");
      if (!f) { perror("fopen read_unique_id"); exit(1); }
      size_t n = fread(&id, 1, sizeof(id), f);
      fclose(f);
      if (n != sizeof(id)) { fprintf(stderr, "short read of unique id\n"); exit(1); }
      return;
    }
    usleep(100000);  // 100 ms
  }
  fprintf(stderr, "Timed out waiting for %s\n", path);
  exit(1);
}

int main() {
  // Slurm provides these without MPI:
  const int world_size = get_env_int("SLURM_NTASKS", 1);
  const int world_rank = get_env_int("SLURM_PROCID", 0);
  const int local_rank = get_env_int("SLURM_LOCALID", 0);

  int ndev = 0;
  CHECK_CUDA(cudaGetDeviceCount(&ndev));
  int device = local_rank % ndev;
  CHECK_CUDA(cudaSetDevice(device));

  // Create or read ncclUniqueId via shared file
  const char* jobid = getenv("SLURM_JOB_ID");
  const char* dir   = getenv("SLURM_SUBMIT_DIR"); // usually shared; fallback to $HOME
  if (!dir) dir = getenv("HOME");
  std::string uid_path = std::string(dir ? dir : ".") + "/nccl_uid." + (jobid ? jobid : "default");

  ncclUniqueId id;
  if (world_rank == 0) {
    CHECK_NCCL(ncclGetUniqueId(&id));
    write_unique_id(uid_path.c_str(), id);
  } else {
    read_unique_id_blocking(uid_path.c_str(), id);
  }

  ncclComm_t comm;
  CHECK_NCCL(ncclCommInitRank(&comm, world_size, id, world_rank));

  // --- Example: an AllReduce to prove inter-node comm without MPI ---
  const int N = 16;
  std::vector<float> h(N, 1.0f * (world_rank + 1));
  float *d = nullptr;
  CHECK_CUDA(cudaMalloc(&d, N*sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d, h.data(), N*sizeof(float), cudaMemcpyHostToDevice));

  cudaStream_t stream; CHECK_CUDA(cudaStreamCreate(&stream));
  CHECK_NCCL(ncclAllReduce(d, d, N, ncclFloat, ncclSum, comm, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  if (world_rank == 0) {
    std::vector<float> out(N); CHECK_CUDA(cudaMemcpy(out.data(), d, N*sizeof(float), cudaMemcpyDeviceToHost));
    printf("AllReduce sum[0] = %g (expected %d)\n", out[0], (world_size*(world_size+1))/2);
  }

  // Cleanup
  CHECK_CUDA(cudaFree(d));
  CHECK_CUDA(cudaStreamDestroy(stream));
  ncclCommDestroy(comm);
  return 0;
}

