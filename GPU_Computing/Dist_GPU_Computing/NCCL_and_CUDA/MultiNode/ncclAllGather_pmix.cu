/*
 * =============================================================================
 *  NCCL + CUDA AllGather (No MPI) with PMIx Bootstrap
 * =============================================================================
 *
 *  Overview
 *  --------
 *  Demonstrates a fully MPI-free, multi-node NCCL program that uses **PMIx**
 *  for out-of-band bootstrap/rendezvous and CUDA for device work. Each process
 *  controls exactly one GPU (chosen by its per-node local rank), contributes a
 *  single float equal to `10*(rank+1)`, and participates in an `ncclAllGather`
 *  so every rank receives the full vector:
 *
 *      [10.0, 20.0, 30.0, ..., 10.0 * world_size]
 *
 *  Bootstrap (via PMIx)
 *  --------------------
 *  - Rank 0 calls `ncclGetUniqueId()` and **publishes** it using:
 *      `PMIx_Put(PMIX_GLOBAL, "nccl_uid", ...)` + `PMIx_Commit()`
 *  - All ranks then enter `PMIx_Fence` with **PMIX_COLLECT_DATA=true** to push
 *    KVs to the server.
 *  - Non-root ranks fetch the UID with `PMIx_Get` **from the publisher proc**
 *    `{ nspace, rank=0 }` (some PMIx stacks require targeting the origin).
 *
 *  What this shows
 *  ---------------
 *  - Multi-node collectives using **NCCL only** (no MPI) for data movement
 *  - **PMIx** for rank/size discovery and exchanging the `ncclUniqueId`
 *  - One-process-per-GPU pattern across nodes
 *  - Ordered, readable output using PMIx fences as barriers
 *
 *  Requirements
 *  ------------
 *  - CUDA Toolkit
 *  - NCCL library
 *  - PMIx runtime
 *  - A cluster with ≥ 1 GPU per process (example assumes 2 nodes × 4 GPUs)
 *
 *  Build
 *  -----
 *  nvcc -O3 -std=c++17 -o ncclAllGather_pmix.x ncclAllGather_pmix.cu -lnccl -lpmix
 *
 *  Expected output (shape)
 *  -----------------------
 *  - Rank 0 prints the seed once:
 *      10.00 20.00 30.00 ... (up to 10*world_size)
 *  - Then, for ranks 0..world_size-1 in order:
 *      This is rank R, device D
 *          10.00 20.00 30.00 ... (same gathered vector, printed from device)
 *
 *  Notes / Troubleshooting
 *  -----------------------
 *  - If `PMIx_Get("nccl_uid")` times out:
 *      • Ensure a **collecting fence** runs after `PMIx_Put` (`PMIX_COLLECT_DATA=true`).
 *      • Query the **publisher proc** `{ nspace, rank=0 }`, not your own proc.
 *      • Launch with the site-supported PMIx flavor (e.g., `--mpi=pmix_v3`/`v5`).
 *  - Device selection: `device = local_rank % cudaDeviceCount()`.
 *  - For sites where Put/Get is restricted, you can swap to `PMIx_Publish` /
 *    `PMIx_Lookup` or use a tiny TCP rendezvous—still MPI-free.
 *
 *  Author: (Your Name)
 *  License: MIT (or your preferred license)
 * =============================================================================
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <nccl.h>
#include <pmix.h>

#define CHECK_CUDA(cmd) do {                                  \
  cudaError_t e = (cmd);                                      \
  if (e != cudaSuccess) {                                     \
    fprintf(stderr, "CUDA error %s:%d: %s\n",                 \
            __FILE__, __LINE__, cudaGetErrorString(e));       \
    exit(1);                                                  \
  }                                                           \
} while (0)

#define CHECK_NCCL(cmd) do {                                  \
  ncclResult_t r = (cmd);                                     \
  if (r != ncclSuccess) {                                     \
    fprintf(stderr, "NCCL error %s:%d: %s\n",                 \
            __FILE__, __LINE__, ncclGetErrorString(r));       \
    exit(1);                                                  \
  }                                                           \
} while (0)

#define CHECK_PMIX(rc, what) do {                               \
  if ((rc) != PMIX_SUCCESS) {                                   \
    fprintf(stderr, "PMIx error %s:%d: %s: %s\n",               \
            __FILE__, __LINE__, (what), PMIx_Error_string(rc)); \
    exit(1);                                                    \
  }                                                             \
} while (0)

static int getenv_int(const char* k, int defv) {
  const char* v = getenv(k);
  return v ? atoi(v) : defv;
}

__global__ void Dev_print(const float* x, int n) {
  int i = threadIdx.x;
  if (i < n) printf("%1.2f\t", x[i]);
}

static void print_line_host(const float* a, int n) {
  for (int i = 0; i < n; ++i) printf("%1.2f\t", a[i]);
  printf("\n");
}

static void pmix_fence_collect_all(const pmix_proc_t& me) {
  pmix_proc_t wild;
  PMIX_PROC_CONSTRUCT(&wild);
  std::strncpy(wild.nspace, me.nspace, PMIX_MAX_NSLEN);
  wild.rank = PMIX_RANK_WILDCARD;

  pmix_info_t* info = nullptr;
  PMIX_INFO_CREATE(info, 1);
  bool collect = true;
  PMIX_INFO_LOAD(&info[0], PMIX_COLLECT_DATA, &collect, PMIX_BOOL);

  pmix_status_t rc = PMIx_Fence(&wild, 1, info, 1);
  PMIX_INFO_FREE(info, 1);
  CHECK_PMIX(rc, "PMIx_Fence(PMIX_COLLECT_DATA=true)");

  PMIX_PROC_DESTRUCT(&wild);
}

static void pmix_barrier_all(const pmix_proc_t& me) {
  pmix_proc_t wild;
  PMIX_PROC_CONSTRUCT(&wild);
  std::strncpy(wild.nspace, me.nspace, PMIX_MAX_NSLEN);
  wild.rank = PMIX_RANK_WILDCARD;
  pmix_status_t rc = PMIx_Fence(&wild, 1, nullptr, 0);
  CHECK_PMIX(rc, "PMIx_Fence");
  PMIX_PROC_DESTRUCT(&wild);
}

// Query a job-scoped uint32 (JOB_SIZE or UNIV_SIZE), using wildcard proc
static bool pmix_get_u32_job(const pmix_proc_t& me, const char* key, uint32_t* out) {
  pmix_proc_t pr;
  PMIX_PROC_CONSTRUCT(&pr);
  std::strncpy(pr.nspace, me.nspace, PMIX_MAX_NSLEN);
  pr.rank = PMIX_RANK_WILDCARD;

  pmix_value_t* val = nullptr;
  pmix_status_t rc = PMIx_Get(&pr, key, nullptr, 0, &val);
  PMIX_PROC_DESTRUCT(&pr);
  if (rc != PMIX_SUCCESS || !val) return false;

  bool ok = false;
  if (val->type == PMIX_UINT32) { *out = val->data.uint32; ok = true; }
  else if (val->type == PMIX_SIZE) { *out = static_cast<uint32_t>(val->data.size); ok = true; }
  PMIX_VALUE_RELEASE(val);
  return ok;
}

int main(int /*argc*/, char** /*argv*/) {
  // --- PMIx init ---
  pmix_proc_t me;
  pmix_status_t prc = PMIx_Init(&me, nullptr, 0);
  CHECK_PMIX(prc, "PMIx_Init");
  int world_rank = static_cast<int>(me.rank);

  // --- world_size (robust) ---
  uint32_t wsize_u32 = 0;
  if (!pmix_get_u32_job(me, PMIX_JOB_SIZE, &wsize_u32)) {
    pmix_get_u32_job(me, PMIX_UNIV_SIZE, &wsize_u32);
  }
  if (wsize_u32 == 0) {
    int envsize = getenv_int("PMI_SIZE", -1);
    if (envsize < 0) envsize = getenv_int("SLURM_NTASKS", -1);
    if (envsize < 0) envsize = getenv_int("WORLD_SIZE", -1);
    if (envsize <= 0) {
      fprintf(stderr, "[%d] Could not determine world size via PMIx or env\n", world_rank);
      PMIx_Finalize(nullptr, 0);
      return 1;
    }
    wsize_u32 = static_cast<uint32_t>(envsize);
  }
  int world_size = static_cast<int>(wsize_u32);

  // --- local_rank (robust) ---
  int local_rank = -1;
  {
    pmix_value_t* val = nullptr;
    prc = PMIx_Get(&me, PMIX_LOCAL_RANK, nullptr, 0, &val);
    if (prc == PMIX_SUCCESS && val) {
      if (val->type == PMIX_UINT16)      local_rank = static_cast<int>(val->data.uint16);
      else if (val->type == PMIX_UINT32) local_rank = static_cast<int>(val->data.uint32);
      PMIX_VALUE_RELEASE(val);
    }
    if (local_rank < 0) {
      local_rank = getenv_int("SLURM_LOCALID", -1);
      if (local_rank < 0) local_rank = getenv_int("PMI_LOCAL_RANK", -1);
      if (local_rank < 0) local_rank = getenv_int("OMPI_COMM_WORLD_LOCAL_RANK", -1);
      if (local_rank < 0) local_rank = getenv_int("LOCAL_RANK", -1);
    }
    if (local_rank < 0) {
      fprintf(stderr, "[%d] Could not determine local_rank via PMIx or env\n", world_rank);
      PMIx_Finalize(nullptr, 0);
      return 1;
    }
  }

  // --- select GPU by local rank ---
  int ndev = 0;
  CHECK_CUDA(cudaGetDeviceCount(&ndev));
  if (ndev < 1) {
    if (world_rank == 0) fprintf(stderr, "No CUDA devices found.\n");
    PMIx_Finalize(nullptr, 0);
    return 1;
  }
  int device = local_rank % ndev;
  CHECK_CUDA(cudaSetDevice(device));

  // --- rank 0 publishes ncclUniqueId; then Fence(collect=true); non-root GETs from publisher proc ---
  ncclUniqueId id;
  if (world_rank == 0) {
    CHECK_NCCL(ncclGetUniqueId(&id));
    pmix_value_t putv;
    PMIX_VALUE_CONSTRUCT(&putv);
    putv.type = PMIX_BYTE_OBJECT;
    putv.data.bo.bytes = reinterpret_cast<char*>(&id);
    putv.data.bo.size  = sizeof(id);
    prc = PMIx_Put(PMIX_GLOBAL, "nccl_uid", &putv);
    CHECK_PMIX(prc, "PMIx_Put nccl_uid");
    prc = PMIx_Commit();
    CHECK_PMIX(prc, "PMIx_Commit");
  }

  // Push KVs to server and make visible
  pmix_fence_collect_all(me);

  if (world_rank != 0) {
    pmix_proc_t publisher;
    PMIX_PROC_CONSTRUCT(&publisher);
    std::strncpy(publisher.nspace, me.nspace, PMIX_MAX_NSLEN);
    publisher.rank = 0;  // <- query the publisher (rank 0)
    pmix_value_t* got = nullptr;

    // Optional "wait" hint (some stacks ignore it, but harmless)
    pmix_info_t* info = nullptr;
    PMIX_INFO_CREATE(info, 1);
    bool wait = true;
    PMIX_INFO_LOAD(&info[0], PMIX_WAIT, &wait, PMIX_BOOL);

    prc = PMIx_Get(&publisher, "nccl_uid", info, 1, &got);
    PMIX_INFO_FREE(info, 1);
    CHECK_PMIX(prc, "PMIx_Get nccl_uid");

    if (!got || got->type != PMIX_BYTE_OBJECT || got->data.bo.size != sizeof(id)) {
      fprintf(stderr, "[%d] Bad nccl_uid object\n", world_rank);
      if (got) PMIX_VALUE_RELEASE(got);
      PMIx_Finalize(nullptr, 0);
      return 1;
    }
    std::memcpy(&id, got->data.bo.bytes, sizeof(id));
    PMIX_VALUE_RELEASE(got);
    PMIX_PROC_DESTRUCT(&publisher);
  }

  // --- NCCL communicator ---
  ncclComm_t comm;
  CHECK_NCCL(ncclCommInitRank(&comm, world_size, id, world_rank));

  // --- CUDA stream ---
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  // --- AllGather setup ---
  const int size = world_size;
  const int sendcount = 1;

  std::vector<float> hsend(size, 0.0f);
  hsend[world_rank] = 10.0f * (world_rank + 1);

  float *sendbuff_d = nullptr, *recvbuff_d = nullptr;
  CHECK_CUDA(cudaMalloc(&sendbuff_d, size * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&recvbuff_d, size * sizeof(float)));
  CHECK_CUDA(cudaMemcpyAsync(sendbuff_d, hsend.data(),
                             size * sizeof(float), cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  if (world_rank == 0) {
    std::vector<float> init(size);
    for (int i = 0; i < size; ++i) init[i] = 10.0f * (i + 1);
    print_line_host(init.data(), size);
  }

  CHECK_NCCL(ncclAllGather(sendbuff_d + world_rank, recvbuff_d,
                           sendcount, ncclFloat, comm, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  // Ordered prints via PMIx barriers
  for (int r = 0; r < world_size; ++r) {
    pmix_barrier_all(me);
    if (world_rank == r) {
      printf("\nThis is rank %d, device %d\n", world_rank, device);
      fflush(stdout);
      Dev_print<<<1, size, 0, stream>>>(recvbuff_d, size);
      CHECK_CUDA(cudaGetLastError());
      CHECK_CUDA(cudaDeviceSynchronize());
      printf("\n");
      fflush(stdout);
    }
    pmix_barrier_all(me);
  }

  // Cleanup
  CHECK_CUDA(cudaFree(sendbuff_d));
  CHECK_CUDA(cudaFree(recvbuff_d));
  CHECK_CUDA(cudaStreamDestroy(stream));
  ncclCommDestroy(comm);

  PMIx_Finalize(nullptr, 0);
  return 0;
}

