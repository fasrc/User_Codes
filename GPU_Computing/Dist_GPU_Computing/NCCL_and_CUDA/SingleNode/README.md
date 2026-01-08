# Multi-GPU Computing with NCCL - Single Node

NVIDIA Collective Communications Library (NCCL) and CUDA can be used together to efficiently implement multi-GPU communication and computation on a single node. NCCL provides optimized primitives for collective operations like `all-reduce`, `broadcast`, `reduce`, and `gather`, specifically tuned for NVIDIA GPUs and interconnects such as NVLink. When used in conjunction with CUDA, developers can offload computation to GPUs and use NCCL to synchronize and exchange data between them without needing to go through the CPU or host memory.

Here we include a collection of examples illustrating the use of NCCL on the FAS Cannon cluster. These examples demonstrate basic NCCL functionality for single-node setups using multiple GPUs, covering initialization with NCCL communicators, memory allocation and data transfers with CUDA, and synchronization across devices.

## Contents

* `ncclAllGather.cu`: NCCL+CUDA source code illustrating the `ncclAllGather` operation  
* `ncclBcast.cu`: NCCL+CUDA source code illustrating the `ncclBcast` operation
* `ncclReduce.cu`: NCCL+CUDA source code illustrating the `ncclReduce` operation
* `ncclReduceScatter.cu`: NCCL+CUDA source code illustrating the `ncclReduceScatter` operation

>Note: For details, look at the header of the source code.

## Compile

All examples in this directory can be compiled using the [NVIDIA HPC Software Development Kit (SDK)](https://developer.nvidia.com/hpc-sdk), which includes the required CUDA toolkit and NCCL libraries. The NVIDIA SDK is available as a software module:

```bash
module load nvhpc/24.11-fasrc01
```

From the folder containing your `.cu` files (e.g., `ncclAllGather.cu`, `ncclBcast.cu`, etc.), run the following:

```bash
nvcc -o ncclAllGather.x ncclAllGather.cu -lnccl -Wno-deprecated-gpu-targets
nvcc -o ncclBcast.x     ncclBcast.cu     -lnccl -Wno-deprecated-gpu-targets
nvcc -o ncclReduce.x    ncclReduce.cu    -lnccl -Wno-deprecated-gpu-targets
nvcc -o ncclReduceScatter.x ncclReduceScatter.cu -lnccl -Wno-deprecated-gpu-targets
```

Each of these commands will produce a corresponding `.x` executable, e.g., `ncclAllGather.x`, ready to run on a GPU node.

### Example source code: `ncclAllGather.cu`

```c
/*
 * NCCL AllGather Example on a Single Node with 4 GPUs
 *
 * Description:
 * This program demonstrates how to use NVIDIA's NCCL library in conjunction with CUDA
 * to perform an AllGather operation across 4 GPUs on a single node.
 * Each GPU starts with a distinct vector (with one non-zero value),
 * and after the AllGather, all GPUs receive the combined vector [10, 20, 30, 40].
 *
 * Components:
 *  - CUDA for device memory allocation, kernel launch, and synchronization
 *  - NCCL for collective communication (AllGather)
 *  - A simple CUDA kernel to print the contents of the receive buffer on each GPU
 *
 * Requirements:
 *  - CUDA Toolkit
 *  - NCCL library
 *  - 4 GPUs available on the same node
 *
 * Compile:
 * nvcc -o ncclAllGather.x ncclAllGather.cu -lnccl -Wno-deprecated-gpu-targets
 * 
 * Run:
 * ./ncclAllGather.x    (or submit via Slurm with appropriate GPU request)
 */
#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"

__global__ void Dev_print(float *x) {
   
   int i = threadIdx.x;
  
   printf("%1.2f\t", x[i]); 
  
}/*Dev_print*/   


void print_vector(float *in, int n){

 for(int i=0; i < n; i++)
  if(in[i])
   printf("%1.2f\t", in[i]);

}/*print_vector*/


int main(int argc, char* argv[]){

 /*Variables*/
  int size      = 4;
  int nGPUs     = 4;
  int sendcount = 1;
  int DeviceList[4] = {0, 1, 2, 3}; /* (GPUs Id) Testbed on environment with 4 GPUs*/
  
 /*Initializing NCCL with Multiples Devices per Thread*/
  ncclComm_t* comms = (ncclComm_t*)  malloc(sizeof(ncclComm_t)  * nGPUs);  
  cudaStream_t* s   = (cudaStream_t*)malloc(sizeof(cudaStream_t)* nGPUs);
  ncclCommInitAll(comms, nGPUs, DeviceList);

  /*Allocating and initializing device buffers*/
  float** sendbuff = (float**) malloc(nGPUs * sizeof(float*));
  float** recvbuff = (float**) malloc(nGPUs * sizeof(float*));

  /*Host vectors*/ 
  float host_x0[4] = { 10,   0,  0,  0};
  float host_x1[4] = {  0,  20,  0,  0};
  float host_x2[4] = {  0,   0, 30,  0};
  float host_x3[4] = {  0,   0,  0,  40};
    
  print_vector(host_x0, size); 
  print_vector(host_x1, size);
  print_vector(host_x2, size);
  print_vector(host_x3, size);

  for (int i = 0; i < nGPUs; ++i) {

   cudaSetDevice(i);

   cudaMalloc(&sendbuff[i],  size * sizeof(float));
   cudaMalloc(&recvbuff[i],  size * sizeof(float));

    switch(i) { /*Copy from host to devices*/
      case 0 : cudaMemcpy(sendbuff[i] , host_x0,   size * sizeof(float), cudaMemcpyHostToDevice); break; 
      case 1 : cudaMemcpy(sendbuff[i] , host_x1,   size * sizeof(float), cudaMemcpyHostToDevice); break; 
      case 2 : cudaMemcpy(sendbuff[i] , host_x2,   size * sizeof(float), cudaMemcpyHostToDevice); break; 
      case 3 : cudaMemcpy(sendbuff[i] , host_x3,   size * sizeof(float), cudaMemcpyHostToDevice); break; 
    }

   cudaStreamCreate(s+i);

  } 

  ncclGroupStart();
        
        for(int g = 0; g < nGPUs; g++) {
   	      cudaSetDevice(g);
          ncclAllGather(sendbuff[g] + g, recvbuff[g], sendcount, ncclFloat, comms[g], s[g]); /*All Gathering the data on GPUs*/
        }

  ncclGroupEnd();


  for(int g = 0; g < nGPUs; g++) {
    cudaSetDevice(g); 
    printf("\nThis is device %d\n", g);
    Dev_print <<< 1, size >>> (recvbuff[g]); /*Call the CUDA Kernel: Print vector on GPUs*/
    cudaDeviceSynchronize();    
  }

  printf("\n");

  for (int i = 0; i < nGPUs; ++i) { /*Synchronizing CUDA Streams*/
   cudaSetDevice(i);
   cudaStreamSynchronize(s[i]);
  }

  for (int i = 0; i < nGPUs; ++i) { /*Destroy CUDA Streams*/
   cudaSetDevice(i);
   cudaFree(sendbuff[i]);
   cudaFree(recvbuff[i]);
  }

  for(int i = 0; i < nGPUs; ++i)   /*Finalizing NCCL*/
    ncclCommDestroy(comms[i]);

 /*Freeing memory*/
  cudaFree(sendbuff);
  cudaFree(recvbuff);

  return 0;

}/*main*/
```

## Example Batch-Job Submission Script: `run_ncclAllGather.slurm`

```bash
#!/bin/bash
#SBATCH --job-name=ncclAllGather
#SBATCH --output=ncclAllGather.out
#SBATCH --error=ncclAllGather.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00

# Load the required software modules
module load nvhpc/24.11-fasrc01 

# Run the executable
./ncclAllGather.x
```

Run with:

```bash
sbatch run_ncclAllGather.slurm
```
## Example Output: `ncclAllGather.out`

```bash
10.00	20.00	30.00	40.00	
This is device 0
10.00	20.00	30.00	40.00	
This is device 1
10.00	20.00	30.00	40.00	
This is device 2
10.00	20.00	30.00	40.00	
This is device 3
10.00	20.00	30.00	40.00
```


