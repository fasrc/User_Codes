/*
 * NCCL ReduceScatter Example on a Single Node with 4 GPUs
 *
 * Description:
 * This program demonstrates how to use NVIDIA's NCCL library in conjunction with CUDA
 * to perform a ReduceScatter operation across 4 GPUs on a single node.
 * 
 * - Each GPU is initialized with a distinct 4-element vector.
 * - NCCL's `ncclReduceScatter` operation is used to:
 *     1. Sum the vectors element-wise across all GPUs
 *     2. Scatter one element of the result to each GPU (since recvcount = 1)
 * - A CUDA kernel is used on each GPU to print the received element.
 *
 * Example:
 * Input vectors across GPUs:
 *   GPU 0: [10, 50, 90, 130]
 *   GPU 1: [20, 60, 100, 140]
 *   GPU 2: [30, 70, 110, 150]
 *   GPU 3: [40, 80, 120, 160]
 * Resulting reduce-sum vector: [100, 260, 420, 580]
 * Each GPU receives one element:
 *   GPU 0: 100
 *   GPU 1: 260
 *   GPU 2: 420
 *   GPU 3: 580
 *
 * Components:
 *  - CUDA: for memory management, kernel execution, and synchronization
 *  - NCCL: for collective communication (ReduceScatter across devices)
 *  - CUDA kernel: to print the received element from each GPU
 *
 * Requirements:
 *  - CUDA Toolkit
 *  - NCCL library
 *  - 4 GPUs available on a single node
 *
 * Compile:
 * nvcc -o ncclReduceScatter.x ncclReduceScatter.cu -lnccl -Wno-deprecated-gpu-targets
 *
 * Run:
 * ./ncclReduceScatter.x    (or submit via Slurm with --gres=gpu:4)
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
  printf("%1.2f\t", in[i]);

 printf("\n");

}/*print_vector*/


int main(int argc, char* argv[]){

 /*Variables*/
  int size      = 4;
  int nGPUs     = 4;
  int recvcount = 1;
  int DeviceList[4] = {0, 1, 2, 3}; /* (GPUs Id) Testbed on environment with 4 GPUs*/
  
 /*Initializing NCCL with Multiples Devices per Thread*/
  ncclComm_t* comms = (ncclComm_t*)  malloc(sizeof(ncclComm_t)  * nGPUs);  
  cudaStream_t* s   = (cudaStream_t*)malloc(sizeof(cudaStream_t)* nGPUs);
  ncclCommInitAll(comms, nGPUs, DeviceList);

  /*Allocating and initializing device buffers*/
  float** sendbuff = (float**) malloc(nGPUs * sizeof(float*));
  float** recvbuff = (float**) malloc(nGPUs * sizeof(float*));

  /*Host vectors*/ 
  float host_x0[4] = { 10,  50,  90,   130};
  float host_x1[4] = { 20,  60,  100,  140};
  float host_x2[4] = { 30,  70,  110,  150};
  float host_x3[4] = { 40,  80,  120,  160};
    
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
       ncclReduceScatter(sendbuff[g], recvbuff[g], recvcount, ncclFloat, ncclSum, comms[g], s[g]); /*All Reducing and Scattering the data on GPUs*/   
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
