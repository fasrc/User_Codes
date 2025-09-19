/*
 * NCCL Broadcast (Bcast) Example on a Single Node with Multiple GPUs
 *
 * Description:
 * This program demonstrates how to use NVIDIA's NCCL library with CUDA
 * to broadcast data from one GPU (rank 0) to all other GPUs on a single node.
 * 
 * - GPU 0 initializes a vector with random integers.
 * - The data is broadcast from GPU 0 to all other GPUs using `ncclBcast`.
 * - Each GPU then runs a CUDA kernel to multiply each element of the received vector by 2.
 * - The updated values are printed directly from each GPU.
 *
 * Components:
 *  - CUDA: for device memory allocation, memory copies, kernel launch, and synchronization
 *  - NCCL: for collective communication (Broadcast from GPU 0 to all others)
 *  - CUDA kernel: to modify and print the vector contents from each GPU
 *
 * Requirements:
 *  - CUDA Toolkit
 *  - NCCL library
 *  - At least 2 GPUs on the same node (tested with 4)
 *
 * Compile:
 * nvcc -o ncclBcast.x ncclBcast.cu -lnccl -Wno-deprecated-gpu-targets
 *
 * Run:
 * ./ncclBcast.x    (or submit via Slurm with --gres=gpu:<n>)
 */
#include <nccl.h>
#include <cstdio>
#include <cstdlib>
 
__global__ void kernel(int *a) 
{
  int index = threadIdx.x;

  a[index] *= 2;
  printf("%d\t", a[index]);

}/*kernel*/
 

void print_vector(int *in, int n){

 for(int i=0; i < n; i++)
  printf("%d\t", in[i]);

 printf("\n");

}/*print_vector*/


int main(int argc, char* argv[]) {

  int data_size = 8 ;
  int nGPUs = 0;
  cudaGetDeviceCount(&nGPUs);
  
  int *DeviceList = (int *) malloc (nGPUs     * sizeof(int));
  int *data       = (int*)  malloc (data_size * sizeof(int));
  int **d_data    = (int**) malloc (nGPUs     * sizeof(int*));
  
  for(int i = 0; i < nGPUs; i++)
      DeviceList[i] = i;
  
  /*Initializing NCCL with Multiples Devices per Thread*/
  ncclComm_t* comms = (ncclComm_t*)  malloc(sizeof(ncclComm_t)  * nGPUs);  
  cudaStream_t* s   = (cudaStream_t*)malloc(sizeof(cudaStream_t)* nGPUs);
  ncclCommInitAll(comms, nGPUs, DeviceList);
  
  /*Population the data vector*/
  for(int i = 0; i < data_size; i++)
      data[i] = rand()%(10-2)*2;
 
  print_vector(data, data_size);
      
  for(int g = 0; g < nGPUs; g++) {
      cudaSetDevice(DeviceList[g]);
      cudaStreamCreate(&s[g]);
      cudaMalloc(&d_data[g], data_size * sizeof(int));
     
      if(g == 0)  /*Copy from Host to Device*/
         cudaMemcpy(d_data[g], data, data_size * sizeof(int), cudaMemcpyHostToDevice);
  }
        
  ncclGroupStart();
 
  		for(int g = 0; g < nGPUs; g++) {
  	  	    cudaSetDevice(DeviceList[g]);
    	  	    ncclBcast(d_data[g], data_size, ncclInt, 0, comms[g], s[g]); /*Broadcasting it to all*/
  		}

  ncclGroupEnd();       

  for (int g = 0; g < nGPUs; g++) {
      cudaSetDevice(DeviceList[g]);
      printf("\nThis is device %d\n", g);
      kernel <<< 1 , data_size >>> (d_data[g]);/*Call the CUDA Kernel: The code multiple the vector position per 2 on GPUs*/
      cudaDeviceSynchronize();             
  }

  printf("\n");

  for (int g = 0; g < nGPUs; g++) { /*Synchronizing CUDA Streams*/
      cudaSetDevice(DeviceList[g]);
      cudaStreamSynchronize(s[g]);
  }
 
  for(int g = 0; g < nGPUs; g++) {  /*Destroy CUDA Streams*/
      cudaSetDevice(DeviceList[g]);
      cudaStreamDestroy(s[g]);
  }

  for(int g = 0; g < nGPUs; g++)    /*Finalizing NCCL*/
     ncclCommDestroy(comms[g]);
  
  /*Freeing memory*/
  free(s);
  free(data); 
  free(DeviceList);

  cudaFree(d_data);

  return 0;

}/*main*/

