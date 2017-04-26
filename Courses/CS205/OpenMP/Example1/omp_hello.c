/*
  PROGRAM: omp_hello.c
  DESCRIPTION: 
    In this simple example, the master thread forks a parallel region.
    All threads in the team obtain their unique thread number and print it.
    The master thread only prints the total number of threads.  Two OpenMP
    library routines are used to obtain the number of threads and each
    thread's number.
*/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
int main (int argc, char *argv[]) {
  int nthreads;
  int tid;
  // Parallel region starts here............................
#pragma omp parallel private(nthreads,tid)
  {    
    // Get thread ID........................................
    tid = omp_get_thread_num();
    printf("Hello World from thread = %d\n", tid);
    if ( tid == 0 ){
      // Get total number of threads........................
      nthreads = omp_get_num_threads();
      printf("Number of threads = %d\n", nthreads);
    }
  }
  // End of parallel region.................................
}
