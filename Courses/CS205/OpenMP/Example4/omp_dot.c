/*
  PROGRAM: omp_dot.c
  DESCRIPTION:
    OpenMP Example - Combined Parallel Loop Reduction
    This example demonstrates a sum reduction within 
    a combined parallel loop construct.
 */
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char *argv[])  {

  int n = 100;
  int i, chunk, tid, nthreads;
  float a[n], b[n], d1, d2, pdot, ddot;
  
  /* Some initializations */
  chunk = 5;
  ddot = 0.0;
  for (i=0; i < n; i++) {
    a[i] = i*1.0;
    b[i] = i*2.0;
  }

#pragma omp parallel default(shared) private(i,d1,d2,tid,pdot)	\
  reduction(+:ddot)  
  {
    tid = omp_get_thread_num();
    nthreads = omp_get_num_threads();
    if ( tid == 0 ){
      printf("Running on %d threads.\n", nthreads);
    }
#pragma omp for schedule(static,chunk)
    for ( i = 0; i < n; i++ ) {
      d1 = a[i];
      d2 = b[i];
      pdot = pdot + ( d1 * d2 );
    }
    printf("Thread %d: partial dot product = %f\n",tid, pdot);
    ddot = ddot + pdot;
  }
  printf("Global dot product = %f\n",ddot);

 return 0;
}
