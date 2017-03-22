/*
  PROGRAM: omp_orphan.c
  DESCRIPTION:
    OpenMP Example - Parallel region with an orphaned directive
    This example demonstrates a dot product  being performed by
    an orphaned loop reduction construct.
 */
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define VECLEN 100

float a[VECLEN], b[VECLEN], sum;

// Function dotprod.........................................
float dotprod (){
  int i,tid, chunk;
  float d1, d2, pdot;
  chunk = 5;
  tid = omp_get_thread_num();
#pragma omp for schedule(static,chunk) reduction(+:sum)
  for ( i = 0; i < VECLEN; i++ ){
    d1 = a[i];
    d2 = b[i];
    sum = sum + ( d1 * d2 );
    pdot = sum;
  }
  printf("Thread %d: partial dot product = %f\n",tid, pdot);
}

// Main program.............................................
int main (int argc, char *argv[]) {
  int i, tid, nthreads;

  for ( i = 0; i < VECLEN; i++ ){
    a[i] = 1.0 * i;
    b[i] = 2.0 * i;
  }
  sum = 0.0;

#pragma omp parallel
  {
    tid = omp_get_thread_num();
    nthreads = omp_get_num_threads();
    if ( tid == 0 ){
      printf("Running on %d threads.\n", nthreads);
    }

    dotprod();
  }
  
  printf("Global dot product = %f\n",sum);
 
}
