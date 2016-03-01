/*
  Program: omp_dot.cpp
           Program creates 2 random vectors and computes their
           dot product
 */
#include <iostream>
#include <string>
#include <iomanip>
#include <new>
#include <cmath>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
using namespace std;

// Main program.............................................
int main(){
  int tid;
  int nthreads;
  int i;
  long N;
  double pdot;
  double ddot;
  double *a;
  double *b;
  double d1;
  double d2;
  double t1;
  double t2;
  double dt;

  N = 1e7;
  ddot = 0.0;

  // Allocate vectors.......................................
  a = new double[N];
  b = new double[N];

  // Initialize vectors.....................................
  for ( i = 0; i < N; i++ ){
    a[i] = (double)rand() / RAND_MAX;
    b[i] = (double)rand() / RAND_MAX;
  }

#pragma omp parallel private(i,d1,d2,tid,nthreads)
#pragma omp          reduction(+:ddot)
  {
    tid = omp_get_thread_num();
    nthreads = omp_get_num_threads();
    if ( tid == 0 ){
      cout << "Running on " << nthreads << " threads." << endl;
    }
    
    pdot = 0;
    t1 = omp_get_wtime();
# pragma omp for
    for ( i = 0; i < N; i++ ){
      d1 = a[i];
      d2 = b[i];
      pdot = pdot + ( d1 * d2 );
    }
    t2 = omp_get_wtime();
    dt = t2 - t1;
    ddot = ddot + pdot;
//    if ( tid == 0 ){
//      cout << "Partial scallar product of A and B: " << setprecision(8) << pdot << endl;
//    }
  }
  // End parallel...........................................

  cout << "Scallar product of A and B: " << setprecision(8) << ddot << endl;
  cout << "Time in FOR loop: " << dt << " seconds." << endl;

  // Deallocate vectors.....................................
  delete [] a;
  delete [] b;

  return 0;
}
