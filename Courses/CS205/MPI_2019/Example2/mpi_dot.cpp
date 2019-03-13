/*
  PROGRAM: mpi_dot.c
  DESCRIPTION:
       MPI Example - parallel dot product
       This example demonstrates a sum reduction within 
       a parallel loop construct. We use the para_range
       function to distribute the loop iterations.
*/
#include <iostream>
#include <mpi.h>
using namespace std;

#define XTAB '\t'
#define YTAB '\v'

// Fuction prototypes............................................................
void para_range(int n1, int n2, int &nprocs, int &irank, int &ista, int &iend);

// Main program starts here......................................................
int main(int argc, char** argv){
  int i;
  int iproc;
  int nproc;
  int ista;
  int iend;
  int loc_dim;
  int N = 100;
  float *a;
  float *b;
  float d1;
  float d2;
  float pdot;
  float ddot;

  // Initialize MPI..............................................................
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&iproc);
  MPI_Comm_size(MPI_COMM_WORLD,&nproc);

  // Call "para_range" to compute lowest and highest iteration ranges for each MPI task
  para_range( 0, N-1, nproc, iproc, ista, iend );

  // Calculate local dot product.................................................
  loc_dim = iend - ista + 1; // Local DIM
  a = new float[loc_dim];
  b = new float[loc_dim];

  // Some initializations........................................................
  for ( i = 0; i < loc_dim; i++ ){
    a[i] = ( i + ista + 1 ) * 1.0;
    b[i] = ( i + ista + 1 ) * 2.0;
  }
  
  pdot = 0.0;
  for ( i = 0; i < loc_dim; i++ ) {
    d1 = a[i];
    d2 = b[i];
    pdot = pdot + ( d1 * d2 );
  }
  
  // Get global dot product......................................................
  MPI_Reduce(&pdot, &ddot, 1, MPI_REAL, MPI_SUM, 0, MPI_COMM_WORLD);

  // Print out results...........................................................
  cout << "Local dot product for MPI process " 
       << iproc << ": " << pdot << endl;
  MPI_Barrier(MPI_COMM_WORLD);
  if ( iproc == 0 ) cout << "Global dot product: " << ddot << endl;

  // Shut down MPI...............................................................
  MPI_Finalize();
  return 0;
}

// Functions used my main program................................................
//===============================================================================
// Calculates iteration range and/or array dimension for each MPI process
// Adapted from "RS/6000 SP: Practical MPI Programming", IBM red book
//===============================================================================
void para_range(int n1, int n2, int &nprocs, int &irank, int &ista, int &iend){
  int iwork1;
  int iwork2;
  iwork1 = ( n2 - n1 + 1 )  /  nprocs;
  iwork2 = ( ( n2 - n1 + 1 ) %  nprocs );
  ista = irank * iwork1 + n1 + min(irank, iwork2);
  iend = ista + iwork1 - 1;
  if ( iwork2 > irank ) iend = iend + 1;
}
