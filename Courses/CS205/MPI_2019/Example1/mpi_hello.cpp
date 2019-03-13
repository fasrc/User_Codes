/*
  Program: mpi_hello.cpp

           An intro MPI Hello World program that uses MPI_Init, 
           MPI_Comm_size, MPI_Comm_rank and MPI_Finalize.
 */
#include <iostream>
#include <mpi.h>
using namespace std;

// Main program.............................................
int main(int argc, char** argv){
  int i;
  int iproc;
  int nproc;

// Initialize MPI...........................................
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&iproc);
  MPI_Comm_size(MPI_COMM_WORLD,&nproc);

  for ( i = 0; i < nproc; i++ ){
    MPI_Barrier(MPI_COMM_WORLD);
    if ( i == iproc ){
      cout << "Hello world from process " << iproc 
	   << " out of " << nproc << endl;
    }
  }

// Shut down MPI............................................
  MPI_Finalize();
  return 0;
}
