//==============================================================
// C++ MPI example: mpitest.cpp
//==============================================================
#include <iostream>
#include <mpi.h>
using namespace std;
int main(int argc, char** argv){
  int iproc;
  int nproc;
  int i;
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&iproc);
  MPI_Comm_size(MPI_COMM_WORLD,&nproc);
  for ( i = 0; i <= nproc - 1; i++ ){
    MPI_Barrier(MPI_COMM_WORLD);
    if ( i == iproc ){
      cout << "Rank " << iproc << " out of " << nproc << endl;
    }
  }
  MPI_Finalize();
  if ( iproc == 0 ) cout << "End of program." << endl;
  return 0;
}
