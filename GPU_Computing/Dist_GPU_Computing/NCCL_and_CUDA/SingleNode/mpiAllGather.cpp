#include <mpi.h>
#include <cstdio>
#include <cstdlib>

int main( int argc, char **argv){

int isend;
int *irecv = (int *) calloc (4, sizeof(int));
int rank, size;

MPI_Init( &argc, &argv );
MPI_Comm_rank( MPI_COMM_WORLD, &rank );
MPI_Comm_size( MPI_COMM_WORLD, &size );

switch(rank) { 
  case 0 : isend = rank + 10;  break; 
  case 1 : isend = rank + 19;  break; 
  case 2 : isend = rank + 28;  break; 
  case 3 : isend = rank + 37;  break; 
}

MPI_Allgather(&isend, 1, MPI_INT, irecv, 1, MPI_INT, MPI_COMM_WORLD);

printf("rank = %d\tisend = %d\tirecv = %d %d %d %d\n", rank, isend, irecv[0], irecv[1], irecv[2], irecv[3]);
    
free(irecv);

MPI_Finalize();

return 0;

}/*main*/
