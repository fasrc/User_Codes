#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main( int argc, char **argv){

int isend;
int *irecv = (int *) calloc (4, sizeof(int));
int rank, size;

MPI_Init( &argc, &argv );
MPI_Comm_rank( MPI_COMM_WORLD, &rank );
MPI_Comm_size( MPI_COMM_WORLD, &size );

isend = rank + 1;

MPI_Gather(&isend, 1, MPI_INT, irecv, 1, MPI_INT, 0, MPI_COMM_WORLD);

if(rank == 0)
  printf("rank = %d\tisend = %d\tirecv = %d %d %d %d\n", rank, isend, irecv[0], irecv[1], irecv[2], irecv[3]);
    else 
       printf("rank = %d\tisend = %d\tirecv = %d %d %d %d\n", rank, isend, irecv[0], irecv[1], irecv[2], irecv[3]); 

free(irecv); 
                        
MPI_Finalize();

return 0;

}/*main*/
