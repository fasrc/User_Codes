#include <mpi.h>
#include <cstdio>
#include <cstdlib>
 
void print_vector(int rank, int *in, int n, int label){

 if(label)
  printf("[%d]\t", rank);
   else
     printf("  \t");
  
 for(int i=0; i < n; i++)
  printf("%d\t", in[i]);

 printf("\n");

}

int main(int argc, char* argv[]) {

  int i, rank, size;

  MPI_Init (&argc, &argv);                
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);  
  MPI_Comm_size (MPI_COMM_WORLD, &size); 
      
  int data_size = 8;
  
  int *data  = (int*) malloc(data_size * sizeof(int));
      
  if(rank == 0) {                            
      for(int i = 0; i < data_size; i++)
         data[i] = rand()%(10-2)*2;

      print_vector(rank, data, data_size, 0);
  }

  MPI_Bcast(data, data_size, MPI_INT, 0, MPI_COMM_WORLD);
    
  for(int i = 0; i < data_size; i++)
      data[i] *= 2;
             
  print_vector(rank, data, data_size, 1);

  MPI_Finalize();

  return 0;

}/*main*/

