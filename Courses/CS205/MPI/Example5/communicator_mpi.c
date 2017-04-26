# include <stdlib.h>
# include <stdio.h>
# include <time.h>

# include "mpi.h"

int main ( int argc, char *argv[] );
void timestamp ( );

/******************************************************************************/

int main ( int argc, char *argv[] )

/******************************************************************************/
/*
  Purpose:

    MAIN is the main program for COMMUNICATOR_MPI.

  Discussion:

    This program demonstrates how an MPI program can start with the
    default communicator MPI_COMM_WORLD, and create new communicators
    referencing a subset of the total number of processes.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    10 January 2012

  Author:

    John Burkardt

  Reference:

    William Gropp, Ewing Lusk, Anthony Skjellum,
    Using MPI: Portable Parallel Programming with the
    Message-Passing Interface,
    Second Edition,
    MIT Press, 1999,
    ISBN: 0262571323,
    LC: QA76.642.G76.
*/
{
  MPI_Comm even_comm_id;
  MPI_Group even_group_id;
  int even_id;
  int even_id_sum;
  int even_p;
  int *even_rank;
  int i;
  int id;
  int ierr;
  int j;
  MPI_Comm odd_comm_id;
  MPI_Group odd_group_id;
  int odd_id;
  int odd_id_sum;
  int odd_p;
  int *odd_rank;
  int p;
  MPI_Group world_group_id;
  double wtime;
/*
  Initialize MPI.
*/
  ierr = MPI_Init ( &argc, &argv );
/*
  Get the number of processes.
*/
  ierr = MPI_Comm_size ( MPI_COMM_WORLD, &p );
/*
  Get the individual process ID.
*/
  ierr = MPI_Comm_rank ( MPI_COMM_WORLD, &id );
/*
  Process 0 prints an introductory message.
*/
  if ( id == 0 ) 
  {
    timestamp ( );
    printf ( "\n" );
    printf ( "COMMUNICATOR_MPI - Master process:\n" );
    printf ( "  C/MPI version\n" );
    printf ( "  An MPI example program.\n" );
    printf ( "\n" );
    printf ( "  The number of processes is %d.\n", p );
    printf ( "\n" );
  }
/*
  Every process prints a hello.
*/
  printf ( "  Process %d says 'Hello, world!'\n", id );
/*
  Get a group identifier for MPI_COMM_WORLD.
*/
  MPI_Comm_group ( MPI_COMM_WORLD, &world_group_id );
/*
  List the even processes, and create their group.
*/
  even_p = ( p + 1 ) / 2;
  even_rank = ( int * ) malloc ( even_p * sizeof ( int ) );
  j = 0;
  for ( i = 0; i < p; i = i + 2 )
  {
    even_rank[j] = i;
    j = j + 1;
  }
  MPI_Group_incl ( world_group_id, even_p, even_rank, &even_group_id );

  MPI_Comm_create ( MPI_COMM_WORLD, even_group_id, &even_comm_id );
/*
  List the odd processes, and create their group.
*/
  odd_p = p / 2;
  odd_rank = ( int * ) malloc ( odd_p * sizeof ( int ) );
  j = 0;
  for ( i = 1; i < p; i = i + 2 )
  {
    odd_rank[j] = i;
    j = j + 1;
  }
  MPI_Group_incl ( world_group_id, odd_p, odd_rank, &odd_group_id );

  MPI_Comm_create ( MPI_COMM_WORLD, odd_group_id, &odd_comm_id );
/*
  Try to get ID of each process in both groups.  
  If a process is not in a communicator, set its ID to -1.
*/
  if ( id % 2 == 0 )
  {
    ierr = MPI_Comm_rank ( even_comm_id, &even_id );
    odd_id = -1;
  }
  else
  {
    ierr = MPI_Comm_rank ( odd_comm_id,  &odd_id );
    even_id = -1;
  }
/*
  Use MPI_Reduce to sum the global ID of each process in the even group.
  Assuming 4 processes: EVEN_SUM = 0 + 2 = 2;
*/
  if ( even_id != -1 )
  {
    MPI_Reduce ( &id, &even_id_sum, 1, MPI_INT, MPI_SUM, 0, even_comm_id );
  }
  if ( even_id == 0 )
  {
    printf ( "  Number of processes in even communicator = %d\n", even_p );
    printf ( "  Sum of global ID's in even communicator  = %d\n", even_id_sum );
  }
/*
  Use MPI_Reduce to sum the global ID of each process in the odd group.
  Assuming 4 processes: ODD_SUM = 1 + 3 = 4;
*/
  if ( odd_id != -1 )
  {
    MPI_Reduce ( &id, &odd_id_sum,  1, MPI_INT, MPI_SUM, 0, odd_comm_id );
  }
  if ( odd_id == 0 )
  {
    printf ( "  Number of processes in odd communicator  = %d\n", odd_p );
    printf ( "  Sum of global ID's in odd communicator   = %d\n", odd_id_sum );
  }
/*
  Terminate MPI.
*/
  ierr = MPI_Finalize ( );
/*
  Terminate
*/
  if ( id == 0 )
  {
    printf ( "\n" );
    printf ( "COMMUNICATOR_MPI:\n" );
    printf ( "  Normal end of execution.\n" );
    printf ( "\n" );
    timestamp ( );
  }
  return 0;
}
/******************************************************************************/

void timestamp ( void )

/******************************************************************************/
/*
  Purpose:

    TIMESTAMP prints the current YMDHMS date as a time stamp.

  Example:

    31 May 2001 09:45:54 AM

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    24 September 2003

  Author:

    John Burkardt

  Parameters:

    None
*/
{
# define TIME_SIZE 40

  static char time_buffer[TIME_SIZE];
  const struct tm *tm;
  size_t len;
  time_t now;

  now = time ( NULL );
  tm = localtime ( &now );

  len = strftime ( time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm );

  fprintf ( stdout, "%s\n", time_buffer );

  return;
# undef TIME_SIZE
}
