### Purpose:

**Matrix-Matrix Multiplication**  

### Contents:

* <code>omp_mm.c</code>: C source code
* <code>omp_mm.dat</code>: Output file
* <code>Makefile</code>: Makefile to compile the code
* <code>sbatch.run</code>: Batch-job submission script

### Example Usage:

```bash
module load gcc/9.3.0-fasrc01		# Load required software modules
make             			# Compile
sbatch sbatch.run 			# Send the job to the queue
```

### Source Code:

```c
# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <omp.h>

int main ( int argc, char *argv[] );
void r8_mxm ( int l, int m, int n );
double r8_uniform_01 ( int *seed );

/******************************************************************************/

int main ( int argc, char *argv[] )

/******************************************************************************/
/*
  Purpose:

    MAIN is the main program for MXM.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    19 April 2009

  Author:

    John Burkardt
*/
{
  int id;
  int l;
  int m;
  int n;

  printf ( "\n" );
  printf ( "Matrix multiplication tests.\n" );
  printf ( "\n" );
  printf ( "  Number of processors available = %d\n", omp_get_num_procs ( ) );
  printf ( "  Number of threads              = %d\n", omp_get_max_threads ( ) );

  l = 1000;
  m = 1000;
  n = 1000;

  r8_mxm ( l, m, n );
/*
  Terminate.
*/
  printf ( "\n" );
  printf ( "omp_mm:\n" );
  printf ( "  Normal end of execution.\n" );

  return 0;
}
/******************************************************************************/

void r8_mxm ( int l, int m, int n )

/******************************************************************************/
/*
  Purpose:

    R8_MXM carries out a matrix-matrix multiplication in R8 arithmetic.

  Discussion:

    A(LxN) = B(LxM) * C(MxN).

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    13 February 2008

  Author:

    John Burkardt

  Parameters:

    Input, int L, M, N, the dimensions that specify the sizes of the
    A, B, and C matrices.
*/
{
  double *a;
  double *b;
  double *c;
  int i;
  int j;
  int k;
  int ops;
  double rate;
  int seed;
  double time_begin;
  double time_elapsed;
  double time_stop;
/*
  Allocate the matrices.
*/
  a = ( double * ) malloc ( l * n * sizeof ( double ) );
  b = ( double * ) malloc ( l * m * sizeof ( double ) );
  c = ( double * ) malloc ( m * n * sizeof ( double ) );
/*
  Assign values to the B and C matrices.
*/
  seed = 123456789;

  for ( k = 0; k < l * m; k++ )
  {
    b[k] = r8_uniform_01 ( &seed );
  }

  for ( k = 0; k < m * n; k++ )
  {
    c[k] = r8_uniform_01 ( &seed );
  }
/*
  Compute A = B * C.
*/
  time_begin = omp_get_wtime ( );

# pragma omp parallel \
  shared ( a, b, c, l, m, n ) \
  private ( i, j, k )

# pragma omp for
  for ( j = 0; j < n; j++)
  {
    for ( i = 0; i < l; i++ )
    {
      a[i+j*l] = 0.0;
      for ( k = 0; k < m; k++ )
      {
        a[i+j*l] = a[i+j*l] + b[i+k*l] * c[k+j*m];
      }
    }
  }
  time_stop = omp_get_wtime ( );
/*
  Report.
*/
  ops = l * n * ( 2 * m );
  time_elapsed = time_stop - time_begin;
  rate = ( double ) ( ops ) / time_elapsed / 1000000.0;

  printf ( "\n" );
  printf ( "Matrix multiplication timing.\n" );
  printf ( "  A(LxN) = B(LxM) * C(MxN).\n" );
  printf ( "  L = %d\n", l );
  printf ( "  M = %d\n", m );
  printf ( "  N = %d\n", n );
  printf ( "  Floating point OPS roughly %d\n", ops );
  printf ( "  Elapsed time dT = %f\n", time_elapsed );
  printf ( "  Rate = MegaOPS/dT = %f\n", rate );

  free ( a );
  free ( b );
  free ( c );

  return;
}
/******************************************************************************/

double r8_uniform_01 ( int *seed )

/******************************************************************************/
/*
  Purpose:

    R8_UNIFORM_01 is a unit pseudorandom R8.

  Discussion:

    This routine implements the recursion

      seed = 16807 * seed mod ( 2**31 - 1 )
      unif = seed / ( 2**31 - 1 )

    The integer arithmetic never requires more than 32 bits,
    including a sign bit.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    11 August 2004

  Author:

    John Burkardt

  Reference:

    Paul Bratley, Bennett Fox, Linus Schrage,
    A Guide to Simulation,
    Springer Verlag, pages 201-202, 1983.

    Bennett Fox,
    Algorithm 647:
    Implementation and Relative Efficiency of Quasirandom
    Sequence Generators,
    ACM Transactions on Mathematical Software,
    Volume 12, Number 4, pages 362-376, 1986.

  Parameters:

    Input/output, int *SEED, a seed for the random number generator.

    Output, double R8_UNIFORM_01, a new pseudorandom variate, strictly between
    0 and 1.
*/
{
  int k;
  double r;

  k = *seed / 127773;

  *seed = 16807 * ( *seed - k * 127773 ) - k * 2836;

  if ( *seed < 0 )
  {
    *seed = *seed + 2147483647;
  }

  r = ( double ) ( *seed ) * 4.656612875E-10;

  return r;
}
```

### Batch-Job Submission Script:

```bash
#!/bin/bash
#SBATCH -J omp_mm
#SBATCH -o omp_mm.out
#SBATCH -e omp_mm.err
#SBATCH -t 0-00:30
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=4000

PRO=omp_mm
rm -rf ${PRO}.dat
touch ${PRO}.dat

# Load required software modules
module load gcc/9.3.0-fasrc01

# Run program with 1, 2, 4, and 8 OpenMP threads
for i in 1 2 4 8
do
    export OMP_NUM_THREADS=${i}
    srun -c ${i} ./${PRO}.x >> ${PRO}.dat
done

```

### Example Output:

```bash
> cat omp_mm.dat 

Matrix multiplication tests.

  Number of processors available = 8
  Number of threads              = 1

Matrix multiplication timing.
  A(LxN) = B(LxM) * C(MxN).
  L = 1000
  M = 1000
  N = 1000
  Floating point OPS roughly 2000000000
  Elapsed time dT = 1.612048
  Rate = MegaOPS/dT = 1240.657594

omp_mm:
  Normal end of execution.

Matrix multiplication tests.

  Number of processors available = 8
  Number of threads              = 2

Matrix multiplication timing.
  A(LxN) = B(LxM) * C(MxN).
  L = 1000
  M = 1000
  N = 1000
  Floating point OPS roughly 2000000000
  Elapsed time dT = 0.796694
  Rate = MegaOPS/dT = 2510.374282

omp_mm:
  Normal end of execution.

Matrix multiplication tests.

  Number of processors available = 8
  Number of threads              = 4

Matrix multiplication timing.
  A(LxN) = B(LxM) * C(MxN).
  L = 1000
  M = 1000
  N = 1000
  Floating point OPS roughly 2000000000
  Elapsed time dT = 0.401481
  Rate = MegaOPS/dT = 4981.555457

omp_mm:
  Normal end of execution.

Matrix multiplication tests.

  Number of processors available = 8
  Number of threads              = 8

Matrix multiplication timing.
  A(LxN) = B(LxM) * C(MxN).
  L = 1000
  M = 1000
  N = 1000
  Floating point OPS roughly 2000000000
  Elapsed time dT = 0.199809
  Rate = MegaOPS/dT = 10009.543178

omp_mm:
  Normal end of execution.
```

