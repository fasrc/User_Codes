### Purpose:

**Helmholtz Equation -** A program which solves the 2D Helmholtz equation.

### Contents:

* <code>omp_helmholtz.c</code>: C source code
* <code>omp_helmholtz.dat</code>: Output file
* <code>Makefile</code>: Makefile to compile the code
* <code>sbatch.run</code>: Batch-job submission script

### Example Usage:

```bash
source new-modules.sh				# Set up Harvard's LMOD environment
module load gcc/4.8.2-fasrc01		# Load required software modules
make             					# Compile
sbatch sbatch.run 					# Send the job to the queue
```

### Source Code:

```c
# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <omp.h>

int main ( int argc, char *argv[] );
void driver ( int m, int n, int it_max, double alpha, double omega, double tol );
void error_check ( int m, int n, double alpha, double u[], double f[] );
void jacobi ( int m, int n, double alpha, double omega, double u[], double f[], 
  double tol, int it_max );
double *rhs_set ( int m, int n, double alpha );
double u_exact ( double x, double y );
double uxx_exact ( double x, double y );
double uyy_exact ( double x, double y );

/******************************************************************************/

int main ( int argc, char *argv[] )

/******************************************************************************/
/*
  Purpose:

    MAIN is the main program for HELMHOLTZ.

  Discussion:

    HELMHOLTZ solves a discretized Helmholtz equation.

    The two dimensional region given is:

      -1 <= X <= +1
      -1 <= Y <= +1

    The region is discretized by a set of M by N nodes:

      P(I,J) = ( X(I), Y(J) )

    where, for 0 <= I <= M-1, 0 <= J <= N - 1, (C/C++ convention)

      X(I) = ( 2 * I - M + 1 ) / ( M - 1 )
      Y(J) = ( 2 * J - N + 1 ) / ( N - 1 )

    The Helmholtz equation for the scalar function U(X,Y) is

      - Uxx(X,Y) -Uyy(X,Y) + ALPHA * U(X,Y) = F(X,Y)

    where ALPHA is a positive constant.  We suppose that Dirichlet
    boundary conditions are specified, that is, that the value of
    U(X,Y) is given for all points along the boundary.

    We suppose that the right hand side function F(X,Y) is specified in 
    such a way that the exact solution is

      U(X,Y) = ( 1 - X^2 ) * ( 1 - Y^2 )

    Using standard finite difference techniques, the second derivatives
    of U can be approximated by linear combinations of the values
    of U at neighboring points.  Using this fact, the discretized
    differential equation becomes a set of linear equations of the form:

      A * U = F

    These linear equations are then solved using a form of the Jacobi 
    iterative method with a relaxation factor.

    Directives are used in this code to achieve parallelism.
    All do loops are parallized with default 'static' scheduling.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    19 April 2009

  Author:

    Original FORTRAN77 version by Joseph Robicheaux, Sanjiv Shah.
    C version by John Burkardt
*/
{
  double alpha = 0.25;
  int it_max = 100;
  int m = 500;
  int n = 500;
  double omega = 1.1;
  double tol = 1.0E-08;
  double wtime;

  printf ( "\n" );
  printf ( "HELMHOLTZ\n" );
  printf ( "  C/OpenMP version\n" );
  printf ( "\n" );
  printf ( "  A program which solves the 2D Helmholtz equation.\n" );

  printf ( "\n" );
  printf ( "  This program is being run in parallel.\n" );

  printf ( "\n" );
  printf ( "  Number of processors available = %d\n", omp_get_num_procs ( ) );
  printf ( "  Number of threads =              %d\n", omp_get_max_threads ( ) );

  printf ( "\n" );
  printf ( "  The region is [-1,1] x [-1,1].\n" );
  printf ( "  The number of nodes in the X direction is M = %d\n", m );
  printf ( "  The number of nodes in the Y direction is N = %d\n", n );
  printf ( "  Number of variables in linear system M * N  = %d\n", m * n );
  printf ( "  The scalar coefficient in the Helmholtz equation is ALPHA = %f\n", 
    alpha );
  printf ( "  The relaxation value is OMEGA = %f\n", omega );
  printf ( "  The error tolerance is TOL = %f\n", tol );
  printf ( "  The maximum number of Jacobi iterations is IT_MAX = %d\n", 
    it_max );
/*
  Call the driver routine.
*/
  wtime = omp_get_wtime ( );

  driver ( m, n, it_max, alpha, omega, tol );

  wtime = omp_get_wtime ( ) - wtime;

  printf ( "\n" );
  printf ( "  Elapsed wall clock time = %f\n", wtime );
/*
  Terminate.
*/
  printf ( "\n" );
  printf ( "HELMHOLTZ\n" );
  printf ( "  Normal end of execution.\n" );

  return 0;
}
/******************************************************************************/

void driver ( int m, int n, int it_max, double alpha, double omega, double tol )

/******************************************************************************/
/*
  Purpose:

    DRIVER allocates arrays and solves the problem.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    21 November 2007

  Author:

    Original FORTRAN77 version by Joseph Robicheaux, Sanjiv Shah.
    C version by John Burkardt

  Parameters:

    Input, int M, N, the number of grid points in the 
    X and Y directions.

    Input, int IT_MAX, the maximum number of Jacobi 
    iterations allowed.

    Input, double ALPHA, the scalar coefficient in the
    Helmholtz equation.

    Input, double OMEGA, the relaxation parameter, which
    should be strictly between 0 and 2.  For a pure Jacobi method,
    use OMEGA = 1.

    Input, double TOL, an error tolerance for the linear
    equation solver.
*/
{
  double *f;
  int i;
  int j;
  double *u;
/*
  Initialize the data.
*/
  f = rhs_set ( m, n, alpha );

  u = ( double * ) malloc ( m * n * sizeof ( double ) );

# pragma omp parallel \
  shared ( m, n, u ) \
  private ( i, j )

# pragma omp for

  for ( j = 0; j < n; j++ )
  {
    for ( i = 0; i < m; i++ )
    {
      u[i+j*m] = 0.0;
    }
  }
/*
  Solve the Helmholtz equation.
*/
  jacobi ( m, n, alpha, omega, u, f, tol, it_max );
/*
  Determine the error.
*/
  error_check ( m, n, alpha, u, f );

  free ( f );
  free ( u );

  return;
}
/******************************************************************************/

void error_check ( int m, int n, double alpha, double u[], double f[] )

/******************************************************************************/
/*
  Purpose:

    ERROR_CHECK determines the error in the numerical solution.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    21 November 2007

  Author:

    Original FORTRAN77 version by Joseph Robicheaux, Sanjiv Shah.
    C version by John Burkardt

  Parameters:

    Input, int M, N, the number of grid points in the 
    X and Y directions.

    Input, double ALPHA, the scalar coefficient in the
    Helmholtz equation.  ALPHA should be positive.

    Input, double U[M*N], the solution of the Helmholtz equation 
    at the grid points.

    Input, double F[M*N], values of the right hand side function 
    for the Helmholtz equation at the grid points.
*/
{
  double error_norm;
  int i;
  int j;
  double u_norm;
  double u_true;
  double u_true_norm;
  double x;
  double y;

  u_norm = 0.0;

# pragma omp parallel \
  shared ( m, n, u ) \
  private ( i, j )

# pragma omp for reduction ( + : u_norm )

  for ( j = 0; j < n; j++ )
  {
    for ( i = 0; i < m; i++ )
    {
      u_norm = u_norm + u[i+j*m] * u[i+j*m];
    }
  }

  u_norm = sqrt ( u_norm );

  u_true_norm = 0.0;
  error_norm = 0.0;

# pragma omp parallel \
  shared ( m, n, u ) \
  private ( i, j, u_true, x, y )

# pragma omp for reduction ( + : error_norm, u_true_norm)

  for ( j = 0; j < n; j++ )
  {
    for ( i = 0; i < m; i++ )
    {
      x = ( double ) ( 2 * i - m + 1 ) / ( double ) ( m - 1 );
      y = ( double ) ( 2 * j - n + 1 ) / ( double ) ( n - 1 );
      u_true = u_exact ( x, y );
      error_norm = error_norm + ( u[i+j*m] - u_true ) * ( u[i+j*m] - u_true );
      u_true_norm = u_true_norm + u_true * u_true;
    }
  }

  error_norm = sqrt ( error_norm );
  u_true_norm = sqrt ( u_true_norm );

  printf ( "\n" );
  printf ( "  Computed U l2 norm :       %f\n", u_norm );
  printf ( "  Computed U_EXACT l2 norm : %f\n", u_true_norm );
  printf ( "  Error l2 norm:             %f\n", error_norm );

  return;
}
/******************************************************************************/

void jacobi ( int m, int n, double alpha, double omega, double u[], double f[], 
  double tol, int it_max )

/******************************************************************************/
/*
  Purpose:

    JACOBI applies the Jacobi iterative method to solve the linear system.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    21 November 2007

  Author:

    Original FORTRAN77 version by Joseph Robicheaux, Sanjiv Shah.
    C version by John Burkardt

  Parameters:

    Input, int M, N, the number of grid points in the 
    X and Y directions.

    Input, double ALPHA, the scalar coefficient in the
    Helmholtz equation.  ALPHA should be positive.

    Input, double OMEGA, the relaxation parameter, which
    should be strictly between 0 and 2.  For a pure Jacobi method,
    use OMEGA = 1.

    Input/output, double U(M,N), the solution of the Helmholtz
    equation at the grid points.

    Input, double F(M,N), values of the right hand side function 
    for the Helmholtz equation at the grid points.

    Input, double TOL, an error tolerance for the linear
    equation solver.

    Input, int IT_MAX, the maximum number of Jacobi 
    iterations allowed.
*/
{
  double ax;
  double ay;
  double b;
  double dx;
  double dy;
  double error;
  double error_norm;
  int i;
  int it;
  int j;
  double *u_old;
/*
  Initialize the coefficients.
*/
  dx = 2.0 / ( double ) ( m - 1 );
  dy = 2.0 / ( double ) ( n - 1 );

  ax = - 1.0 / dx / dx;
  ay = - 1.0 / dy / dy;
  b  = + 2.0 / dx / dx + 2.0 / dy / dy + alpha;

  u_old = ( double * ) malloc ( m * n * sizeof ( double ) );

  for ( it = 1; it <= it_max; it++ )
  {
    error_norm = 0.0;
/*
  Copy new solution into old.
*/
# pragma omp parallel \
  shared ( m, n, u, u_old ) \
  private ( i, j )

# pragma omp for
    for ( j = 0; j < n; j++ )
    {
      for ( i = 0; i < m; i++ )
      {
        u_old[i+m*j] = u[i+m*j];
      }
    }
/*
  Compute stencil, residual, and update.
*/
# pragma omp parallel \
  shared ( ax, ay, b, f, m, n, omega, u, u_old ) \
  private ( error, i, j )

# pragma omp for reduction ( + : error_norm )

    for ( j = 0; j < n; j++ )
    {
      for ( i = 0; i < m; i++ )
      {
/*
  Evaluate the residual.
*/
        if ( i == 0 || i == m - 1 || j == 0 || j == n - 1 )
        {
          error = u_old[i+j*m] - f[i+j*m];
        }
        else
        {
          error = ( ax * ( u_old[i-1+j*m] + u_old[i+1+j*m] ) 
            + ay * ( u_old[i+(j-1)*m] + u_old[i+(j+1)*m] ) 
            + b * u_old[i+j*m] - f[i+j*m] ) / b;
        }
/*
  Update the solution.
*/
        u[i+j*m] = u_old[i+j*m] - omega * error;
/*
  Accumulate the residual error.
*/
        error_norm = error_norm + error * error;
      }
    }
/*
  Error check.
*/
    error_norm = sqrt ( error_norm ) / ( double ) ( m * n );

    printf ( "  %d  Residual RMS %e\n", it, error_norm );

    if ( error_norm <= tol )
    {
      break;
    }

  }

  printf ( "\n" );
  printf ( "  Total number of iterations %d\n", it );

  free ( u_old );

  return;
}
/******************************************************************************/

double *rhs_set ( int m, int n, double alpha )

/******************************************************************************/
/*
  Purpose:

    RHS_SET sets the right hand side F(X,Y).

  Discussion:

    The routine assumes that the exact solution and its second
    derivatives are given by the routine EXACT.

    The appropriate Dirichlet boundary conditions are determined
    by getting the value of U returned by EXACT.

    The appropriate right hand side function is determined by
    having EXACT return the values of U, UXX and UYY, and setting

      F = -UXX - UYY + ALPHA * U

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    21 November 2007

  Author:

    Original FORTRAN77 version by Joseph Robicheaux, Sanjiv Shah.
    C version by John Burkardt

  Parameters:

    Input, int M, N, the number of grid points in the 
    X and Y directions.

    Input, double ALPHA, the scalar coefficient in the
    Helmholtz equation.  ALPHA should be positive.

    Output, double RHS[M*N], values of the right hand side function 
    for the Helmholtz equation at the grid points.
*/
{
  double *f;
  double f_norm;
  int i;
  int j;
  double x;
  double y;

  f = ( double * ) malloc ( m * n * sizeof ( double ) );

# pragma omp parallel \
  shared ( f, m, n ) \
  private ( i, j )

# pragma omp for

  for ( j = 0; j < n; j++ )
  {
    for ( i = 0; i < m; i++ )
    {
      f[i+j*m] = 0.0;
    }
  }
/*
  Set the boundary conditions.
*/

# pragma omp parallel \
  shared ( alpha, f, m, n ) \
  private ( i, j, x, y )
  {

# pragma omp for
    for ( i = 0; i < m; i++ )
    {
      j = 0;
      y = ( double ) ( 2 * j - n + 1 ) / ( double ) ( n - 1 );
      x = ( double ) ( 2 * i - m + 1 ) / ( double ) ( m - 1 );
      f[i+j*m] = u_exact ( x, y );
    }

# pragma omp for
    for ( i = 0; i < m; i++ )
    {
      j = n - 1;
      y = ( double ) ( 2 * j - n + 1 ) / ( double ) ( n - 1 );
      x = ( double ) ( 2 * i - m + 1 ) / ( double ) ( m - 1 );
      f[i+j*m] = u_exact ( x, y );
    }

# pragma omp for
    for ( j = 0; j < n; j++ )
    {
      i = 0;
      x = ( double ) ( 2 * i - m + 1 ) / ( double ) ( m - 1 );
      y = ( double ) ( 2 * j - n + 1 ) / ( double ) ( n - 1 );
      f[i+j*m] = u_exact ( x, y );
    }

# pragma omp for

    for ( j = 0; j < n; j++ )
    {
      i = m - 1;
      x = ( double ) ( 2 * i - m + 1 ) / ( double ) ( m - 1 );
      y = ( double ) ( 2 * j - n + 1 ) / ( double ) ( n - 1 );
      f[i+j*m] = u_exact ( x, y );
    }
/*
  Set the right hand side F.
*/
# pragma omp for

    for ( j = 1; j < n - 1; j++ )
    {
      for ( i = 1; i < m - 1; i++ )
      {
        x = ( double ) ( 2 * i - m + 1 ) / ( double ) ( m - 1 );
        y = ( double ) ( 2 * j - n + 1 ) / ( double ) ( n - 1 );
        f[i+j*m] = - uxx_exact ( x, y ) - uyy_exact ( x, y ) + alpha * u_exact ( x, y );
      }
    }  
  }

  f_norm = 0.0;

# pragma omp parallel \
  shared ( f, m, n ) \
  private ( i, j )

# pragma omp for reduction ( + : f_norm )

  for ( j = 0; j < n; j++ )
  {
    for ( i = 0; i < m; i++ )
    {
      f_norm = f_norm + f[i+j*m] * f[i+j*m];
    }
  }
  f_norm = sqrt ( f_norm );

  printf ( "\n" );
  printf ( "  Right hand side l2 norm = %f\n", f_norm );

  return f;
}
/******************************************************************************/

double u_exact ( double x, double y )

/******************************************************************************/
/*
  Purpose:

    U_EXACT returns the exact value of U(X,Y).

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    21 November 2007

  Author:

    John Burkardt

  Parameters:

    Input, double X, Y, the point at which the values are needed.

    Output, double U_EXACT, the value of the exact solution.
*/
{
  double value;

  value = ( 1.0 - x * x ) * ( 1.0 - y * y );

  return value;
}
/******************************************************************************/

double uxx_exact ( double x, double y )

/******************************************************************************/
/*
  Purpose:

    UXX_EXACT returns the exact second X derivative of the solution.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    21 November 2007

  Author:

    John Burkardt

  Parameters:

    Input, double X, Y, the point at which the values are needed.

    Output, double UXX_EXACT, the exact second X derivative.
*/
{
  double value;

  value = -2.0 * ( 1.0 + y ) * ( 1.0 - y );

  return value;
}
/******************************************************************************/

double uyy_exact ( double x, double y )

/******************************************************************************/
/*
  Purpose:

    UYY_EXACT returns the exact second Y derivative of the solution.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    21 November 2007

  Author:

    John Burkardt

  Parameters:

    Input, double X, Y, the point at which the values are needed.

    Output, double UYY_EXACT, the exact second Y derivative.
*/
{
  double value;

  value = -2.0 * ( 1.0 + x ) * ( 1.0 - x );

  return value;
}
```

### Batch-Job Submission Script:

```bash
#!/bin/bash
#SBATCH -J omp_helmholtz
#SBATCH -o omp_helmholtz.out
#SBATCH -e omp_helmholtz.err
#SBATCH -p shared
#SBATCH -t 0-00:30
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=4000

# Set up environment
WORK_DIR=/scratch/${USER}/${SLURM_JOB_ID}
PRO=omp_helmholtz
### or WORK_DIR=/n/regal/cs205/${USER}/${SLURM_JOB_ID}
mkdir -pv ${WORK_DIR}
cd $WORK_DIR
cp ${SLURM_SUBMIT_DIR}/${PRO}.x .

# Load required software modules
source new-modules.sh
module load gcc/4.8.2-fasrc01

# Run program
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun -c $SLURM_CPUS_PER_TASK ./${PRO}.x > ${PRO}.dat

# Copy back the result and clean up
cp *.dat ${SLURM_SUBMIT_DIR}
rm -rf ${WORK_DIR}
```

### Example Output:

```bash
> cat omp_helmholtz.dat 

HELMHOLTZ
  C/OpenMP version

  A program which solves the 2D Helmholtz equation.

  This program is being run in parallel.

  Number of processors available = 4
  Number of threads =              4

  The region is [-1,1] x [-1,1].
  The number of nodes in the X direction is M = 500
  The number of nodes in the Y direction is N = 500
  Number of variables in linear system M * N  = 250000
  The scalar coefficient in the Helmholtz equation is ALPHA = 0.250000
  The relaxation value is OMEGA = 1.100000
  The error tolerance is TOL = 0.000000
  The maximum number of Jacobi iterations is IT_MAX = 100

  Right hand side l2 norm = 1458.457517
  1  Residual RMS 2.342892e-08
  2  Residual RMS 2.341733e-08
  3  Residual RMS 2.340960e-08
  4  Residual RMS 2.340331e-08
  5  Residual RMS 2.339784e-08
  6  Residual RMS 2.339291e-08
  7  Residual RMS 2.338839e-08
  8  Residual RMS 2.338417e-08
  9  Residual RMS 2.338021e-08
  10  Residual RMS 2.337644e-08
  11  Residual RMS 2.337285e-08
  12  Residual RMS 2.336941e-08
  13  Residual RMS 2.336609e-08
  14  Residual RMS 2.336289e-08
  15  Residual RMS 2.335979e-08
  16  Residual RMS 2.335678e-08
  17  Residual RMS 2.335385e-08
  18  Residual RMS 2.335099e-08
  19  Residual RMS 2.334821e-08
  20  Residual RMS 2.334548e-08
  21  Residual RMS 2.334282e-08
  22  Residual RMS 2.334020e-08
  23  Residual RMS 2.333764e-08
  24  Residual RMS 2.333512e-08
  25  Residual RMS 2.333265e-08
  26  Residual RMS 2.333022e-08
  27  Residual RMS 2.332782e-08
  28  Residual RMS 2.332547e-08
  29  Residual RMS 2.332315e-08
  30  Residual RMS 2.332086e-08
  31  Residual RMS 2.331860e-08
  32  Residual RMS 2.331637e-08
  33  Residual RMS 2.331416e-08
  34  Residual RMS 2.331199e-08
  35  Residual RMS 2.330984e-08
  36  Residual RMS 2.330772e-08
  37  Residual RMS 2.330561e-08
  38  Residual RMS 2.330354e-08
  39  Residual RMS 2.330148e-08
  40  Residual RMS 2.329944e-08
  41  Residual RMS 2.329743e-08
  42  Residual RMS 2.329543e-08
  43  Residual RMS 2.329345e-08
  44  Residual RMS 2.329149e-08
  45  Residual RMS 2.328954e-08
  46  Residual RMS 2.328762e-08
  47  Residual RMS 2.328571e-08
  48  Residual RMS 2.328381e-08
  49  Residual RMS 2.328193e-08
  50  Residual RMS 2.328007e-08
  51  Residual RMS 2.327821e-08
  52  Residual RMS 2.327638e-08
  53  Residual RMS 2.327455e-08
  54  Residual RMS 2.327274e-08
  55  Residual RMS 2.327094e-08
  56  Residual RMS 2.326916e-08
  57  Residual RMS 2.326738e-08
  58  Residual RMS 2.326562e-08
  59  Residual RMS 2.326387e-08
  60  Residual RMS 2.326213e-08
  61  Residual RMS 2.326040e-08
  62  Residual RMS 2.325868e-08
  63  Residual RMS 2.325698e-08
  64  Residual RMS 2.325528e-08
  65  Residual RMS 2.325359e-08
  66  Residual RMS 2.325191e-08
  67  Residual RMS 2.325024e-08
  68  Residual RMS 2.324858e-08
  69  Residual RMS 2.324693e-08
  70  Residual RMS 2.324529e-08
  71  Residual RMS 2.324365e-08
  72  Residual RMS 2.324203e-08
  73  Residual RMS 2.324041e-08
  74  Residual RMS 2.323880e-08
  75  Residual RMS 2.323720e-08
  76  Residual RMS 2.323560e-08
  77  Residual RMS 2.323402e-08
  78  Residual RMS 2.323244e-08
  79  Residual RMS 2.323087e-08
  80  Residual RMS 2.322930e-08
  81  Residual RMS 2.322774e-08
  82  Residual RMS 2.322619e-08
  83  Residual RMS 2.322465e-08
  84  Residual RMS 2.322311e-08
  85  Residual RMS 2.322158e-08
  86  Residual RMS 2.322006e-08
  87  Residual RMS 2.321854e-08
  88  Residual RMS 2.321703e-08
  89  Residual RMS 2.321552e-08
  90  Residual RMS 2.321402e-08
  91  Residual RMS 2.321253e-08
  92  Residual RMS 2.321104e-08
  93  Residual RMS 2.320956e-08
  94  Residual RMS 2.320808e-08
  95  Residual RMS 2.320661e-08
  96  Residual RMS 2.320514e-08
  97  Residual RMS 2.320368e-08
  98  Residual RMS 2.320223e-08
  99  Residual RMS 2.320078e-08
  100  Residual RMS 2.319933e-08

  Total number of iterations 101

  Computed U l2 norm :       0.640273
  Computed U_EXACT l2 norm : 266.133333
  Error l2 norm:             265.516926

  Elapsed wall clock time = 0.034580

HELMHOLTZ
  Normal end of execution.
```

