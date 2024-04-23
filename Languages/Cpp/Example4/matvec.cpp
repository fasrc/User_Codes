/*
  Program: matvec.cpp

           Matrix-vector multiplication
	   Program generates random vector (vin) and matrix (h)
	   and performs matrix-vector multiplication (creating
           the vector vout)

  Compile: g++ -o matvec.x matvec.cpp -O2

  Run: ./matvec.x

  Example output:

  ./matvec.x 
  Problem dimension: 10
  Final vector:
           1.77476
	   2.40325
	   2.80046
	   2.94970
	   2.80791
	   2.92653
	   2.62936
	   2.40287
	   2.96731
	   2.76708
 */
#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include <new>
#include <cmath>
using namespace std;

#define XTAB '\t'
#define YTAB '\v'
#define width 20
#define prec  8

// Function prototypes......................................
double ran3(long *idum);
void vector_init( double *v, int n );
void matrix_init( double **h, int n );
void mat_vec( double **h, double *vin, double *vout, int n);
void vector_print( double *v, int n );

// Main program.............................................
int main(){
  int          i;
  int          j;
  int          k;
  int          n;
  string       s;
  stringstream ss;

  double       *vin  = NULL;
  double       *vout = NULL;
  double       **h   = NULL;

  // Get problem dimension..................................
  cout << "Problem dimension: ";
  getline( cin, s );
  ss.clear();
  ss << s;
  ss >> n;

  // Allocate arrays........................................
  vin  = new double[n];
  vout = new double[n];
  h    = new double*[n];
  for ( i = 0; i < n; i++ ){
    h[i] = new double[n];
  }

  // Matrix-vector multiplication...........................
  vector_init( vin, n );
  matrix_init( h, n );
  mat_vec( h, vin, vout, n );

  // Print out results......................................
  cout << "Final vector:" << endl;
  vector_print( vout, n);

  // Deallocate arrays......................................
  delete [] vin;
  delete [] vout;
  for ( i = 0; i < n; i++ ){
    delete [] h[i];
  }
  delete [] h;

  return 0;
}

// Functions used by main program...........................

// Generate random vector of dimension n....................
void vector_init( double *v, int n ){
  int i;
  long iseed = -99;
  for ( i = 0; i < n; i++ ){
    v[i] = ran3(&iseed);
  }
}

// Generate random symmetric matrix of dimension n X n......
void matrix_init( double **h, int n ){
  int i;
  int j;
  long iseed = -99;
  for ( i = 0; i < n; i++ ){
    for ( j = 0; j <= i; j++ ){
      h[i][j] = ran3(&iseed);
      h[j][i] = h[i][j];
    }
  }
}

// Matrix-vector multiplication.............................
void mat_vec( double **h, double *vin, double *vout, int n ){
  int i;
  int j;
  for ( i = 0; i < n; i++ ){
    vout[i] = 0.0;
    for ( j = 0; j < n; j++ ){
      vout[i] = vout[i] + h[i][j] * vin[j];
    }
  }
}

// Print out vector.........................................
void vector_print( double *v, int n){
  int i;
  for ( i = 0; i < n; i++ ){
    cout << showpoint << setw(width) << v[i] << endl;
  }
}

/*
  The function
  ran3()
  returns a uniform random number deviate between 0.0 and 1.0. Set
  the idum to any negative value to initialize or reinitialize the
  sequence. Any large MBIG, and any small (but still large) MSEED
  can be substituted for the present values. 
*/

#define MBIG 1000000000
#define MSEED 161803398
#define MZ 0
#define FAC (1.0/MBIG)

double ran3(long *idum)
{
  static int        inext;
  static int        inextp;
  static long       ma[56];                 // value 56 is special, do not modify
  static int        iff = 0;
  long              mj;
  long              mk;
  int               i;
  int               ii; 
  int               k;

  if ( *idum < 0 || iff == 0 ) {             // initialization
    iff    = 1;
    
    mj     = MSEED - (*idum < 0 ? -*idum : *idum);
    mj    %= MBIG;
    ma[55] = mj;                             // initialize ma[55] 

    for ( i = 1, mk = 1; i <= 54; i++ ) {    // initialize rest of table 
      ii     = (21*i) % 55;
      ma[ii] = mk;
      mk     = mj - mk;
      if ( mk < MZ ) mk += MBIG;
      mj = ma[ii];
    }

    for ( k = 1; k <= 4; k++ ) {             // randimize by "warming up" the generator
      for ( i = 1; i <= 55; i++ ) {
	ma[i] -= ma[1 + ( i + 30 ) % 55];
	if ( ma[i] < MZ ) ma[i] += MBIG;
      }
    }
    
    inext  =  0;                             // prepare indices for first generator number
    inextp = 31;                             // 31 is special
    *idum  = 1;
  }

  if ( ++inext == 56 )  inext  = 1;
  if ( ++inextp == 56 ) inextp = 1;
  mj = ma[inext] - ma[inextp];
  if ( mj < MZ ) mj += MBIG;
  ma[inext] = mj;
  return mj*FAC;
}
#undef MBIG
#undef MSEED
#undef MZ
#undef FAC
