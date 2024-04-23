/*
  Program: dot_prod.cpp

           Program computes dot product of 2 random vectors
	   and illustrates use of random numbers

  Compile: g++ -o dot_prod.x dot_prod.cpp
 */
#include <iostream>
#include <string>
#include <iomanip>
#include <new>
#include <cmath>
#include <stdlib.h>
#include <time.h>
using namespace std;

#define XTAB '\t'  // Horizantal tab
#define YTAB '\v'  // Vertical tab

// Function declaration.....................................
double dot_product( double x1[], double x2[], int N );

// Main program.............................................
int main(){
  int    i;        // Loop variable
  int    N = 20;   // Vector dimension
  double x1[N];    // Vector of dimension N
  double x2[N];    // Vector of dimension N
  double ddot;     // DOT product

  // Initialize random vectors..............................
  for ( i = 0; i < N; i++ ){
    x1[i] = (double)rand() / RAND_MAX;
    x2[i] = (double)rand() / RAND_MAX;
  }

  // Compute DOT product....................................
  ddot = dot_product( x1, x2, N );

  // Result.................................................
  cout << YTAB;
  cout << " Program computes scalar product of 2 random vectors.\n ";
  cout << YTAB;
  cout << setw(10) << "x1" << setw(10) << "x2" << endl;
  for ( i = 0; i < N; i++ ){
    cout << setprecision(4) 
	 << setw(12) << x1[i] << setw(10) << x2[i] << endl;
  }
  cout << YTAB;
  cout << " Scallar product of x1 and x2: " 
       << setprecision(5) << ddot << endl;
  cout << YTAB;

  return 0;
}

// Function for computing the DOT product...................
double dot_product( double x1[], double x2[], int N ){
  int    i;
  double d1;
  double d2;
  double ddot;
  ddot = 0.0;
  for ( i = 0; i < N; i++ ){
    d1 = x1[i];
    d2 = x2[i];
    ddot = ddot + ( d1 * d2 );
  }
  return(ddot);
}
