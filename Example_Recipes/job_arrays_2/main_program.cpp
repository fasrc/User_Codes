/*
  Program: main_program.cpp
           Program takes as a command line argument an integer N
	   and then computes the sum from 1 to N
 */
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <new>
#include <math.h>
#include <stdio.h>
using namespace std;

#define XTAB '\t'
#define YTAB '\v'

// Function defenition................................................
int summation( int q );

// Main program.......................................................
int main(int argc, char *argv[] ){
  int          i;
  int          j;
  int          N;
  stringstream ss;
  ss << argv[1];
  ss >> N;
  j = 0;
  for ( i = 1; i <=N; i++ ){
    j = j + i;
  }
  cout << "Sum of 1 to " << N << " is " << j << "." << endl;
  cout << "End of program." << endl;
  return 0;
}

/*
  Function: summation
            Recursive function computing sum of inetegers
	    from 1 to N
 */
int summation( int q ){
  int r;
  if ( q > 1 ){
    r = q + summation( q - 1 );
    return( r );
  }
  else {
    return( 1 );
  }
}
