/*
  Program: arrays_and_pointers.cpp
           Arrays and pointers in C++

           Compile: g++ -oarrays_and_pointers.x arrays_and_pointers.cpp
 */
#include <iostream>
#include <iomanip>
#include <stdlib.h>
using namespace std;

#define XTAB '\t'
#define YTAB '\v'

// Main program.............................................
int main(){
  int i;
  int n = 10;
  double arr_1d[n];
  double *p_1d;

  p_1d = arr_1d;
  for( i = 0; i < n; i++ ){
    *( p_1d + i ) = ( double )rand() / RAND_MAX;
  }

  // Results................................................
  cout << YTAB;
  cout << " Vector: " << endl;
  for( i = 0; i < n; i++ ){
    cout << setw(4) << i << setw(10) << setprecision(5) 
	 << arr_1d[i] << endl;
  }

  return 0;
}
/*
  Example Output:

  Vector: 
   0   0.84019
   1   0.39438
   2    0.7831
   3   0.79844
   4   0.91165
   5   0.19755
   6   0.33522
   7   0.76823
   8   0.27777
   9   0.55397
 */ 
 
