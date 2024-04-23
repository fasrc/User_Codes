/*
  Program: allocate.cpp
  This program illustrates using dynamic memory in C++
  
  Compile: g++ -o allocate.x allocate.cpp
*/
#include <iostream>
#include <string>
#include <sstream>
#include <new>
using namespace std;

#define XTAB '\t'
#define YTAB '\v'

// Main program.............................................
int main(){
  int i;
  int n;
  int *darr;
  string mystr;

  cout << "Enter vector dimension: ";
  getline( cin, mystr );
  stringstream( mystr ) >> n;
  
  // Allocate memory........................................
  darr = new ( nothrow ) int [ n ];

  // Check if memory can be allocated.......................
  if ( darr == 0 ){
    cout << "Error: could not allocate memory.";
    cout << "Program terminates...\n";
  }
  else{
    for ( i = 0; i < n; i++ ){
      darr[i] = i;
    }
  }
  
  for ( i = 0; i < n; i++ ){
    cout << darr[i] << ", ";
  }
  cout << "End of array." << endl;

  // Free memory............................................
  delete [] darr;

  return 0;
}
