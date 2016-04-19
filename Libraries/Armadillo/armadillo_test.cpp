/*
  Program: armadillo_test.cpp

           Illustrates use of Armadillo C++ Linear
	   Algebra Numeric Library
 */
#include <iostream>
#include <armadillo>
using namespace std;
using namespace arma;

// Main program.............................................
int main(int argc, char** argv){
  mat A = randu<mat>(4,5);
  mat B = randu<mat>(4,5);
  
  cout << A*B.t() << endl;
  
  return 0;
}
