/*
  Program: sum2.cpp

	   Computes sum of integers from 1 to N
	   ( N is read from command line )

  Compile: g++ -o sum2.x sum2.cpp
 */
#include <iostream>
#include <string>
#include <sstream>
using namespace std;

#define XTAB '\t'
#define YTAB '\v'

// Main program.............................................
int main(){
  int i;
  int n;
  int k;
  string mystr;
  cout << YTAB;
  cout << "*********************************************\n";
  cout << "* This program computes the sum of integers *\n";
  cout << "* from 1 to N, where N is an integer of our *\n";
  cout << "* choice.                                   *\n";
  cout << "*********************************************\n";
  cout << YTAB;
  cout << "Please, enter an integer: ";
  getline(cin,mystr);
  stringstream(mystr) >> n;
  k = 0; // Initialize
  for ( i = 1; i <= n; i++ ){
    k = k + i;
  }
  /* Write out results */
  cout << YTAB;
  cout << "You have entered " << n << '.' << endl;
  cout << "Sum of integers from 1 to " << n << " is " << 
    k << ".\n";
  cout << YTAB;
  return 0;
}
