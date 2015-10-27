//==========================================================
// Program: sum.cpp
//          Computes integer sum from 1 to N where N
//          is a number read from the command line
// Compile: g++ -o sum.x sum.cpp
//==========================================================
#include <iostream>
#include <string>
#include <sstream>
using namespace std;

// Main program.............................................
int main(){
  int i;
  int n;
  int k;
  string mystr;
  cout << "Enter an integer: ";
  getline(cin,mystr);
  stringstream(mystr) >> n;
  k = 0;
  for ( i = 0; i <= n; i++ ){
    k = k + i;
  }
  cout << "Sum of integers from 1 to " << n << " is " << 
    k << '\n';
  return 0;
}
