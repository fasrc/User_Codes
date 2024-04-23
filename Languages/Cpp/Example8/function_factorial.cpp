/*
  Program: function_factorial.cpp
           Calculates factorial of a number (e.g., 10! )

  Compile: g++ -o function_factorial.x function_factorial.cpp -O2
 */
#include <iostream>
#include <string>
#include <sstream>
#include <cmath>
#include <iomanip>
using namespace std;

#define XTAB '\t'

// Factorial function prototype.............................
long factorial( long a );

// Main program.............................................
int main(){
  long i;
  long n;
  string s;
  long r;
  stringstream ss;
  cout << "Please type an integer: ";
  getline(cin,s);
  ss.clear();
  ss << s;
  ss >> n;
  for ( i = 1 ; i <= n; i++ ){
    r = factorial(i);
    cout << n << XTAB << r << endl;
  }
  return 0;
}

// Factorial function.......................................
long factorial( long a ){
  long r;
  if ( a > 1 ){
    r = a * factorial( a - 1 );
    return( r );
  }
  else{
    return(1);
  }
}
/*
  Example output:

  Please type an integer: 10 
  10	1
  10	2
  10	6
  10	24
  10	120
  10	720
  10	5040
  10	40320
  10	362880
  10	3628800
 */

