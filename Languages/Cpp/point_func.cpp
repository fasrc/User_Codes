/*
  Program: point_func.cpp
           Illustrates use of pointers to functions

  Compile: g++ -o point_func.x point_func.cpp -O2
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

// Function declarations....................................
int addition( int a, int b );
int subtraction( int a, int b );
int multiplication( int a, int b );
int operation( int x, int y, int(*functiontocall)( int, int ) );

// Main program starts here.................................
int main(){
  int a = 10;
  int b = 5;
  int op1;
  int op2;
  int op3;

  int ( *plus )( int, int )  = addition;
  int ( *minus )( int, int ) = subtraction;
  int ( *star ) ( int, int ) = multiplication;

  op1 = operation( a, b, plus );
  op2 = operation( a, b, minus );
  op3 = operation( a, b, star );

  cout << "Addition: " << a << " + "<< b << " = " << op1 << endl;
  cout << "Subtraction: "  << a << " - "<< b << " = " << op2 << endl;
  cout << "Multiplication: "  << a << " * "<< b << " = " << op3 << endl; 
  cout << YTAB;
  cout << "Plus: " << (*plus)(a,b) << endl;
  cout << "Minus: " << (*minus)(a,b) << endl;
  cout << "Star: " << (*star)(a,b) << endl;

  return 0;
}

// Functions used...........................................
int addition( int a, int b ){
  int c;
  c = a + b;
  return ( c );
}

int subtraction( int a, int b ){
  int c;
  c = a - b;
  return ( c );
}

int multiplication( int a, int b ){
  int c;
  c = a * b;
  return ( c );
}

int operation( int x, int y, int (*functiontocall)( int, int) ){
  int z;
  z = ( *functiontocall )( x, y );
  return ( z );
}

/*
  Example Output:

  Addition: 10 + 5 = 15
  Subtraction: 10 - 5 = 5
  Multiplication: 10 * 5 = 50

  Plus: 15
  Minus: 5
  Star: 50
 */
