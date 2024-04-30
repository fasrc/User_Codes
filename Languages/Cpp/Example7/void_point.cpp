/*
  Program: void_point.cpp
           Code illustrates use of void pointers in C++

  Compile: g++ -o void_point.x void_point.cpp
 */
#include <iostream>
using namespace std;

#define YTAB '\v'

// Function defenition......................................
void increase( void *data, int psize );

// Main program.............................................
int main(){
  char a = 'x';
  int  b = 10;

  cout << "Before:" << endl;
  cout << " a = " << a << endl;
  cout << " b = " << b << endl;

  increase( &a, sizeof(a) );
  increase( &b, sizeof(b) );

  cout << YTAB;
  cout << "After:" << endl;
  cout << " a = " << a << endl;
  cout << " b = " << b << endl;

  return 0;
}

// Functions................................................
void increase( void *data, int psize ){
  if ( psize == sizeof(char) ){
    char *pchar;
    pchar = (char*)data;
    ++(*pchar);
  }
  else if ( psize == sizeof(int) ){
    int *pint;
    pint = (int*) data;
    ++(*pint);
  }
}

/*
  Example Output:

  Before:
  a = x
  b = 10

  After:
  a = y
  b = 11
 */
