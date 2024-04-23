//====================================================================
// Program: pro.c
//          Computes integer sum from 1 to N where N is an integer
//          read from the command line
//
// Compile: gcc -o sum.x sum.c -O2
//
// Usage:   ./sum.x N
//====================================================================
#include <stdio.h>
#include <stdlib.h>

// Main program.......................................................
int main(int argc, char *argv[] ){
  int i;
  int j;
  int n;
  n = atoi( argv[1] );
  printf("%s %d \n", "Welcome! This program prints out the sum of 1 to", n);
  j = 0;
  for ( i = 1; i <= n; i++ ){
    j = j + i;
  }
  printf("%s %d %s %d \n", "Sum of 1 to", n, "is", j);
  printf("%s \n", "End of program.");
  return 0;
}
