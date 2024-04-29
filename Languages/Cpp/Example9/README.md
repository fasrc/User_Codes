###  Purpose

C++ code example on the FASRC cluster. `point_func.cpp` illustrates how to use pointers to functions.

### Contents

* `point_func.cpp`: c++ source code 
* `runscript.sh`: batch-job submission script 

#### C++ code

```cpp
/*
  Program: point_func.cpp
           Illustrates use of pointers to functions
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

```

#### Batch-job submission script

```bash
#!/bin/bash
#SBATCH -J point_func            # job name
#SBATCH -o point_func.out        # standard output file
#SBATCH -e point_func.err        # standard error file
#SBATCH -p serial_requeue        # partition
#SBATCH -c 1                     # number of cores
#SBATCH -t 0-00:30               # time in D-HH:MM
#SBATCH --mem=4000               # total memory

# load required modules
# (these must be the same modules that were used for compiling)
module load gcc

# run code
./point_func.x
```

### Compile

We recommend compiling on a compute node. Request an interactive job to use a compute node, e.g.,

```bash
salloc --partition test --time 00:30:00 -c 2 --mem-per-cpu 2G
```

* Intel compilers, e.g.,

```bash
module load intel
icpc -O2 -o point_func.x point_func.cpp  # for intel version < 23.2, use `icpc`
icpx -O2 -o point_func.x point_func.cpp  # for intel version >= 23.2, use `icpx`.
```

* GNU compilers, e.g.,

```bash
module load gcc
g++ -O2 -o point_func.x point_func.cpp
```

### Run job

Submit job

```bash
sbatch runscript.sh
```

Example output:

```bash
Addition: 10 + 5 = 15
Subtraction: 10 - 5 = 5
Multiplication: 10 * 5 = 50

Plus: 15
Minus: 5
Star: 50
```
