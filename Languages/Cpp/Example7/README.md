###  Purpose

C++ code example on the FASRC cluster. `void_point.cpp` illustrates arrays and pointers.

### Contents

* `void_point.cpp`: c++ source code 
* `runscript.sh`: batch-job submission script 

#### C++ code

```cpp
/*
  Program: void_point.cpp
           Code illustrates use of void pointers in C++
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
```

#### Batch-job submission script

```bash
#!/bin/bash
#SBATCH -J void_point            # job name
#SBATCH -o void_point.out        # standard output file
#SBATCH -e void_point.err        # standard error file
#SBATCH -p serial_requeue        # parition
#SBATCH -c 1                     # number of cores
#SBATCH -t 0-00:30               # time in D-HH:MM
#SBATCH --mem=4000               # total memory

# load required modules
# (these must be the same modules that were used for compiling)
module load gcc

# run code
./void_point.x
```

### Compile

We recommend compiling on a compute node. Request an interactive job to use a compute node, e.g.,

```bash
salloc --partition test --time 00:30:00 -c 2 --mem-per-cpu 2G
```

* Intel compilers, e.g.,

```bash
module load intel
icpc -O2 -o void_point.x void_point.cpp  # for intel version < 23.2, use `icpc`
icpx -O2 -o void_point.x void_point.cpp  # for intel version >= 23.2, use `icpx`.
```

* GNU compilers, e.g.,

```bash
module load gcc
g++ -O2 -o void_point.x void_point.cpp
```

### Run job

Submit job

```bash
sbatch runscript.sh
```

Example output:

```bash
Before:
 a = x
 b = 10

After:
 a = y
 b = 11
```
