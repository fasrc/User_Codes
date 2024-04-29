###  Purpose

C++ code example on the FASRC cluster. `dot_prod.cpp` computes dot product of 2 random vectors.

### Contents

* `dot_prod.cpp`: c++ source code 
* `runscript.sh`: batch-job submission script 

#### C++ code

```cpp
/*
  Program: dot_prod.cpp

           Program computes dot product of 2 random vectors
	   and illustrates use of random numbers
 */
#include <iostream>
#include <string>
#include <iomanip>
#include <new>
#include <cmath>
#include <stdlib.h>
#include <time.h>
using namespace std;

#define XTAB '\t'  // Horizantal tab
#define YTAB '\v'  // Vertical tab

// Function declaration.....................................
double dot_product( double x1[], double x2[], int N );

// Main program.............................................
int main(){
  int    i;        // Loop variable
  int    N = 20;   // Vector dimension
  double x1[N];    // Vector of dimension N
  double x2[N];    // Vector of dimension N
  double ddot;     // DOT product

  // Initialize random vectors..............................
  for ( i = 0; i < N; i++ ){
    x1[i] = (double)rand() / RAND_MAX;
    x2[i] = (double)rand() / RAND_MAX;
  }

  // Compute DOT product....................................
  ddot = dot_product( x1, x2, N );

  // Result.................................................
  cout << YTAB;
  cout << " Program computes scalar product of 2 random vectors.\n ";
  cout << YTAB;
  cout << setw(10) << "x1" << setw(10) << "x2" << endl;
  for ( i = 0; i < N; i++ ){
    cout << setprecision(4) 
	 << setw(12) << x1[i] << setw(10) << x2[i] << endl;
  }
  cout << YTAB;
  cout << " Scallar product of x1 and x2: " 
       << setprecision(5) << ddot << endl;
  cout << YTAB;

  return 0;
}

// Function for computing the DOT product...................
double dot_product( double x1[], double x2[], int N ){
  int    i;
  double d1;
  double d2;
  double ddot;
  ddot = 0.0;
  for ( i = 0; i < N; i++ ){
    d1 = x1[i];
    d2 = x2[i];
    ddot = ddot + ( d1 * d2 );
  }
  return(ddot);
}
```

#### Batch-job submission script

```bash
#!/bin/bash
#SBATCH -J dot_prod            # job name
#SBATCH -o dot_prod.out        # standard output file
#SBATCH -e dot_prod.err        # standard error file
#SBATCH -p serial_requeue      # partition
#SBATCH -c 1                   # number of cores
#SBATCH -t 0-00:30             # time in D-HH:MM
#SBATCH --mem=4000             # total memory

# load required modules
# (these must be the same modules that were used for compiling)
module load gcc

# run code
./dot_prod.x
```

### Compile

We recommend compiling on a compute node. Request an interactive job to use a compute node, e.g.,

```bash
salloc --partition test --time 00:30:00 -c 2 --mem-per-cpu 2G
```

* Intel compilers, e.g.,

```bash
module load intel
icpc -O2 -o dot_prod.x dot_prod.cpp  # for intel version < 23.2, use `icpc`
icpx -O2 -o dot_prod.x dot_prod.cpp  # for intel version >= 23.2, use `icpx`.
```

* GNU compilers, e.g.,

```bash
module load gcc
g++ -O2 -o dot_prod.x dot_prod.cpp
```

### Run job

Submit job

```bash
sbatch runscript.sh
```

Example output:

```bash
 Program computes scalar product of 2 random vectors.

         x1        x2
      0.8402    0.3944
      0.7831    0.7984
      0.9116    0.1976
      0.3352    0.7682
      0.2778     0.554
      0.4774    0.6289
      0.3648    0.5134
      0.9522    0.9162
      0.6357    0.7173
      0.1416     0.607
      0.0163    0.2429
      0.1372    0.8042
      0.1567    0.4009
      0.1298    0.1088
      0.9989    0.2183
      0.5129    0.8391
      0.6126     0.296
      0.6376    0.5243
      0.4936    0.9728
      0.2925    0.7714

 Scallar product of x1 and x2: 5.5111
```
