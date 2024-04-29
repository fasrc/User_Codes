###  Purpose

C++ code example on the FASRC cluster. `arrays_and_pointers.cpp` computes dot product of 2 random vectors.

### Contents

* `arrays_and_pointers.cpp`: c++ source code 
* `runscript.sh`: batch-job submission script 

#### C++ code

```cpp
/*
  Program: arrays_and_pointers.cpp
           Arrays and pointers in C++
 */
#include <iostream>
#include <iomanip>
#include <stdlib.h>
using namespace std;

#define XTAB '\t'
#define YTAB '\v'

// Main program.............................................
int main(){
  int i;
  int n = 10;
  double arr_1d[n];
  double *p_1d;

  p_1d = arr_1d;
  for( i = 0; i < n; i++ ){
    *( p_1d + i ) = ( double )rand() / RAND_MAX;
  }

  // Results................................................
  cout << YTAB;
  cout << " Vector: " << endl;
  for( i = 0; i < n; i++ ){
    cout << setw(4) << i << setw(10) << setprecision(5) 
	 << arr_1d[i] << endl;
  }

  return 0;
}
```

#### Batch-job submission script

```bash
#!/bin/bash
#SBATCH -J arrays_and_pointers            # job name
#SBATCH -o arrays_and_pointers.out        # standard output file
#SBATCH -e arrays_and_pointers.err        # standard error file
#SBATCH -p serial_requeue                 # parition
#SBATCH -c 1                              # number of cores
#SBATCH -t 0-00:30                        # time in D-HH:MM
#SBATCH --mem=4000                        # total memory

# load required modules
# (these must be the same modules that were used for compiling)
module load gcc

# run code
./arrays_and_pointers.x
```

### Compile

We recommend compiling on a compute node. Request an interactive job to use a compute node, e.g.,

```bash
salloc --partition test --time 00:30:00 -c 2 --mem-per-cpu 2G
```

* Intel compilers, e.g.,

```bash
module load intel
icpc -O2 -o arrays_and_pointers.x arrays_and_pointers.cpp  # for intel version < 23.2, use `icpc`
icpx -O2 -o arrays_and_pointers.x arrays_and_pointers.cpp  # for intel version >= 23.2, use `icpx`.
```

* GNU compilers, e.g.,

```bash
module load gcc
g++ -O2 -o arrays_and_pointers.x arrays_and_pointers.cpp
```

### Run job

Submit job

```bash
sbatch runscript.sh
```

Example output:

```bash
 Vector:
   0   0.84019
   1   0.39438
   2    0.7831
   3   0.79844
   4   0.91165
   5   0.19755
   6   0.33522
   7   0.76823
   8   0.27777
   9   0.55397
```
