###  Purpose

C++ code example on the FASRC cluster. `allocate.cpp` illustrates using dynamic memory.

Since this code reads an input from the command line, it cannot be run as a batch job and only as an interactive job.

### Contents

* `allocate.cpp`: c++ source code 

### C++ code

```cpp
/*
  Program: allocate.cpp
  This program illustrates using dynamic memory in C++
*/
#include <iostream>
#include <string>
#include <sstream>
#include <new>
using namespace std;

#define XTAB '\t'
#define YTAB '\v'

// Main program.............................................
int main(){
  int i;
  int n;
  int *darr;
  string mystr;

  cout << "Enter vector dimension: ";
  getline( cin, mystr );
  stringstream( mystr ) >> n;

  // Allocate memory........................................
  darr = new ( nothrow ) int [ n ];

  // Check if memory can be allocated.......................
  if ( darr == 0 ){
    cout << "Error: could not allocate memory.";
    cout << "Program terminates...\n";
  }
  else{
    for ( i = 0; i < n; i++ ){
      darr[i] = i;
    }
  }

  for ( i = 0; i < n; i++ ){
    cout << darr[i] << ", ";
  }
  cout << "End of array." << endl;

  // Free memory............................................
  delete [] darr;

  return 0;
}
```

### Compile

We recommend compiling on a compute node. Request an interactive job to use a compute node, e.g.,

```bash
salloc --partition test --time 00:30:00 -c 2 --mem-per-cpu 2G
```

* Intel compilers, e.g.,

```bash
module load intel
icpc -O2 -o allocate.x allocate.cpp   # for intel version < 23.2, use `icpc`
icpx -O2 -o allocate.x allocate.cpp   # for intel version >= 23.2, use `icpx`.
```

* GNU compilers, e.g.,

```bash
module load gcc
g++ -O2 -o allocate.x allocate.cpp
```

### Run interactive code

On a compute node

```bash
./allocate.x
```

Example output

```bash
Enter vector dimension: 12
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, End of array.
```
