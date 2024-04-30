###  Purpose

C++ code example on the FASRC cluster. `function_factorial.cpp` shows how to do recursion.

Since this code reads an input from the command line, it cannot be run as a batch job and only as an interactive job.

### Contents

* `function_factorial.cpp`: c++ source code 

### C++ code

```cpp
/*
  Program: function_factorial.cpp
           Calculates factorial of a number (e.g., 10! )
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
    cout << i << XTAB << r << endl;
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
```

### Compile

We recommend compiling on a compute node. Request an interactive job to use a compute node, e.g.,

```bash
salloc --partition test --time 00:30:00 -c 2 --mem-per-cpu 2G
```

* Intel compilers, e.g.,

```bash
module load intel
icpc -O2 -o function_factorial.x function_factorial.cpp  # for intel version < 23.2, use `icpc`
icpx -O2 -o function_factorial.x function_factorial.cpp  # for intel version >= 23.2, use `icpx`.
```

* GNU compilers, e.g.,

```bash
module load gcc
g++ -O2 -o function_factorial.x function_factorial.cpp
```

### Run interactive code

On a compute node

```bash
./function_factorial.x
```

Example output

```
Please type an integer: 10
1	1
2	2
3	6
4	24
5	120
6	720
7	5040
8	40320
9	362880
10	3628800
```
