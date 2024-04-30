###  Purpose

C++ code illustrating using C++ on the FASRC cluster. `sum.cpp` computes integer sum from 1 to N (N is read from command line). Since this code reads an input from the command line, it cannot be run as a batch job and only as an interactive job.

### Contents

* `sum.cpp`: c++ source code 

### C++ code

```cpp
//==========================================================
// Program: sum.cpp
//          Computes integer sum from 1 to N where N
//          is a number read from the command line
// Compile: g++ -o sum.x sum.cpp
//==========================================================
#include <iostream>
#include <string>
#include <sstream>
using namespace std;

// Main program.............................................
int main(){
  int i;
  int n;
  int k;
  string mystr;
  cout << "Enter an integer: ";
  getline(cin,mystr);
  stringstream(mystr) >> n;
  k = 0;
  for ( i = 0; i <= n; i++ ){
    k = k + i;
  }
  cout << "Sum of integers from 1 to " << n << " is " <<
    k << '\n';
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
icpc -o sum.x sum.cpp -O2  # for intel version < 24, use `icpc`
icpx -o sum.x sum.cpp -O2  # for intel version >= 24, use `icpx`.
```

* GNU compilers, e.g.,

```bash
module load gcc
g++ -o sum.x sum.cpp -O2
```

### Run interactive code

On a compute node

```bash
./sum.x
```

Example output

```bash
Enter an integer: 7
Sum of integers from 1 to 7 is 28
```
