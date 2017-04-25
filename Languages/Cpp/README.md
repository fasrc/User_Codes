###  Purpose:

Collection of example codes illustrating using C++ on the Odyssey cluster.

### Contents:

#### Source codes

* <code>sum.cpp</code>: Computes integer sum from 1 to N (N is read from command line)
* <code>sum2.cpp</code>: Variation of the above program (sum.cpp)
* <code>allocate.cpp</code>: Illustrates using dynamic memory
* <code>matvec.cpp</code>: Performs matrix-vector multiplication
* <code>dot_prod.cpp</code>: Computes DOT product of 2 random vectors
* <code>arrays\_and\_pointers.cpp</code>: Illustrates arrays and pointers
* <code>void_point.cpp</code>: Void pointers
* <code>function_factorial.cpp</code>: Illustrates recursion
* <code>point_func.cpp</code>: Pointers to functions

#### Example Submission Script

* run.sbatch

### Compile:

* Intel compilers, e.g.,

```bash
source new-modules.sh
module load intel/17.0.2-fasrc01
icpc -o dot_prod.x dot_prod.cpp -O2
```

* GNU compilers, e.g.,

```bash
source new-modules.sh
module load gcc/6.3.0-fasrc01
g++ -o dot_prod.x dot_prod.cpp -O2
```

### Submit Job:

```bash
sbatch run.sbatch
```

### Resources:

* [cplusplus.com](http://www.cplusplus.com)
* [C++ Language Tutorial](http://www.cplusplus.com/doc/tutorial)