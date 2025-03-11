# Exercise 3: Scaling - OpenMP 

This example illustrates evaluating the speedup of parallel applications. 
The specific example is an OpenMP (OMP) implementation of a Monte-Carlo algorithm for 
calculating $\pi$ in parallel. We will run the program on 1, 2, 4, 8, 16, 32, and 64
OMP threads, calculate the speedup and create a speedup figure. 
This is a C++ implementation.

## Contents:

* <code>omp_pi.c</code>: C++ source code
* <code>Makefile</code>: Makefile to compile the code
* <code>run.sbatch</code>: Batch-job submission script
* <code>scaling_results.txt</code>: Scaling results / Timing
* <code>speedup.py</code>: Python code to generate speedup figure
* <code>speedup.png</code>: Speedup figure

### Step 1: Compile the source code

The code is compiled with

```bash
module load intel/24.2.1-fasrc01	# Load required software modules
make             			        # Compile
```
using the `Makefile`:

```makefile
#=================================================
# Makefile
#=================================================
CFLAGS   = -c -O2 -qopenmp
COMPILER = icpx
PRO         = omp_pi
OBJECTS     = ${PRO}.o

${PRO}.x : $(OBJECTS)
	$(COMPILER) -o ${PRO}.x $(OBJECTS) -qopenmp

%.o : %.cpp
	$(COMPILER) $(CFLAGS) $(<F)

clean :
	rm -fr *.o *.x *.out *.err *.dat scaling_results.txt
```
This will generate the executable `omp_pi.x`. The C source code is included below:

```c++
#include <iostream>
#include <iomanip>    // Added for setprecision
#include <random>
#include <chrono>
#include <cmath>
#include <omp.h>

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <number_of_samples> <number_of_threads>" << std::endl;
        return 1;
    }

    int samples = std::atoi(argv[1]);    // Number of samples
    int nthreads = std::atoi(argv[2]);   // Number of threads
    omp_set_num_threads(nthreads);

    std::cout << "Number of threads: " << nthreads << std::endl;

    // Get start time
    auto t0 = omp_get_wtime();
    int count = 0;

#pragma omp parallel
    {
        // Each thread gets its own random number generator
        unsigned int seed = 1202107158 + omp_get_thread_num() * 1999;
        std::mt19937 gen(seed);  // Mersenne Twister random number generator
        std::uniform_real_distribution<double> dist(0.0, 1.0);

#pragma omp for reduction(+:count)
        for (int i = 0; i < samples; i++) {
            double x = dist(gen);
            double y = dist(gen);
            double z = x * x + y * y;
            if (z <= 1.0) {
                count++;
            }
        }
    }

    // Get end time
    auto t1 = omp_get_wtime();
    double tf = t1 - t0;

    // Estimate PI
    double PI = 4.0 * count / samples;

    // Output results
    std::cout << std::fixed << std::setprecision(5);
    std::cout << "Exact value of PI: " << M_PI << std::endl;
    std::cout << "Estimate of PI:    " << PI << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Time: " << tf << " sec." << std::endl;

    return 0;
}
```
### Step 2: Create a job submission script 

Below is an example batch-job submission script for this exercise. Use this
script to run the program with 1, 2, 4, 8, 16, 32, and 64 OMP threads. 

```bash
#!/bin/bash
#SBATCH -J omp_pi
#SBATCH -o omp_pi.out
#SBATCH -e omp_pi.err
#SBATCH -t 0-00:30
#SBATCH -p test
#SBATCH -N 1
#SBATCH -c 64
#SBATCH --mem=4G

PRO=omp_pi

# --- Load required software modules ---
module load intel/24.2.1-fasrc01
unset OMP_NUM_THREADS

# --- Run program with 1, 2, 4, 8, 16, and 32 OpenMP threads ---
echo "Number of threads: ${i}"
./${PRO}.x 100000000 ${SLURM_CPUS_PER_TASK} > ${PRO}.dat
```

### Step 3: Submit the Job

If the job-submission script is named `run.sbatch`, for instance, the job is submitted
to the queue with:

```bash
sbatch run.sbatch
```

### Step 4: Check the job status and output

Upon job completion, the results are recorded in the file `omp_pi.dat`.
You can check the job status with `sacct`, e.g.,

```bash
sacct -j 6306397
JobID           JobName  Partition    Account  AllocCPUS      State ExitCode 
------------ ---------- ---------- ---------- ---------- ---------- -------- 
6306397          omp_pi       test   rc_admin         16  COMPLETED      0:0 
6306397.bat+      batch              rc_admin         16  COMPLETED      0:0 
6306397.ext+     extern              rc_admin         16  COMPLETED      0:0 
```

and output with. e.g.,

```bash
cat omp_pi.dat
Number of threads: 16
Exact value of PI: 3.14159
Estimate of PI:    3.14161
Time: 0.77 sec.
```

### Step 5: Speedup figure

For each run, we record the runtime in a file, e.g., `scaling_results.txt`. An example is given below:

```bash
cat scaling_results.txt 
 1 12.26
 2 6.14
 4 3.07
 8 1.53
16 0.87
32 0.66
64 0.54
```

This file is used by a Python code, `speedup.py`, to generate the 
speedup figure `speedup.png`:

![Speedup](speedup.png)

We see that the program displays an good strong scaling up to 16 OMP threads, and
deteriorates afterwards.

Below we include the Python code used to calculate the speedup and generate the speedup
figure, and also an example submission script to send the figure-generating job to the queue.

**Python source code:**

```python
"""
Program: speedup.py
         Code generates speedup plot
         for nthreads = [1, 2, 4 ,8, 16, 32, 64]
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['xtick.labelsize'] = 17
mpl.rcParams['ytick.labelsize'] = 17

with open('scaling_results.txt','r') as f: 
     nproc,walltime = zip(*[ (int(i.strip().split(' ')[0]),float(i.strip().split(' ')[1])) for i in f.readlines()])

nproc      = list(nproc)
walltime   = list(walltime)

speedup = []
efficiency = []
for i in range(len(walltime)):
    s = walltime[0] / walltime[i]
    e = 100 * s / (2**i)
    speedup.append(s)
    efficiency.append(e)

# Print out results
print ("    Nthreads  Walltime  Speedup  Efficiency (%)")
for i in range(len(walltime)):
    print ("%8d %11.2f %8.2f %11.2f" % \
        (nproc[i], walltime[i], speedup[i], efficiency[i]))
    

fig, ax = plt.subplots(figsize=(8,6))
p1 = plt.plot(nproc, nproc, linewidth = 2.0, color="black",
        linestyle='-', label='Ideal speedup')
p2 = plt.plot(nproc, speedup, linewidth = 2.0, color="red",
        linestyle='--', label='Speedup')
plt.xlabel('Number of threads', fontsize=20)
plt.ylabel('Speedup', fontsize=20)
plt.legend(fontsize=15,loc=2)

plt.savefig('speedup.png', format='png')
```

> **NOTE:** To generate the scaling figure, you will need to load a Python 
module and activate a `conda` environment, e.g., `python-3.10_env`, 
(see below) containing the `numpy` and `matplotlib` packages.

Below is an example `conda` environment, e.g.,

```bash
module load python/3.10.13-fasrc01
mamba create -n python-3.10_env python=3.10 pip wheel numpy scipy matplotlib pandas seaborn h5py
```

**Submission script for the figure-generating job:**

```bash
#!/bin/bash
#SBATCH -J speedup
#SBATCH -o speedup.out
#SBATCH -e speedup.err
#SBATCH -t 0-00:30
#SBATCH -p test
#SBATCH -c 1
#SBATCH --mem=4G

# --- Set up environment ---
module load python/3.10.13-fasrc01
source activate python-3.10_env

# --- Run the python code speedup.py ---
python speedup.py
```

If we name the above script `run_speedup.sbatch`, for instance, the job is submitted to the queue as usual with:

```bash
sbatch run_speedup.sbatch
```
In addition to the speedup, the python code computes also the parallel efficiency $E(n)=S(n)/n$, which measures how efficiently you parallelize your code. Here $S(n)=T(1)/T(n)$ is the speedup, $n$ is the number of parallel processes (threads), $T(1)$ is the time to complete the program on one thread, and $T(n)$ is the time to complete the program on $n$ threads. 

Table with the results is given below:

```bash
cat speedup.out 
    Nthreads  Walltime  Speedup  Efficiency (%)
       1       12.26     1.00      100.00
       2        6.14     2.00       99.84
       4        3.07     3.99       99.84
       8        1.53     8.01      100.16
      16        0.87    14.09       88.07
      32        0.66    18.58       58.05
      64        0.54    22.70       35.47
```