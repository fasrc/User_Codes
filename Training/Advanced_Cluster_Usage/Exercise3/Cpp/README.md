# Exercise 3: Scaling - OpenMP 

This example illustrates evaluating the speedup of parallel applications. 
The specific example is an OpenMP (OMP) implementation of a Monte-Carlo algorithm for 
calculating $\pi$ in parallel. We will run the program on 1, 2, 4, 8, 16, 32, and 64
OMP threads, calculate the speedup and create a speedup figure. 
This is a C++ implementation.

## Contents:

* <code>omp_pi.cpp</code>: C++ source code
* <code>Makefile</code>: Makefile to compile the code
* <code>run.sbatch</code>: Batch-job submission script
* <code>scaling_results.txt</code>: Scaling results / Timing
* <code>speedup.py</code>: Python code to generate speedup figure
* <code>speedup.png</code>: Speedup figure

### Step 1: Compile the source code

The code is compiled with

```bash
module load gcc/14.2.0-fasrc01	# Load required software modules
make             			    # Compile
```
using the `Makefile`:

```makefile
#=================================================
# Makefile
#=================================================
CFLAGS   = -c -O2 -fopenmp
COMPILER = g++
PRO         = omp_pi
OBJECTS     = ${PRO}.o

${PRO}.x : $(OBJECTS)
	$(COMPILER) -o ${PRO}.x $(OBJECTS) -fopenmp

%.o : %.cpp
	$(COMPILER) $(CFLAGS) $(<F)

clean :
	rm -fr *.o *.x *.out *.err *.dat
```
This will generate the executable `omp_pi.x`. The C++ source code is included below:

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
#SBATCH -c 1
#SBATCH --mem=4G

PRO=omp_pi

# --- Load required software modules ---
module load gcc/14.2.0-fasrc01
unset OMP_NUM_THREADS

# --- Run program with 1, 2, 4, 8, 16, 32, and 64 OpenMP threads ---
echo "Number of threads: ${i}"
srun -c ${SLURM_CPUS_PER_TASK} ./${PRO}.x 1000000000 ${SLURM_CPUS_PER_TASK} > ${PRO}.dat
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
Number of threads: 4
Exact value of PI: 3.14159
Estimate of PI:    3.14165
Time: 4.94 sec.
```

### Step 5: Speedup figure

For each run, we record the runtime in a file, e.g., `scaling_results.txt`. An example is given below:

```bash
cat scaling_results.txt 
 1 19.72
 2 9.87
 4 4.94
 8 2.47
16 1.24
32 0.63
64 0.32
```

This file is used by a Python code, `speedup.py`, to generate the 
speedup figure `speedup.png`:

![Speedup](speedup.png)

We see that the program displays an excellent strong scaling up to 64 OMP threads.

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

# Get data
infile = "scaling_results.txt"
darr = np.loadtxt(infile, skiprows=0)
nproc    = darr[:,0]
walltime = darr[:,1]

# Compute speedup and parallel efficiency
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
    
# Speedup figure
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
module and activate a `conda` environment, e.g., `speedup_env`, containing the `numpy` and `matplotlib` packages. If not already, please create the `speedup_env` conda environment.

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
module load python/3.12.8-fasrc01
source activate speedup_env 

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
       1       19.72     1.00      100.00
       2        9.87     2.00       99.90
       4        4.94     3.99       99.80
       8        2.47     7.98       99.80
      16        1.24    15.90       99.40
      32        0.63    31.30       97.82
      64        0.32    61.62       96.29    
```