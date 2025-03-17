# Exercise 2: Job Efficiency - Memory per CPU/core ( `--mem-per-cpu` )

We use a C++ code, `omp_mem_test.cpp`, to generate a random matrix of dimension 60,000. 
Using `double precision`, the program needs ~28.8 GB of memory (since, 
60000 x 60000 x 8 bytes = 28,800,000,000 bytes)  to execute successfully. We use OpenMP to parallelize the creation of the random matrix. The specific example uses 2 OMP threads. The purpose of this exercise is to illustrate requesting the memory via the `--mem-per-cpu` option.

## Step 1: Compile C source code
We compile the code with:

```bash
module load gcc/14.2.0-fasrc01
make
```

using the `Makefile`:

```make
#==========================================================
# Make file
#==========================================================
CFLAGS   = -c -O2 -fopenmp
COMPILER = g++
PRO         = omp_mem_test
OBJECTS     = omp_mem_test.o

# Default target
all: ${PRO}.x

${PRO}.x : $(OBJECTS)
	$(COMPILER) -o ${PRO}.x $(OBJECTS) -fopenmp

%.o : %.cpp
	$(COMPILER) $(CFLAGS) $<

.PHONY: clean
clean:
	rm -f *.o *.x *.out *.err
```

This will generate the executable `omp_mem_test.x`. The C++ source code is included 
below:

```c++
#include <iostream>
#include <vector>
#include <omp.h>

// Structure to hold random number generator state
struct RNGState {
    std::vector<long> ma;
    int inext, inextp, iff;
    RNGState() : ma(55, 0), inext(0), inextp(0), iff(0) {}
};

// Function prototype for the random number generator
double ran3(int& idum, RNGState& state);

int main() {
    const int n = 60000; // Matrix dimension

    // Allocate memory for the matrix using std::vector
    std::vector<std::vector<double>> h(n, std::vector<double>(n));

    // Create random symmetric test matrix with OpenMP
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int iseed = -(99 + thread_id); // Unique seed per thread
        RNGState state; // Thread-local RNG state

        #pragma omp for schedule(dynamic)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                double val = ran3(iseed, state);
                h[i][j] = val;
                h[j][i] = val; // Symmetric matrix
            }
        }
    }
    std::cout << "Hamiltonian matrix created successfully with " 
              << omp_get_max_threads() << " threads (n=" << n << ")!" 
              << std::endl;

    // No manual memory freeing needed; std::vector handles it

    return 0;
}

// Random number generator (ran3)
double ran3(int& idum, RNGState& state) {
    const int mbig = 1000000000, mseed = 161803398, mz = 0;
    const double fac = 1.0 / mbig;
    long mj, mk;
    int i, ii, k;

    if (idum < 0 || state.iff == 0) {
        state.iff = 1;
        mj = mseed - (idum < 0 ? -idum : idum);
        mj %= mbig;
        state.ma[54] = mj;
        mk = 1;
        for (i = 1; i <= 54; i++) {
            ii = (21 * i) % 55;
            state.ma[ii - 1] = mk;
            mk = mj - mk;
            if (mk < mz) mk += mbig;
            mj = state.ma[ii - 1];
        }
        for (k = 1; k <= 4; k++) {
            for (i = 0; i < 55; i++) {
                state.ma[i] -= state.ma[(i + 30) % 55];
                if (state.ma[i] < mz) state.ma[i] += mbig;
            }
        }
        state.inext = 0;
        state.inextp = 31;
        idum = 1;
    }

    if (++state.inext == 55) state.inext = 0;
    if (++state.inextp == 55) state.inextp = 0;
    mj = state.ma[state.inext] - state.ma[state.inextp];
    if (mj < mz) mj += mbig;
    state.ma[state.inext] = mj;
    return mj * fac;
}
```

## Step 2: Create a job submission  script

The below job-submission script intentionally requests less memory than what the job
actually needs:

```bash
#!/bin/bash
#SBATCH -p test
#SBATCH -J omp_mem_test
#SBATCH -o omp_mem_test.out
#SBATCH -e omp_mem_test.err
#SBATCH -N 1
#SBATCH -c 2
#SBATCH -t 30
#SBATCH --mem-per-cpu=10G 

# Load required modules
module load gcc/14.2.0-fasrc01
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Run the code
srun -c ${SLURM_CPUS_PER_TASK} ./omp_mem_test.x
```

## Step 3: Submit the Job

If the job-submission script is named `run.sbatch`, for instance, the job is submitted
to the queue with:

```bash
sbatch run.sbatch
```
>**NOTE:** The job should fail due to insufficient memory. 

## Step 4: Diagnose the Issue

You can check the job status with:

```bash
sacct -j 6961359
JobID           JobName  Partition    Account  AllocCPUS      State ExitCode 
------------ ---------- ---------- ---------- ---------- ---------- -------- 
6961359      omp_mem_t+       test   rc_admin          2     FAILED      1:0 
6961359.bat+      batch              rc_admin          2     FAILED      1:0 
6961359.ext+     extern              rc_admin          2  COMPLETED      0:0 
6961359.0    omp_mem_t+              rc_admin          2 OUT_OF_ME+    0:125 
```

You can also check the STD error file with:

```bash
cat omp_mem_test.err 
slurmstepd: error: Detected 1 oom_kill event in StepId=6961359.0. Some of the step tasks have been OOM Killed.
srun: error: holy8a24101: task 0: Out Of Memory
```

## Step 5: Adjust the Memory Request and Resubmit the Job

Modify the job-submission script to request more memory, e.g., double the memory,

```bash
#SBATCH --mem-per-cpu=20G  # Double the original memory request 
```

and resubmit the job:

```bash
sbatch run.sbatch
Submitted batch job 6961588
```

>**NOTE:** This time the job should complete successfully.

## Step 6: Check the Job Status and Memory Efficiency

First, check the job status, e.g.,
```bash
sacct -j 6961588
JobID           JobName  Partition    Account  AllocCPUS      State ExitCode 
------------ ---------- ---------- ---------- ---------- ---------- -------- 
6961588      omp_mem_t+       test   rc_admin          2  COMPLETED      0:0 
6961588.bat+      batch              rc_admin          2  COMPLETED      0:0 
6961588.ext+     extern              rc_admin          2  COMPLETED      0:0 
6961588.0    omp_mem_t+              rc_admin          2  COMPLETED      0:0 
```
You can also check the STD output, e.g.,

```bash
cat omp_mem_test.out 
Hamiltonian matrix created successfully with 2 threads (n=60000)!
```

Second, check the memory efficiency with the `seff` command:

```bash
seff 6961588
Job ID: 6961588
Cluster: odyssey
User/Group: pkrastev/rc_admin
State: COMPLETED (exit code 0)
Nodes: 1
Cores per node: 2
CPU Utilized: 00:02:33
CPU Efficiency: 87.93% of 00:02:54 core-walltime
Job Wall-clock time: 00:01:27
Memory Utilized: 27.06 GB
Memory Efficiency: 67.66% of 40.00 GB (20.00 GB/core)
```

The Memory Efficiency is about 68%. The job used 27.06 GB while the requested memory 
is 40 GB. Please, notice that the requested memory is per core (20.00 GB/core). Adjust the requested memory so that the efficiency
is at least 80%, and resubmit the job, e.g.,

```bash
#SBATCH --mem=14G
```

Submit the job with the updated batch-job submission script,

```bash
ssbatch run.sbatch
Submitted batch job 6961865
```

check the job status,

```bash
sacct -j 6961865
JobID           JobName  Partition    Account  AllocCPUS      State ExitCode 
------------ ---------- ---------- ---------- ---------- ---------- -------- 
6961865      omp_mem_t+       test   rc_admin          2  COMPLETED      0:0 
6961865.bat+      batch              rc_admin          2  COMPLETED      0:0 
6961865.ext+     extern              rc_admin          2  COMPLETED      0:0 
6961865.0    omp_mem_t+              rc_admin          2  COMPLETED      0:0
```

and memory efficiency:

```bash
seff 6961865
Job ID: 6961865
Cluster: odyssey
User/Group: pkrastev/rc_admin
State: COMPLETED (exit code 0)
Nodes: 1
Cores per node: 2
CPU Utilized: 00:02:32
CPU Efficiency: 89.41% of 00:02:50 core-walltime
Job Wall-clock time: 00:01:25
Memory Utilized: 27.06 GB
Memory Efficiency: 96.65% of 28.00 GB (14.00 GB/core)
```

We see that the Memory Efficiency is 96.65%, while the CPU Efficiency is 89.41%.
