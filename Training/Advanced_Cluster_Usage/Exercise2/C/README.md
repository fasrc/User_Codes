# Exercise 1: Job Efficiency - Memory per CPU/core ( `--mem-per-cpu` )

We use a C code, `omp_mem_test.c`, to generate a random matrix of dimension 60,000. 
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
COMPILER = gcc
PRO         = omp_mem_test
OBJECTS     = omp_mem_test.o

# Default target
all: ${PRO}.x

${PRO}.x : $(OBJECTS)
	$(COMPILER) -o ${PRO}.x $(OBJECTS) -fopenmp

%.o : %.c
	$(COMPILER) $(CFLAGS) $<

.PHONY: clean
clean:
	rm -f *.o *.x *.out *.err
```

This will generate the executable `omp_mem_test.x`. The C source code is included 
below:

```c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Structure to hold random number generator state
typedef struct {
    long ma[55];
    int inext, inextp, iff;
} RNGState;

double ran3(int *idum, RNGState *state);

int main() {
    int n = 60000; // Reduced size for testing (adjust as needed)
    int i;
    double **h;

    // Allocate memory with error checking
    h = (double **)malloc(n * sizeof(double *));
    if (h == NULL) {
        fprintf(stderr, "Failed to allocate row pointers\n");
        return 1;
    }
    for (i = 0; i < n; i++) {
        h[i] = (double *)malloc(n * sizeof(double));
        if (h[i] == NULL) {
            fprintf(stderr, "Failed to allocate row %d\n", i);
            for (int j = 0; j < i; j++) free(h[j]);
            free(h);
            return 1;
        }
    }

    // Parallel region
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int iseed = -(99 + thread_id);
        RNGState state = {{0}, 0, 0, 0}; // Thread-local RNG state

        #pragma omp for schedule(dynamic)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                double val = ran3(&iseed, &state);
                h[i][j] = val;
                h[j][i] = val; // Symmetry (j <= i < n, so no bounds check needed)
            }
        }
    }
    printf("Matrix created successfully with %d threads (n=%d)!\n", 
           omp_get_max_threads(), n);

    // Free memory
    for (i = 0; i < n; i++) {
        free(h[i]);
    }
    free(h);

    return 0;
}

double ran3(int *idum, RNGState *state) {
    const int mbig = 1000000000, mseed = 161803398, mz = 0;
    const double fac = 1.0 / 1000000000.0;
    long mj, mk;
    int i, ii, k;

    if (*idum < 0 || state->iff == 0) {
        state->iff = 1;
        mj = mseed - (*idum < 0 ? -*idum : *idum);
        mj %= mbig;
        state->ma[54] = mj;
        mk = 1;
        for (i = 1; i <= 54; i++) {
            ii = (21 * i) % 55;
            state->ma[ii - 1] = mk;
            mk = mj - mk;
            if (mk < mz) mk += mbig;
            mj = state->ma[ii - 1];
        }
        for (k = 1; k <= 4; k++) {
            for (i = 0; i < 55; i++) {
                state->ma[i] -= state->ma[(i + 30) % 55];
                if (state->ma[i] < mz) state->ma[i] += mbig;
            }
        }
        state->inext = 0;
        state->inextp = 31;
        *idum = 1;
    }

    if (++state->inext == 55) state->inext = 0;
    if (++state->inextp == 55) state->inextp = 0;
    mj = state->ma[state->inext] - state->ma[state->inextp];
    if (mj < mz) mj += mbig;
    state->ma[state->inext] = mj;
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
sacct -j 6959634
JobID           JobName  Partition    Account  AllocCPUS      State ExitCode 
------------ ---------- ---------- ---------- ---------- ---------- -------- 
6959634      omp_mem_t+       test   rc_admin          2     FAILED      1:0 
6959634.bat+      batch              rc_admin          2     FAILED      1:0 
6959634.ext+     extern              rc_admin          2  COMPLETED      0:0 
6959634.0    omp_mem_t+              rc_admin          2 OUT_OF_ME+    0:125 
```

You can also check the STD error file with:

```bash
cat omp_mem_test.err 
slurmstepd: error: Detected 1 oom_kill event in StepId=6959634.0. Some of the step tasks have been OOM Killed.
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
Submitted batch job 6960135
```

>**NOTE:** This time the job should complete successfully.

## Step 6: Check the Job Status and Memory Efficiency

First, check the job status, e.g.,
```bash
sacct -j 6960135
JobID           JobName  Partition    Account  AllocCPUS      State ExitCode 
------------ ---------- ---------- ---------- ---------- ---------- -------- 
6960135      omp_mem_t+       test   rc_admin          2  COMPLETED      0:0 
6960135.bat+      batch              rc_admin          2  COMPLETED      0:0 
6960135.ext+     extern              rc_admin          2  COMPLETED      0:0 
6960135.0    omp_mem_t+              rc_admin          2  COMPLETED      0:0 
```
You can also check the STD output, e.g.,

```bash
cat omp_mem_test.out 
Matrix created successfully with 2 threads (n=60000)!
```

Second, check the memory efficiency with the `seff` command:

```bash
seff 6960135
Job ID: 6960135
Cluster: odyssey
User/Group: pkrastev/rc_admin
State: COMPLETED (exit code 0)
Nodes: 1
Cores per node: 2
CPU Utilized: 00:02:10
CPU Efficiency: 95.59% of 00:02:16 core-walltime
Job Wall-clock time: 00:01:08
Memory Utilized: 25.27 GB
Memory Efficiency: 63.17% of 40.00 GB (20.00 GB/core)
```

The Memory Efficiency is about 63%. The job used 25.27 GB while the requested memory 
is 40 GB. Please, notice that the requested memory is per core (20.00 GB/core). Adjust the requested memory so that the efficiency
is at least 80%, and resubmit the job, e.g.,

```bash
#SBATCH --mem=14G
```

Submit the job with the updated batch-job submission script,

```bash
sbatch run.sbatch
Submitted batch job 6960985
```

check the job status,

```bash
sacct -j 6960985
JobID           JobName  Partition    Account  AllocCPUS      State ExitCode 
------------ ---------- ---------- ---------- ---------- ---------- -------- 
6960985      omp_mem_t+       test   rc_admin          2  COMPLETED      0:0 
6960985.bat+      batch              rc_admin          2  COMPLETED      0:0 
6960985.ext+     extern              rc_admin          2  COMPLETED      0:0 
6960985.0    omp_mem_t+              rc_admin          2  COMPLETED      0:0
```

and memory efficiency:

```bash
seff 6960985
Job ID: 6960985
Cluster: odyssey
User/Group: pkrastev/rc_admin
State: COMPLETED (exit code 0)
Nodes: 1
Cores per node: 2
CPU Utilized: 00:02:07
CPU Efficiency: 94.78% of 00:02:14 core-walltime
Job Wall-clock time: 00:01:07
Memory Utilized: 25.60 GB
Memory Efficiency: 91.42% of 28.00 GB (14.00 GB/core)
```

We see that the Memory Efficiency is 91.42%, while the CPU Efficiency is 94.78%.
