# Exercise 1: Job Efficiency - Memory per Node ( `--mem` )

We use a C code, `mem_test.c`, to generate a random matrix of dimension 60,000. 
Using `double precision`, the program needs ~28.8 GB of memory (since, 
60000 x 60000 x 8 bytes = 28,800,000,000 bytes)  to execute successfully. 

## Step 1: Compile C source code
We compile the code with:

```bash
module load gcc/14.2.0-fasrc01
gcc -o mem_test.x mem_test.c -O2
```

This will generate the executable `mem_test.x`. The C source code is included 
below:

```c
#include <stdio.h>
#include <stdlib.h>

// Function prototype for the random number generator
double ran3(int *idum);

int main() {
    int n = 60000; // Matrix dimension
    int i, j;
    int iseed = -99;
    double **h;

    // Allocate memory for the matrix
    h = (double **)malloc(n * sizeof(double *));
    for (i = 0; i < n; i++) {
        h[i] = (double *)malloc(n * sizeof(double));
    }

    // Create random symmetric test matrix
    for (i = 0; i < n; i++) {
        for (j = 0; j <= i; j++) {
            h[i][j] = ran3(&iseed);
            h[j][i] = h[i][j]; // Symmetric matrix
        }
    }
    printf("Hamiltonian matrix created successfully!\n");

    // Free memory
    for (i = 0; i < n; i++) {
        free(h[i]);
    }
    free(h);

    return 0;
}

// Random number generator (ran3)
double ran3(int *idum) {
    static int iff = 0;
    static int inext, inextp;
    static long ma[55];
    static int mbig = 1000000000, mseed = 161803398, mz = 0;
    static double fac = 1.0 / 1000000000.0;
    long mj, mk;
    int i, ii, k;

    if (*idum < 0 || iff == 0) {
        iff = 1;
        mj = mseed - (*idum < 0 ? -*idum : *idum);
        mj %= mbig;
        ma[54] = mj;
        mk = 1;
        for (i = 1; i <= 54; i++) {
            ii = (21 * i) % 55;
            ma[ii - 1] = mk;
            mk = mj - mk;
            if (mk < mz) mk += mbig;
            mj = ma[ii - 1];
        }
        for (k = 1; k <= 4; k++) {
            for (i = 0; i < 55; i++) {
                ma[i] -= ma[(i + 30) % 55];
                if (ma[i] < mz) ma[i] += mbig;
            }
        }
        inext = 0;
        inextp = 31;
        *idum = 1;
    }

    if (++inext == 55) inext = 0;
    if (++inextp == 55) inextp = 0;
    mj = ma[inext] - ma[inextp];
    if (mj < mz) mj += mbig;
    ma[inext] = mj;
    return mj * fac;
}
```

## Step 2: Create a job submission  script

The below job-submission script intentionally requests less memory than what the job
actually needs:

```bash
#!/bin/bash
#SBATCH -p test
#SBATCH -J mem_test
#SBATCH -o mem_test.out
#SBATCH -e mem_test.err
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 30
#SBATCH --mem=20G # Requests intentionally less memory than what the program needs

# Load required modules
module load gcc/14.2.0-fasrc01

# Run the code
./mem_test.x
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
sacct -j 3764934
JobID           JobName  Partition    Account  AllocCPUS      State ExitCode 
------------ ---------- ---------- ---------- ---------- ---------- -------- 
3764934        mem_test       test   rc_admin          1 OUT_OF_ME+    0:125 
3764934.bat+      batch              rc_admin          1 OUT_OF_ME+    0:125 
3764934.ext+     extern              rc_admin          1  COMPLETED      0:0
```

You can also check the STD error file with:

```bash
cat mem_test.err 
/var/slurmd/spool/slurmd/job3764934/slurm_script: line 15: 2196449 Killed                  ./mem_test.x
slurmstepd: error: Detected 1 oom_kill event in StepId=3764934.batch. Some of the step tasks have been OOM Killed.
```

## Step 5: Adjust the Memory Request and Resubmit the Job

Modify the job-submission script to request more memory, e.g., double the memory,

```bash
#SBATCH --mem=40G  # Double the original memory request 
```

and resubmit the job:

```bash
sbatch run.sbatch 
Submitted batch job 3765959
```

>**NOTE:** This time the job should complete successfully.

## Step 6: Check the Job Status and Memory Efficiency

First, check the job status, e.g.,
```bash
sacct -j 3765959
JobID           JobName  Partition    Account  AllocCPUS      State ExitCode 
------------ ---------- ---------- ---------- ---------- ---------- -------- 
3765959        mem_test       test   rc_admin          1  COMPLETED      0:0 
3765959.bat+      batch              rc_admin          1  COMPLETED      0:0 
3765959.ext+     extern              rc_admin          1  COMPLETED      0:0 
```
You can also check the STD output, e.g.,

```bash
cat mem_test.out 
Hamiltonian matrix created successfully!
```

Second, check the memory efficiency with the `seff` command:

```bash
seff 3765959
Job ID: 3765959
Cluster: odyssey
User/Group: pkrastev/rc_admin
State: COMPLETED (exit code 0)
Cores: 1
CPU Utilized: 00:01:08
CPU Efficiency: 97.14% of 00:01:10 core-walltime
Job Wall-clock time: 00:01:10
Memory Utilized: 24.04 GB
Memory Efficiency: 60.10% of 40.00 GB (40.00 GB/node)
```

The Memory Efficiency is about 60%. The job used 24.04 GB while the requested memory 
is 40 GB. Adjust the requested memory so that the efficiency
is at least 80%, and resubmit the job, e.g.,

```bash
#SBATCH --mem=28G
```

Submit the job with the updated batch-job submission script,

```bash
sbatch run.sbatch
Submitted batch job 3766476
```

check the job status,

```bash
sacct -j 3766476
JobID           JobName  Partition    Account  AllocCPUS      State ExitCode 
------------ ---------- ---------- ---------- ---------- ---------- -------- 
3766476        mem_test       test   rc_admin          1  COMPLETED      0:0 
3766476.bat+      batch              rc_admin          1  COMPLETED      0:0 
3766476.ext+     extern              rc_admin          1  COMPLETED      0:0 
```

and memory efficiency:

```bash
seff 3766476
Job ID: 3766476
Cluster: odyssey
User/Group: pkrastev/rc_admin
State: COMPLETED (exit code 0)
Cores: 1
CPU Utilized: 00:01:08
CPU Efficiency: 100.00% of 00:01:08 core-walltime
Job Wall-clock time: 00:01:08
Memory Utilized: 24.46 GB
Memory Efficiency: 87.36% of 28.00 GB (28.00 GB/node)
```

We see that the Memory Efficiency is ~87%, while the CPU Efficiency is 100%.
