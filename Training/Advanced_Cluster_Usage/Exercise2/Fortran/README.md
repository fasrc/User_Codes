## Exercise 2: Job Efficiency - Memory per CPU/core ( `--mem-per-cpu` )


We use a program that performs
[Lanczos diagonalization](https://en.wikipedia.org/wiki/Lanczos_algorithm)
with reorthogonalization of a 70,000 x 70,000 random matrix.
The program writes and reads Lanczos vectors to disk at each iteration.
In this specific example, we calculate the first 5 eigenvalues and perform 50 iterations.
The example is intended to explore the memory efficiency of threaded parallel applications.

### Step 1: Compile Fortran source code

We compile the code with:

```bash
module load intel/24.2.1-fasrc01
make
```

This will generate the executable `planczos.x`. The `Makefile` used to build the code is
included below:

```make
#==========================================================
# Make file
#==========================================================
F90CFLAGS   = -c -O2 -qopenmp
F90COMPILER = ifx
PRO         = planczos
OBJECTS     = planczos2.o external_libs.o

# Default target
all: ${PRO}.x

${PRO}.x : $(OBJECTS)
	$(F90COMPILER) -o ${PRO}.x $(OBJECTS) -qopenmp

%.o : %.f90
	$(F90COMPILER) $(F90CFLAGS) $<

.PHONY: clean
clean:
	rm -f *.o *.x *.mod *.lvec *.out *.err
```

### Step 2: Create a job submission script

The below job-submission script intentionally requests less memory than what the job 
actually needs:

```bash
#!/bin/bash
#SBATCH -J planczos
#SBATCH -o planczos.out
#SBATCH -e planczos.err
#SBATCH -p test
#SBATCH -t 30
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem-per-cpu=4G

# Load required modules
module load intel/24.2.1-fasrc01

# Run program
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun -c $SLURM_CPUS_PER_TASK ./planczos.x
```
>**NOTE:** Notice that the memory is
requested per thread / core with `--mem-per-cpu` instead of with `--mem`.

### Step 3: Submit the Job

If the job-submission script is named `run.sbatch`, for instance, the job is submitted
to the queue with:

```bash
sbatch run.sbatch
```
>**NOTE:** The job should fail due to insufficient memory. 

### Step 4: Diagnose the Issue

You can check the job status with:

```bash
sacct -j 6231838
JobID           JobName  Partition    Account  AllocCPUS      State ExitCode 
------------ ---------- ---------- ---------- ---------- ---------- -------- 
6231838        planczos       test   rc_admin          8     FAILED      1:0 
6231838.bat+      batch              rc_admin          8     FAILED      1:0 
6231838.ext+     extern              rc_admin          8  COMPLETED      0:0 
6231838.0    planczos.x              rc_admin          8 OUT_OF_ME+    0:125 
```

### Step 5: Adjust the Memory Request and Resubmit the Job

Modify the job-submission script to request more memory, e.g., double the memory,

```bash
#SBATCH --mem-per-cpu=8G  # Double the original memory request 
```

and resubmit the job:

```bash
sbatch run.sbatch
Submitted batch job 6232860
```

>**NOTE:** This time the job should complete successfully.

## Step 6: Check the Job Status and Memory Efficiency

First, check the job status, e.g.,
```bash
sacct -j 6232860
JobID           JobName  Partition    Account  AllocCPUS      State ExitCode 
------------ ---------- ---------- ---------- ---------- ---------- -------- 
6232860        planczos       test   rc_admin          8  COMPLETED      0:0 
6232860.bat+      batch              rc_admin          8  COMPLETED      0:0 
6232860.ext+     extern              rc_admin          8  COMPLETED      0:0 
6232860.0    planczos.x              rc_admin          8  COMPLETED      0:0 
```
You can also check the STD output, e.g.,

```bash
cat planczos.out
           5  lowest eigenvalues - Lanczos
 iteration:           1
           1   34999.7013987699
           2  0.000000000000000E+000
           3  0.000000000000000E+000
           4  0.000000000000000E+000
           5  0.000000000000000E+000
           5  lowest eigenvalues - Lanczos
 iteration:           2
           1   34999.8688219519
           2  0.000000000000000E+000
           3  0.000000000000000E+000
           4  0.000000000000000E+000
           5  0.000000000000000E+000
           5  lowest eigenvalues - Lanczos
 iteration:           3
           1   34999.8688227451
           2   75.6872979509935
           3  0.000000000000000E+000
           4  0.000000000000000E+000
           5  0.000000000000000E+000
           5  lowest eigenvalues - Lanczos
           ...     ...     ...
 iteration:          50
           1   34999.8688227451
           2   152.659126631821
           3   151.733280206418
           4   150.139760676962
           5   148.032825045063
 Lanczos iterations finished...
```
Second, check the memory efficiency with the `seff` command:

```bash
seff 6232860
Job ID: 6232860
Cluster: odyssey
User/Group: pkrastev/rc_admin
State: COMPLETED (exit code 0)
Nodes: 1
Cores per node: 8
CPU Utilized: 00:04:02
CPU Efficiency: 42.01% of 00:09:36 core-walltime
Job Wall-clock time: 00:01:12
Memory Utilized: 36.63 GB
Memory Efficiency: 57.24% of 64.00 GB (8.00 GB/core)
```

The Memory Efficiency is about 57%. The job used ~37 GB while the requested memory 
is 64.00 GB (8.00 GB/core). We need to further adjust the memory to achieve
memory efficiency of at least 80%.

### Step 7: Adjust the Memory Request, Resubmit the Job, and check the Job Status and Memory Efficiency

Modify the job-submission script to request more memory, e.g.,,

```bash
#SBATCH --mem-per-cpu=5G 
```

resubmit the job,

```bash
sbatch run.sbatch
```

and check the job status

```bash
sacct -j 6233716
JobID           JobName  Partition    Account  AllocCPUS      State ExitCode 
------------ ---------- ---------- ---------- ---------- ---------- -------- 
6233716        planczos       test   rc_admin          8  COMPLETED      0:0 
6233716.bat+      batch              rc_admin          8  COMPLETED      0:0 
6233716.ext+     extern              rc_admin          8  COMPLETED      0:0 
6233716.0    planczos.x              rc_admin          8  COMPLETED      0:0 
```

and efficiency:

```bash
seff 6233716
Job ID: 6233716
Cluster: odyssey
User/Group: pkrastev/rc_admin
State: COMPLETED (exit code 0)
Nodes: 1
Cores per node: 8
CPU Utilized: 00:04:04
CPU Efficiency: 41.22% of 00:09:52 core-walltime
Job Wall-clock time: 00:01:14
Memory Utilized: 36.63 GB
Memory Efficiency: 91.57% of 40.00 GB (5.00 GB/core)
```

In this case, the memory efficiency is above 80% (~92%).