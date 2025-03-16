## Exercise 2: Job Efficiency - Memory per Task ( `--mem-per-cpu` )


We use a program that performs
[Lanczos diagonalization](https://en.wikipedia.org/wiki/Lanczos_algorithm)
with reorthogonalization of a 30,000 x 30,000 random matrix.
The program uses MPI-IO to write and read Lanczos vectors to disk at each iteration.
In this specific example, we calculate the first 5 eigenvalues and perform 50 iterations.
The example is intended to explore the memory efficiency of distributed parallel
applications.

### Step 1: Compile Fortran source code

We compile the code with:

```bash
module load intel/24.2.1-fasrc01 openmpi/5.0.5-fasrc01
make
```

This will generate the executable `planczos.x`. The `Makefile` used to build the code is
included below:

```make
#==========================================================
# Make file
#==========================================================
F90CFLAGS   = -c -O2
F90COMPILER = mpif90
PRO         = planczos
OBJECTS     = planczos2.o \
              external_libs.o

${PRO}.x : $(OBJECTS)
	$(F90COMPILER) -o ${PRO}.x $(OBJECTS)

%.o : %.f90
	$(F90COMPILER) $(F90CFLAGS) $(<F)

clean : 
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
#SBATCH -n 8
#SBATCH --mem-per-cpu=4G

# Load required modules
module load intel/24.2.1-fasrc01 openmpi/5.0.5-fasrc01

# Run program
srun -n $SLURM_NTASKS --mpi=pmix ./planczos.x
```
>**NOTE:** Since this is a distributed parallel job, notice that the memory is
requested per MPI task (core) with `--mem-per-cpu` instead of with `--mem`.

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
sacct -j 3881361
JobID           JobName  Partition    Account  AllocCPUS      State ExitCode 
------------ ---------- ---------- ---------- ---------- ---------- -------- 
3881361        planczos       test   rc_admin          8     FAILED      1:0 
3881361.bat+      batch              rc_admin          8     FAILED      1:0 
3881361.ext+     extern              rc_admin          8  COMPLETED      0:0 
3881361.0    planczos.x              rc_admin          8 OUT_OF_ME+    0:125 
```

### Step 5: Adjust the Memory Request and Resubmit the Job

Modify the job-submission script to request more memory, e.g., double the memory,

```bash
#SBATCH --mem-per-cpu=8G  # Double the original memory request 
```

and resubmit the job:

```bash
sbatch run.sbatch
Submitted batch job 3881384
```

>**NOTE:** This time the job should complete successfully.

## Step 6: Check the Job Status and Memory Efficiency

First, check the job status, e.g.,
```bash
sacct -j 3881384
JobID           JobName  Partition    Account  AllocCPUS      State ExitCode 
------------ ---------- ---------- ---------- ---------- ---------- -------- 
3881384        planczos       test   rc_admin          8  COMPLETED      0:0 
3881384.bat+      batch              rc_admin          8  COMPLETED      0:0 
3881384.ext+     extern              rc_admin          8  COMPLETED      0:0 
3881384.0    planczos.x              rc_admin          8  COMPLETED      0:0
```
You can also check the STD output, e.g.,

```bash
cat planczos.out
           5  lowest eigenvalues - Lanczos
 iteration:           1
           1   14999.3790246901     
           2  0.000000000000000E+000
           3  0.000000000000000E+000
           4  0.000000000000000E+000
           5  0.000000000000000E+000
           5  lowest eigenvalues - Lanczos
 iteration:           2
           1   14999.5453250898     
           2  0.000000000000000E+000
           3  0.000000000000000E+000
           4  0.000000000000000E+000
           5  0.000000000000000E+000
           5  lowest eigenvalues - Lanczos
 iteration:           3
           1   14999.5453269108     
           2   49.9213290567113     
           3  0.000000000000000E+000
           4  0.000000000000000E+000
           5  0.000000000000000E+000
           5  lowest eigenvalues - Lanczos
           ...     ...     ...
 iteration:          50
           1   14999.5453269108     
           2   99.6649122506606     
           3   99.2216836112309     
           4   98.2094189472823     
           5   96.8463473936131     
 Lanczos iterations finished...
```
Second, check the memory efficiency with the `seff` command:

```bash
seff 3881384
Job ID: 3881384
Cluster: odyssey
User/Group: pkrastev/rc_admin
State: COMPLETED (exit code 0)
Nodes: 1
Cores per node: 8
CPU Utilized: 00:06:16
CPU Efficiency: 87.04% of 00:07:12 core-walltime
Job Wall-clock time: 00:00:54
Memory Utilized: 54.84 GB
Memory Efficiency: 85.68% of 64.00 GB (8.00 GB/core)
The task which had the largest memory consumption differs by 100.02% from the average task max memory consumption
```
The Memory Efficiency is about 86%. The job used 54.84 GB while the requested memory 
is 64.00 GB (8.00 GB/core). Since the memory efficiency is above 80%, the job is overall
memory efficient.