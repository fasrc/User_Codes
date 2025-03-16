# Exercise 1: Job Efficiency - Memory per CPU/core ( `--mem-per-cpu` )

We use a Fortran code, `omp_mem_test.f90`, to generate a random matrix of dimension 60,000. Using `double precision`, the program needs ~28.8 GB of memory (since, 
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
COMPILER = gfortran
PRO         = omp_mem_test
OBJECTS     = omp_mem_test.o

# Default target
all: ${PRO}.x

${PRO}.x : $(OBJECTS)
	$(COMPILER) -o ${PRO}.x $(OBJECTS) -fopenmp

%.o : %.f90
	$(COMPILER) $(CFLAGS) $<

.PHONY: clean
clean:
	rm -f *.o *.x *.out *.err
```

This will generate the executable `omp_mem_test.x`. The C source code is included 
below:

```fortran
!=====================================================================
! Program: omp_mem_test.f90
!          Program generates a symmetric random matrix of dimension 60K
!=====================================================================
program mem_test
  use omp_lib       ! Import OpenMP library for thread functions
  use rng_module    ! Import RNGState type
  implicit none
  integer(4) :: n = 60000 ! Matrix dimension
  integer(4) :: i, j
  integer(4) :: iseed
  type(RNGState) :: state
  real(8), allocatable :: h(:,:)

  ! Random number generator function
  real(8), external :: ran3

  ! Allocate memory
  if (.not. allocated(h)) allocate(h(n,n))

  ! Create random test matrix with OpenMP
  !$omp parallel private(i, j, iseed, state)
    iseed = -(99 + omp_get_thread_num()) ! Unique seed per thread
    state%ma = 0
    state%inext = 0
    state%inextp = 0
    state%iff = 0

    !$omp do schedule(dynamic)
    do i = 1, n
       do j = 1, i
          h(i,j) = ran3(iseed, state)
          h(j,i) = h(i,j)
       end do
    end do
    !$omp end do
  !$omp end parallel

  write(6,*) 'Hamiltonian matrix created successfully with ', &
             omp_get_max_threads(), ' threads (n=', n, ')!'

  ! Free memory
  if (allocated(h)) deallocate(h)

end program mem_test

!=====================================================================
!     The function
!        ran3
!     returns a uniform random number deviate between 0.0 and 1.0. Set
!     the idum to any negative value to initialize or reinitialize the
!     sequence. Thread-safe version with state passed as argument.
!=====================================================================
real(8) function ran3(idum, state)
  use rng_module    ! Import RNGState type
  implicit none
  integer(4), intent(inout) :: idum
  type(RNGState), intent(inout) :: state

  integer(4), parameter :: mbig = 1000000000, mseed = 161803398, mz = 0
  real(8), parameter :: fac = 1.0d0 / mbig
  integer(4) :: i, ii, k
  integer(4) :: mj, mk

  if ((idum < 0) .or. (state%iff == 0)) then
     state%iff = 1
     mj = mseed - iabs(idum)
     mj = mod(mj, mbig)
     state%ma(55) = mj
     mk = 1
     do i = 1, 54
        ii = mod(21 * i, 55)
        state%ma(ii) = mk
        mk = mj - mk
        if (mk < mz) mk = mk + mbig
        mj = state%ma(ii)
     end do
     do k = 1, 4
        do i = 1, 55
           state%ma(i) = state%ma(i) - state%ma(1 + mod(i + 30, 55))
           if (state%ma(i) < mz) state%ma(i) = state%ma(i) + mbig
        end do
     end do
     state%inext = 0
     state%inextp = 31
     idum = 1
  end if

  state%inext = state%inext + 1
  if (state%inext == 56) state%inext = 1
  state%inextp = state%inextp + 1
  if (state%inextp == 56) state%inextp = 1
  mj = state%ma(state%inext) - state%ma(state%inextp)
  if (mj < mz) mj = mj + mbig
  state%ma(state%inext) = mj
  ran3 = mj * fac
end function ran3
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
sacct -j 6963943
JobID           JobName  Partition    Account  AllocCPUS      State ExitCode 
------------ ---------- ---------- ---------- ---------- ---------- -------- 
6963943      omp_mem_t+       test   rc_admin          2     FAILED      1:0 
6963943.bat+      batch              rc_admin          2     FAILED      1:0 
6963943.ext+     extern              rc_admin          2  COMPLETED      0:0 
6963943.0    omp_mem_t+              rc_admin          2 OUT_OF_ME+    0:125 
```

You can also check the STD error file with:

```bash
cat omp_mem_test.err
slurmstepd: error: Detected 1 oom_kill event in StepId=6963943.0. Some of the step tasks have been OOM Killed.
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
Submitted batch job 6964594
```

>**NOTE:** This time the job should complete successfully.

## Step 6: Check the Job Status and Memory Efficiency

First, check the job status, e.g.,
```bash
sacct -j 6964594
JobID           JobName  Partition    Account  AllocCPUS      State ExitCode 
------------ ---------- ---------- ---------- ---------- ---------- -------- 
6964594      omp_mem_t+       test   rc_admin          2  COMPLETED      0:0 
6964594.bat+      batch              rc_admin          2  COMPLETED      0:0 
6964594.ext+     extern              rc_admin          2  COMPLETED      0:0 
6964594.0    omp_mem_t+              rc_admin          2  COMPLETED      0:0 
```
You can also check the STD output, e.g.,

```bash
cat omp_mem_test.out 
 Hamiltonian matrix created successfully with            2  threads (n=       60000 )!
```

Second, check the memory efficiency with the `seff` command:

```bash
seff 6964594
Job ID: 6964594
Cluster: odyssey
User/Group: pkrastev/rc_admin
State: COMPLETED (exit code 0)
Nodes: 1
Cores per node: 2
CPU Utilized: 00:01:57
CPU Efficiency: 90.00% of 00:02:10 core-walltime
Job Wall-clock time: 00:01:05
Memory Utilized: 20.16 GB
Memory Efficiency: 50.39% of 40.00 GB (20.00 GB/core)
```

The Memory Efficiency is about 50%. The job used 20.16 GB while the requested memory 
is 40 GB. Please, notice that the requested memory is per core (20.00 GB/core). 

Adjust the requested memory so that the efficiency is at least 80%, and resubmit the job, e.g.,

```bash
#SBATCH --mem=14G
```

Submit the job with the updated batch-job submission script,

```bash
sbatch run.sbatch
Submitted batch job 6965077
```

check the job status,

```bash
[pkrastev@holy8a24301 Fortran]$ sacct -j 6965077
JobID           JobName  Partition    Account  AllocCPUS      State ExitCode 
------------ ---------- ---------- ---------- ---------- ---------- -------- 
6965077      omp_mem_t+       test   rc_admin          2  COMPLETED      0:0 
6965077.bat+      batch              rc_admin          2  COMPLETED      0:0 
6965077.ext+     extern              rc_admin          2  COMPLETED      0:0 
6965077.0    omp_mem_t+              rc_admin          2  COMPLETED      0:0 
```

and memory efficiency:

```bash
seff 6965077
Job ID: 6965077
Cluster: odyssey
User/Group: pkrastev/rc_admin
State: COMPLETED (exit code 0)
Nodes: 1
Cores per node: 2
CPU Utilized: 00:01:16
CPU Efficiency: 92.68% of 00:01:22 core-walltime
Job Wall-clock time: 00:00:41
Memory Utilized: 21.92 GB
Memory Efficiency: 78.27% of 28.00 GB (14.00 GB/core)
```

We see that the Memory Efficiency is 78.27%, while the CPU Efficiency is 92.68%.
