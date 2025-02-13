

# Exercise 1: Job Efficiency - Memory per Node ( `--mem` )

We use a Fortran code, `mem_test.f90`, to generate a random matrix of dimension 60,000. 
Using `double precision`, the program needs ~28.8 GB of memory (since, 
60000 x 60000 x 8 bytes = 28,800,000,000 bytes)  to execute successfully. 

## Step 1: Compile Fortran source code
We compile the code with:

```bash
module load gcc/14.2.0-fasrc01
gfortran -o mem_test.x mem_test.f90 -O2
```

This will generate the executable `mem_test.x`. The Fortran source code is included 
below:

```fortran
!=====================================================================
! Program: mem_test.f90
!          Program generates a symetric random matrix of dimension 60K
!=====================================================================
program mem_test
  implicit none
  integer(4) :: n = 60000 ! Matrix dimension
  integer(4) :: i
  integer(4) :: j
  integer(4) :: iseed
  real(8), allocatable :: h(:,:)
  
  ! Random number generator to compute a random test-matrix
  real(8), external :: ran3
  
  iseed = -99

  ! Allocate memory ..................................................
  if ( .not. allocated (h) ) allocate ( h(n,n) )

  ! Create random test matrix h.......................................
  do i = 1, n
     do j = 1, i
        h(i,j) = ran3(iseed)
        h(j,i) = h(i,j)
     end do
  end do
  write(6,*) 'Hamiltonian matrix created successfully!'

  ! Free memory ......................................................
  if ( allocated( h ) ) deallocate( h )

end program mem_test

!=====================================================================
!     The function
!        ran3
!     returns a uniform random number deviate between 0.0 and 1.0. Set
!     the idum to any negative value to initialize or reinitialize the
!     sequence. Any large MBIG, and any small (but still large) MSEED
!     can be substituted for the present values.
!=====================================================================
REAL(8) FUNCTION ran3(idum)
  IMPLICIT NONE
  INTEGER :: idum
  INTEGER :: mbig,mseed,mz
  REAL(8) ::  fac
  PARAMETER (mbig=1000000000,mseed=161803398,mz=0,fac=1./mbig)
  INTEGER :: i,iff,ii,inext,inextp,k
  INTEGER :: mj,mk,ma(55)
  SAVE iff,inext,inextp,ma
  DATA iff /0/

  IF ( (idum < 0) .or. (iff == 0) ) THEN
     iff=1
     mj=mseed-IABS(idum)
     mj=MOD(mj,mbig)
     ma(55)=mj
     mk=1
     DO i=1,54
        ii=MOD(21*i,55)
        ma(ii)=mk
        mk=mj-mk
        IF(mk < mz)mk=mk+mbig
        mj=ma(ii)
     ENDDO
     DO k=1,4
        DO i=1,55
           ma(i)=ma(i)-ma(1+MOD(i+30,55))
           IF (ma(i) < mz)ma(i)=ma(i)+mbig
        ENDDO
     ENDDO
     inext=0
     inextp=31
     idum=1
  ENDIF
  inext=inext+1
  IF (inext == 56) inext=1
  inextp=inextp+1
  IF (inextp == 56) inextp=1
  mj=ma(inext)-ma(inextp)
  IF (mj < mz) mj=mj+mbig
  ma(inext)=mj
  ran3=mj*fac
  return
END FUNCTION ran3
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
sacct -j 3249906
JobID           JobName  Partition    Account  AllocCPUS      State ExitCode 
------------ ---------- ---------- ---------- ---------- ---------- -------- 
3249906        mem_test       test   rc_admin          1 OUT_OF_ME+    0:125 
3249906.bat+      batch              rc_admin          1 OUT_OF_ME+    0:125 
3249906.ext+     extern              rc_admin          1  COMPLETED      0:0
```

You can also check the STD error file with:

```bash
cat mem_test.err 
/var/slurmd/spool/slurmd/job3249906/slurm_script: line 15: 1407730 Killed                  ./mem_test.x
slurmstepd: error: Detected 1 oom_kill event in StepId=3249906.batch. Some of the step tasks have been OOM Killed.
```

## Step 5: Adjust the Memory Request and Resubmit the Job

Modify the job-submission script to request more memory, e.g., double the memory,

```bash
#SBATCH --mem=40G  # Double the original memory request 
```

and resubmit the job:

```bash
sbatch run.sbatch
Submitted batch job 3253896
```

>**NOTE:** This time the job should complete successfully.

## Step 6: Check the Job Status and Memory Efficiency

First, check the job status, e.g.,
```bash
sacct -j 3253896
JobID           JobName  Partition    Account  AllocCPUS      State ExitCode 
------------ ---------- ---------- ---------- ---------- ---------- -------- 
3253896        mem_test       test   rc_admin          1  COMPLETED      0:0 
3253896.bat+      batch              rc_admin          1  COMPLETED      0:0 
3253896.ext+     extern              rc_admin          1  COMPLETED      0:0
```
You can also check the STD output, e.g.,

```bash
cat mem_test.out 
 Hamiltonian matrix created successfully!
```

Second, check the memory efficiency with the `seff` command:

```bash
seff 3253896
Job ID: 3253896
Cluster: odyssey
User/Group: pkrastev/rc_admin
State: COMPLETED (exit code 0)
Cores: 1
CPU Utilized: 00:00:38
CPU Efficiency: 95.00% of 00:00:40 core-walltime
Job Wall-clock time: 00:00:40
Memory Utilized: 23.83 GB
Memory Efficiency: 59.57% of 40.00 GB (40.00 GB/node)
```

The Memory Efficiency is about 60%. The job used 23.83 GB while the requested memory 
is 40 GB. Adjust the requested memory so that the efficiency
is at least 80%, and resubmit the job, e.g.,

```bash
#SBATCH --mem=28G
```

Submit the job with the updated batch-job submission script,

```bash
sbatch run.sbatch
Submitted batch job 3362881
```

check the job status,

```bash
sacct -j 3362881
JobID           JobName  Partition    Account  AllocCPUS      State ExitCode 
------------ ---------- ---------- ---------- ---------- ---------- -------- 
3362881        mem_test       test   rc_admin          1  COMPLETED      0:0 
3362881.bat+      batch              rc_admin          1  COMPLETED      0:0 
3362881.ext+     extern              rc_admin          1  COMPLETED      0:0 
```

and memory efficiency:

```bash
seff 3362881
Job ID: 3362881
Cluster: odyssey
User/Group: pkrastev/rc_admin
State: COMPLETED (exit code 0)
Cores: 1
CPU Utilized: 00:00:38
CPU Efficiency: 97.44% of 00:00:39 core-walltime
Job Wall-clock time: 00:00:39
Memory Utilized: 23.51 GB
Memory Efficiency: 83.96% of 28.00 GB (28.00 GB/node)
```

We see that the Memory Efficiency is ~84%, while the CPU Efficiency is ~97%.