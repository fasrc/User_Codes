# MPI Software on the FASRC cluster

## Introduction

This web-page is intended to help you compile and run MPI applications on the cluster cluster.

The Message Passing Interface (MPI) library allows processes in your parallel application to communicate with one another by sending and receiving messages. There is no default MPI library in your environment when you log in to the cluster. You need to choose the desired MPI implementation for your applications. This is done by loading an appropriate MPI module. Currently the available MPI implementations on our cluster are [OpenMPI](https://www.open-mpi.org/) and [Mpich](https://www.mpich.org/). For both implementations the MPI libraries are compiled and built with either the [Intel compiler suite](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html) or the [GNU compiler suite](https://www.gnu.org/software/gcc/). These are organized in [software modules](https://docs.rc.fas.harvard.edu/kb/modules-intro/).

For instance, if you want to use OpenMPI compiled with the GNU compiler you need to load appropriate compiler and MPI modules. Below are some possible combinations, check <code>module spider MODULENAME</code> to get a full listing of possibilities.

```bash
# GCC + OpenMPI, e.g.,
module load gcc/13.2.0-fasrc01 openmpi/4.1.5-fasrc03

# GCC + Mpich, e.g.,
module load gcc/13.2.0-fasrc01 mpich/4.1.2-fasrc01

# Intel + OpenMPI, e.g.,
module load intel/23.2.0-fasrc01 openmpi/4.1.5-fasrc03

# Intel + Mpich, e.g.,
module load intel/23.2.0-fasrc01 mpich/4.1.2-fasrc01

# Intel + IntelMPI (IntelMPI runs mpich underneath), e.g.
module load intel/23.2.0-fasrc01 intelmpi/2021.10.0-fasrc01
```

For reproducibility and consistency it is recommended to use the complete module name with the module load command, as illustrated above. Modules on the cluster get updated often so check if there are more recent ones. The modules are set up so that you can only have one MPI module loaded at a time. If you try loading a second one it will automatically unload the first. This is done to avoid dependencies collisions.

There are four ways you can set up your MPI on the cluster:

* Put the module load command in your startup files.<br>
Most users will find this option most convenient. You will likely only want to use a single version of MPI for all your work. This method also works with all MPI modules currently available on the cluster.

* Load the module in your current shell.<br>
For the current MPI versions you do not need to have the module load command in your startup files. If you submit a job the remote processes will inherit the submission shell environment and use the proper MPI library. Note this method does not work with older versions of MPI.

* Load the module in your job script.<br>
If you will be using different versions of MPI for different jobs, then you can put the module load command in your script. You need to ensure your script can execute the module load command properly.

* Do not use modules and set environment variables yourself. <br>
You obviously do not need to use modules but can hard code paths. However, these locations may change without warning so you should set them in one location only and not scatter them throughout your scripts. This option could be useful if you have a customized local build of MPI you would like to use with your applications.

## Your First MPI Program

The below examples are included in this repository.

**Fortran 77:** <code>mpitest.f</code>
```fortran
c=====================================================
c Fortran 77 MPI example: mpitest.f
c=====================================================
      program mpitest
      implicit none
      include 'mpif.h'
      integer(4) :: ierr
      integer(4) :: iproc
      integer(4) :: nproc
      integer(4) :: i
      call MPI_INIT(ierr)
      call MPI_COMM_SIZE(MPI_COMM_WORLD,nproc,ierr)
      call MPI_COMM_RANK(MPI_COMM_WORLD,iproc,ierr)
      do i = 0, nproc-1
         call MPI_BARRIER(MPI_COMM_WORLD,ierr)
         if ( iproc == i ) then
            write (6,*) 'Rank',iproc,'out of',nproc
         end if
      end do
      call MPI_FINALIZE(ierr)
      if ( iproc == 0 ) write(6,*)'End of program.'
      stop
      end
```

**Fortran 90:** <code>mpitest.f90</code>

```fortran
!=====================================================
! Fortran 90 MPI example: mpitest.f90
!=====================================================
program mpitest
  implicit none
  include 'mpif.h'
  integer(4) :: ierr
  integer(4) :: iproc
  integer(4) :: nproc
  integer(4) :: i
  call MPI_INIT(ierr)
  call MPI_COMM_SIZE(MPI_COMM_WORLD,nproc,ierr)
  call MPI_COMM_RANK(MPI_COMM_WORLD,iproc,ierr)
  do i = 0, nproc-1
     call MPI_BARRIER(MPI_COMM_WORLD,ierr)
     if ( iproc == i ) then
        write (6,*) 'Rank',iproc,'out of',nproc
     end if
  end do
  call MPI_FINALIZE(ierr)
  if ( iproc == 0 ) write(6,*)'End of program.'
  stop
end program mpitest
```
**C:** <code>mpitest.c</code>

```c
//==============================================================
// C MPI example: mpitest.c
//==============================================================
#include <stdio.h>
#include <mpi.h>
int main(int argc, char** argv){
  int iproc;
  int nproc;
  int i;
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&iproc);
  MPI_Comm_size(MPI_COMM_WORLD,&nproc);
  for ( i = 0; i <= nproc - 1; i++ ){
    MPI_Barrier(MPI_COMM_WORLD);
    if ( iproc == i ){
      printf("%s %d %s %d \n","Rank",iproc,"out of",nproc);
    }
  }
  MPI_Finalize();
  if ( iproc == 0) printf("End of program.\n");
  return 0;
}
```
**C++:** <code>mpitest.cpp</code>

```c++
//==============================================================
// C++ MPI example: mpitest.cpp
//==============================================================
#include <iostream>
#include <mpi.h>
using namespace std;
int main(int argc, char** argv){
  int iproc;
  int nproc;
  int i;
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&iproc);
  MPI_Comm_size(MPI_COMM_WORLD,&nproc);
  for ( i = 0; i <= nproc - 1; i++ ){
    MPI_Barrier(MPI_COMM_WORLD);
    if ( i == iproc ){
      cout << "Rank " << iproc << " out of " << nproc << endl;
    }
  }
  MPI_Finalize();
  if ( iproc == 0 ) cout << "End of program." << endl;
  return 0;
}
```

### Compile the program

Depending the language you used to implement your code (Fortran 77, Fortran 90, C, C++), and the MPI implementation (OpenMPI, Mpich, Intel-MPI) use one of the following:

```bash
# Fortran 77 with OpenMPI or Mpich
mpif77 -o mpitest.x mpitest.f

# Fortran 90 with OpenMPI or Mpich
mpif90 -o mpitest.x mpitest.f90

# Fortran 90 with Intel-MPI
mpiifx -o mpitest.x mpitest.f90

# C with OpenMPI or Mpich
mpicc -o mpitest.x mpitest.c

# C with Intel-MPI       
mpiicx -o mpitest.x mpitest.c

# C++ with OpenMPI or Mpich
mpicxx -o mpitest.x mpitest.cpp

# C++ with Intel-MPI
mpiicpx -o mpitest.x mpitest.cpp
```

### Create a batch-jobs submission script

The batch script is used to instruct the cluster to reserve computational resources for your job and how your application should be launched on the compute nodes reserved for the job.

With a text editor like emacs or vi open a new file named <code>run.sbatch</code> and paste in the contents of **one** of the examples below:

#### OpenMPI example

```bash
#!/bin/bash
#SBATCH -J mpitest            # job name
#SBATCH -o mpitest.out        # standard output file
#SBATCH -e mpitest.err        # standard error file
#SBATCH -p shared             # partition
#SBATCH -n 8                  # ntasks
#SBATCH -t 00:30:00           # time in HH:MM:SS
#SBATCH --mem-per-cpu=500     # memory in megabytes

# --- Load the required software modules., e.g., ---
module load gcc/13.2.0-fasrc01 openmpi/4.1.5-fasrc03

# --- Run the executable ---
srun -n $SLURM_NTASKS --mpi=pmix ./mpitest.x
```

> **NOTE:** Notice, in the above example we use GCC and OpenMPI, <code>module load gcc/13.2.0-fasrc01 openmpi/4.1.5-fasrc03</code>. As a rule, you **must** load exactly the same modules you used to compile your code.

#### Intel-MPI example

```bash
#!/bin/bash
#SBATCH -J mpitest            # job name
#SBATCH -o mpitest.out        # standard output file
#SBATCH -e mpitest.err        # standard error file
#SBATCH -p shared             # partition
#SBATCH -n 8                  # ntasks
#SBATCH -t 00:30:00           # time in HH:MM:SS
#SBATCH --mem-per-cpu=500     # memory in megabytes

# --- Load the required software modules., e.g., ---
module load intel/23.2.0-fasrc01 intelmpi/2021.10.0-fasrc01

# --- Run the executable ---
# with intelmpi, you need to ensure it uses pmi2 instead of pmix
srun -n $SLURM_NTASKS --mpi=pmi2 ./mpitest.x
```

> **NOTE:** Notice, in the above example we use Intel and IntelMPI, <code>module load intel/23.2.0-fasrc01 intelmpi/2021.10.0-fasrc01</code>. As a rule, you **must** load exactly the same modules you used to compile your code.


### Submit the jobs to the queue

The <code>sbatch</code> command followed the batch-job script name, e.g., <code>run.sbatch</code>, is used to submit your batch script to the cluster compute nodes. Upon submission a job ID is returned, such as:

```bash
sbatch run.sbatch
```

### Monitor your job

After you submit your job, the system scheduler will check to see if there are compute nodes available to run the job. If there are compute nodes available, your job will start running. If there are not, your job will wait in the queue until there are enough resources to run your application. You can monitor your position in the queue with the <code>sacct</code> command:

```bash
$ sacct
JobID           JobName  Partition    Account  AllocCPUS      State ExitCode 
------------ ---------- ---------- ---------- ---------- ---------- -------- 
50077403        mpitest      rocky   rc_admin          8  COMPLETED      0:0 
50077403.ba+      batch              rc_admin          4  COMPLETED      0:0 
50077403.ex+     extern              rc_admin          8  COMPLETED      0:0 
50077403.0    mpitest.x              rc_admin          8  COMPLETED      0:0 
```

### Examine your job's output

When your job has completed you should see a file called <code>mpitest.out</code>

```bash
$ cat mpitest.out 
 Rank           0 out of           8
 Rank           4 out of           8
 Rank           1 out of           8
 Rank           2 out of           8
 Rank           3 out of           8
 Rank           5 out of           8
 Rank           6 out of           8
 Rank           7 out of           8
 End of program.
```

## Examples

Below are some further MPI examples.

* **[Example 1](https://github.com/fasrc/User_Codes/tree/master/Parallel_Computing/MPI/Example1):** Monte-Carlo calculation of $\pi$
* **[Example 2](https://github.com/fasrc/User_Codes/tree/master/Parallel_Computing/MPI/Example2):** Integration of $x^2$ in interval [0, 4] with 80 integration points and the trapezoidal rule
* **[Example 3](https://github.com/fasrc/User_Codes/tree/master/Parallel_Computing/MPI/Example3):** Parallel Lanczos diagonalization with reorthogonalization and MPI I/O

## References

* [Good Parallel Computing/MPI tutorials](https://hpc.llnl.gov/documentation/tutorials)
* [RS/6000 SP: Practical MPI Programming](https://www.cs.kent.ac.uk/people/staff/trh/MPI/mpi_ibm.pdf)
* [Open MPI: Open Source High Performance Computing](https://www.open-mpi.org/)
* [MPICH implementation of MPI](https://www.mpich.org/)
