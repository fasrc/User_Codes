### Purpose:

Program calculates PI via parallel Monte-Carlo algorithm.

### Contents:

* <code>pi\_monte\_carlo.f90</code>: Fortran source code
* <code>Makefile</code>: Makefile to compile the source code
* <code>run.sbatch</code>: Btach-job submission script to send the job to the queue.


### Source Code:

```fortran
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
! Program: pi_monte_carlo.f90
!          Calculate PI via parallel Monte-Carlo algorithm
!
! Compile: mpif90 -o pi_monte_carlo.x pi_monte_carlo.f90 -O2
! 
! Run:     mpirun -np <Number_of_MPI_ranks> ./pi_monte_carlo.x
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
program pi_monte_carlo
  use ifport
  implicit none
  include 'mpif.h'
  integer(4) :: iseed
  integer(4) :: icomm
  integer(4) :: iproc
  integer(4) :: nproc
  integer(4) :: ierr
  integer(8) :: iproc8
  integer(8) :: nproc8
  integer(8) :: i
  integer(8) :: n
  integer(8) :: n_tot
  integer(8) :: sample_number
  real(8)    :: r
  real(8)    :: x
  real(8)    :: y
  real(8)    :: pi_comp
  real(8)    :: pi_err
  real(8)    :: t1
  real(8)    :: t2
  real(8)    :: t_tot
  real(8)    :: randval

  real(8), parameter :: pi=3.141592653589793238462643d0

  ! Initialize MPI....................................................
  call MPI_INIT(ierr)
  icomm = MPI_COMM_WORLD
  call MPI_COMM_SIZE(icomm,nproc,ierr)
  call MPI_COMM_RANK(icomm,iproc,ierr)

  t1 = MPI_WTIME(ierr)

  iseed = -99            ! Seed for random number generator
  r = 1.0d0              ! Unit circle
  sample_number = 1e10   ! Number of samples
  randval =  rand(iseed) ! Iinitialize the random number generator 
 
  ! Convert to INTEGER8...............................................
  iproc8 = iproc
  nproc8 = nproc

  ! Parallel Monte-Carlo sampling.....................................
  n = 0
  do i = 1+iproc8, sample_number, nproc8 
     x = r * rand()
     y = r * rand()
     if ( x**2 + y**2 <= r**2 ) then
        n = n + 1
     end if
  end do

  ! Get total number of hits..........................................
  call MPI_REDUCE(n, n_tot, 1, MPI_INTEGER8, MPI_SUM, 0, icomm, ierr)

  ! Calculate approximated PI.........................................
  pi_comp = 4.0d0 * n_tot / sample_number

  ! Error.............................................................
  pi_err = ( ( dabs(pi-pi_comp) ) / pi ) * 100.0d0

  t2 = MPI_WTIME(ierr)

  t_tot = t2 - t1

  ! Print out result..................................................
  if ( iproc == 0 ) then
     write(6,'(1x,a,4x,f10.8)') 'Exact PI:', pi
     write(6,'(1x,a,1x,f10.8)') 'Computed PI:', pi_comp
     write(6,'(1x,a,7x,f7.5,a)') 'Error:', pi_err, '%'
     write(6,'(1x,a,1x,f5.2,1x,a)') 'Total time:', t_tot, 'sec'
  end if

  ! Shut down MPI.....................................................
  call MPI_FINALIZE(ierr)

  stop

end program pi_monte_carlo
```
**NOTE:** This code uses the [RAND, RANDOM](https://www.intel.com/content/www/us/en/develop/documentation/fortran-compiler-oneapi-dev-guide-and-reference/top/language-reference/a-to-z-reference/q-to-r/rand-random.html) portability functions from Intel to generate random numbers.

### Example Batch-Job Submission Script:

```bash
#!/bin/bash
#SBATCH -J pi_monte_carlo
#SBATCH -o pi_monte_carlo.out
#SBATCH -e pi_monte_carlo.err
#SBATCH -p test
#SBATCH -t 30
#SBATCH -n 8
#SBATCH --mem-per-cpu=4000

# Load required modules
module load intel/21.2.0-fasrc01
module load openmpi/4.1.1-fasrc01

# Run program
srun -n 8 --mpi=pmix ./pi_monte_carlo.x
```

### Example Usage:

```bash
module load intel/21.2.0-fasrc01
module load openmpi/4.1.1-fasrc01
make
sbatch run.sbatch
```

### Example Output:

```bash
 Exact PI:    3.14159265
 Computed PI: 3.14156124
 Error:       0.00100%
 Total time: 14.72 sec
```
