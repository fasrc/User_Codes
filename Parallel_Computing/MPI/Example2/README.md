## Purpose:

Parallel implementation of the trapezoidal rule for integration. Uses "cyclic" distribution of loop iterations. Currently set up to compute integral $\int_0^4 x^2 dx$ with 80 integration points.

### Contents:

* <code>ptrap.f90</code>: Fortran source code
* <code>Makefile</code>: Makefile to compile the source code
* <code>run.sbatch</code>: Btach-job submission script to send the job to the queue.

### Source Code:

```fortran
!=====================================================================
! Parallel trapezoidal integration (PGK @ SDSU)
! Uses cyclic distribution of loop iterations
!=====================================================================
subroutine ptrap_int(a,b,tsum,n,func)
  use nodeinfo
  implicit none
  include 'mpif.h'
  integer(4)             :: ierr
  integer(4), intent(in) :: n
  integer(4)             :: j
  real(8), intent(in)    :: a,b
  real(8), intent(inout) :: tsum
  real(8)                :: fa, fb, x, step, tmpsum

  interface
     double precision function  func(x)
       implicit none
       double precision, intent(in) :: x 
     end function func
  end interface
  
  step = ( b - a ) / dfloat(n)
  fa = func(a) / 2.0d0
  fb = func(b) / 2.0d0
  tsum = 0.0d0
  do j = 1 + iproc, n - 1, nproc 
     x = j * step + a
     tsum = tsum + func(x)
  end do
  call MPI_ALLREDUCE(tsum,tmpsum,1,MPI_REAL8,MPI_SUM,icomm,ierr)
  tsum = tmpsum
  tsum = ( tsum + fb + fa ) * step
  return
end subroutine ptrap_int

!=====================================================================
! User supplied function to integrate
! Currently set to f(x) = x**2
!=====================================================================
double precision function func(x)
  implicit none
  double precision, intent(in) :: x
  func = x * x
  return
end function func
```
### Example <code>Makefile</code>:

```bash
#==========================================================
# Make file for pi_monte_carlo.f90
#==========================================================
F90CFLAGS   = -c -O2
F90COMPILER = mpif90
PRO         = ptrap
OBJECTS     = $(PRO).o

${PRO}.x : $(OBJECTS)
	$(F90COMPILER) -o ${PRO}.x $(OBJECTS)

%.o : %.f90
	$(F90COMPILER) $(F90CFLAGS) $(<F)

clean : 
	rm *.o *.x *.mod
```

### Example Batch-Job Submission Script:

```bash
#!/bin/bash
#SBATCH -J ptrap
#SBATCH -o ptrap.out
#SBATCH -e ptrap.err
#SBATCH -p test
#SBATCH -t 30
#SBATCH -n 8
#SBATCH --mem-per-cpu=4000

# Load required modules
module load intel/24.0.1-fasrc01 openmpi/5.0.2-fasrc01

# Run program
srun -n $SLURM_NTASKS --mpi=pmix ./ptrap.x
```

### Example Usage:

```bash
module load intel/24.0.1-fasrc01 openmpi/5.0.2-fasrc01
make
sbatch run.sbatch
```
    
### Example Output:

```
> cat ptrap.out 
Integral from  0.0  to  4.0  is  21.3350
```
