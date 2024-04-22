### Purpose:

Program generates a random matrix and calculates its eigen values. Uses OpenMP to
generate the matrix and threaded version of MKL diagonalize it.

### Contents:

* `omp_diag.f90`: Fortran source code
* `omp_diag.out`: Output file
* `Makefile`: Makefile to compile the source code
* `run.sbatch`: Btach-job submission script to send the job to the queue.

### Source Code:

```fortran
!====================================================================
! Program: omp_diag.f90
!
! Create a random matrix and diagonalize it
!====================================================================
program omp_diag
  use IFPORT
  implicit none
  integer(4) :: i
  integer(4) :: j
  integer(4), parameter :: n = 1000    ! Matrix dimension ( n x n )
  integer(4), parameter :: m = 10      ! Number of eigen values to print out
  integer(4), parameter :: iseed = -99 ! Seed for random number generator
  real(8), allocatable :: h(:,:)       ! Matrix to diagonalize
  real(8), allocatable :: eig(:)       ! Array with eigen values of h
! OpenMP variables....................................................
  integer(4) :: NTHREADS
  integer(4) :: TID
  integer(4) :: OMP_GET_NUM_THREADS
  integer(4) :: OMP_GET_THREAD_NUM

! Initialize the random number generator
  call srand(iseed)

! Allocate memory.....................................................
  if ( .not. allocated ( h ) ) allocate ( h(n,n) )
  if ( .not. allocated ( eig ) ) allocate ( eig(n) )


!$OMP PARALLEL PRIVATE(TID, i, j) &
!$OMP          SHARED(NTHREADS, h)
  TID = OMP_GET_THREAD_NUM()
  NTHREADS = OMP_GET_NUM_THREADS()
  if ( TID == 0 ) write(6,*) 'Number of threads:', NTHREADS
! Create random test matrix h.........................................
!$OMP DO
  do j = 1, n
     do i = 1, j
        h(i,j) = rand()
        h(j,i) = h(i,j)
     end do
  end do
!$OMP END PARALLEL

! Diagonalize the matrix h............................................
  call diasym(h,eig,n)
! Write eigen values of h.............................................
  write(6,*)'First M eigen values of h:'
  do i = 1, m
     write(6,*)i,eig(i)
  end do

! Free memory.........................................................
  if ( allocated ( h ) ) deallocate ( h )
  if ( allocated ( eig ) ) deallocate ( eig )
 
  stop
end program omp_diag

!=====================================================================
! Call LAPACK diagonalization subroutine DSYEV
! Input:  a(n,n) = real symmetric matrix to be diagonalized!
!             n  = size of a
! Output: a(n,n) = orthonormal eigenvectors of a
!         eig(n) = eigenvalues of a in ascending order
!=====================================================================
 subroutine diasym(h,eig,n)
   implicit none
   integer(4) :: n
   integer(4) :: l
   integer(4) :: inf
   real(8)    :: h(n,n)
   real(8)    :: eig(n)
   real(8)    :: work(n*(3+n/2))
   l=n*(3+n/2)
   call dsyev('V','U',n,h,n,eig,work,l,inf)
   return
 end subroutine diasym
```

### Example Usage:

```bash
module load intel/24.0.1-fasrc01
module load intel-mkl/24.0.1-fasrc01
make
sbatch run.sbatch
```
    
### Example Output:

```
 Number of threads:           4
 First M eigen values of h:
           1  -18.1253011794745     
           2  -17.9175893861649     
           3  -17.8657590396365     
           4  -17.7089875061960     
           5  -17.5353871722488     
           6  -17.4775521893892     
           7  -17.2787813532336     
           8  -17.1980037379388     
           9  -17.1360982340428     
          10  -17.0351232203672 
```
