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
