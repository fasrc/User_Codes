!=====================================================================
! Program: ptrap.f90
!
! Parallel implementation of the trapezoidal rule for integration
! Uses "cyclic" distribution of loop iterations
!=====================================================================
! Modules
!.....................................................................
module nodeinfo
  implicit none
  integer(4) :: iproc
  integer(4) :: nproc
  integer(4) :: icomm  
end module nodeinfo

! Main................................................................
program ptrap
  use nodeinfo
  implicit none
  include 'mpif.h'

  interface
     double precision function  func(x)
       implicit none
       double precision, intent(in) :: x        
     end function func
  end interface

  integer(4) :: ierr
  integer(4) :: n         ! number of steps
  real(8)    :: a         ! lower integration limit
  real(8)    :: b         ! upper integration limit
  real(8)    :: int       ! integral
!.....................................................................
  call MPI_INIT(ierr)
  icomm = MPI_COMM_WORLD
  call MPI_COMM_SIZE(icomm,nproc,ierr)
  call MPI_COMM_RANK(icomm,iproc,ierr)

  a = 0.0d0
  b = 4.0d0
  n = 80

!  if ( iproc == 0 ) then
!     write(6,*) ' Enter lower limit of integration: '
!     read(5,*) a
!     write(6,*) ' Enter upper limit of integration: '
!     read(5,*) b
!     write(6,*) ' Enter number of steps ( >= nproc): '
!     read(5,*)n
!  end if
!  call MPI_BCAST(a,1,MPI_REAL8,0,icomm,ierr)
!  call MPI_BCAST(b,1,MPI_REAL8,0,icomm,ierr)
!  call MPI_BCAST(n,1,MPI_INTEGER,0,icomm,ierr)
! Parallel integration................................................
  call ptrap_int(a,b,int,n,func)
  if ( iproc == 0 ) then
     write(6,'(2(a,1x,f3.1,1x),a,1x,f7.4)')'Integral from ',a,' to ',b,' is ',int
  end if

  call MPI_BARRIER(icomm,ierr)
  call MPI_FINALIZE(ierr)
  stop
end program ptrap

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
