!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
! Program: pi_monte_carlo.f90
!          Calculate PI via parallel Monte-Carlo algorithm
!
! Compile: mpif90 -o pi_monte_carlo.x pi_monte_carlo.f90 -O2
! 
! Run:     mpirun -np <Number_of_MPI_ranks> ./pi_monte_carlo.x
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
program pi_monte_carlo
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

  real(8), external  :: ran3
  real(8), parameter :: pi=3.141592653589793238462643d0

  ! Initialize MPI....................................................
  call MPI_INIT(ierr)
  icomm = MPI_COMM_WORLD
  call MPI_COMM_SIZE(icomm,nproc,ierr)
  call MPI_COMM_RANK(icomm,iproc,ierr)

  t1 = MPI_WTIME(ierr)

  iseed = -99           ! Seed for random number generator
  r = 1.0d0             ! Unit circle
  sample_number = 1e10  ! Number of samples
  
  ! Convert to INTEGER8...............................................
  iproc8 = iproc
  nproc8 = nproc

  ! Parallel Monte-Carlo sampling.....................................
  n = 0
  do i = 1+iproc8, sample_number, nproc8 
     x = r * ran3(iseed)
     y = r * ran3(iseed)
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
  INTEGER(4) :: idum
  INTEGER(4) :: mbig,mseed,mz
  REAL(8)    ::  fac
  PARAMETER (mbig=1000000000,mseed=161803398,mz=0,fac=1./mbig)
  INTEGER(4) :: i,iff,ii,inext,inextp,k
  INTEGER(4) :: mj,mk,ma(55)
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
