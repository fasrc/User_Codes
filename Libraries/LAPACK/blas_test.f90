!====================================================================
! BLAS Test
!
! Creates a random matrix and a random vector and computes
! matrix-vector multiplication
!
! COMPILE: gfortran -o blas_test.x  blas_test.f90 -O2 -lblas
!====================================================================
program blas_test
  implicit none
  integer(4) :: i
  integer(4) :: j
  integer(4) :: iseed               ! Seed for random number generator
  integer(4), parameter :: n = 100  ! Problem dimension
  real(8), allocatable :: h(:,:)    ! Matrix to diagonalize
  real(8), allocatable :: vin(:)
  real(8), allocatable :: vout(:)
  real(8), external :: ran3

  iseed = -99
 
! Allocate memory.....................................................
  if ( .not. allocated ( h ) ) allocate ( h(n,n) )
  if ( .not. allocated ( vin ) ) allocate ( vin(n) )
  if ( .not. allocated ( vout ) ) allocate ( vout(n) )

! Create random test matrix h.........................................
  do i = 1, n
     do j = 1, n
        h(i,j) = ran3(iseed)
!        h(j,i) = h(i,j)
     end do
  end do

! Create a random vector (vin)........................................
  do i = 1, n
     vin(i) = ran3(iseed)
  end do

! Call DGEMV to compute [H] [Vin] = [Vout]...........................
  call DGEMV('N', n, n, 1.0d0, h, 100, vin, 1, 0.0d0, vout, 1)
  
! Write Vout..........................................................
  write(6,*)'Vector Vout:'
  do i = 1, n
     write(6,*)i,vout(i)
  end do

! Free memory.........................................................
  if ( allocated ( h ) ) deallocate ( h )
  if ( allocated ( vin ) ) deallocate ( vin )
  if ( allocated ( vout ) ) deallocate ( vout )

  stop 'End of program.'
end program blas_test

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
