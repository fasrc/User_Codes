!====================================================================
! Program: hdf5_test.f90
!          Create a random vector and write it to a HDF5 file
!====================================================================
program hdf5_test
  use hdf5
  implicit none
  integer(4) :: i
  integer(4) :: iseed               ! Seed for random number generator
  integer(4), parameter :: n = 100  ! Dimension
  real(8), allocatable :: darr(:)   ! Random array
  real(8), external :: ran3         ! Random number generator
! HDF variables.......................................................
  character(len=9), parameter :: filename = "output.h5"   ! File name 
  character(len = 5), parameter :: dsetname_darr = "darr" ! Dataset name
  integer(HID_T)                :: file_id                ! file identifier
  integer(HID_T)                :: dset_id                ! Dataset identifier
  integer(HID_T)                :: dspace_id              ! Dataspace identifier
  integer(HSIZE_T)              :: dims1(1)               ! Dataset dimensions
  integer(4)                    :: rank1 = 1              ! Dataset rank
  integer(4)                    :: error                  ! Error flag

  iseed = -99
 
! Allocate memory.....................................................
  if ( .not. allocated ( darr ) ) allocate ( darr(n) )

! Create random vector DARR...........................................
  do i = 1, n
     darr(i) = ran3(iseed)
  end do

! Write HDF5 file.....................................................
  dims1 = n
  call h5open_f(error) ! open interface
  call h5fcreate_f(filename, H5F_ACC_TRUNC_F, file_id, error) ! open filestream
  call h5screate_simple_f(rank1, dims1, dspace_id, error)
  call h5dcreate_f(file_id, dsetname_darr, h5t_native_double, dspace_id, dset_id, error)
  call h5dwrite_f(dset_id, H5T_NATIVE_DOUBLE, darr, dims1, error)
  call h5dclose_f(dset_id, error)
  call h5sclose_f(dspace_id, error)
  call h5fclose_f(file_id, error) ! close filestream
  call h5close_f(error) ! close interface

! Free memory.........................................................
  if ( allocated ( darr ) ) deallocate ( darr )
 
  stop 'End of program.'
end program hdf5_test

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
