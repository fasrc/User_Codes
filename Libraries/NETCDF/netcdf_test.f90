!=====================================================================
! Program: netcdf_test.f90
!          Create a random vector and write it to a NC file
!=====================================================================
program netcdf_test
  use netcdf
  implicit none
  integer(4)            :: i
  integer(4)            :: iseed
  integer(4), parameter :: n = 100
  real(8), allocatable  :: darr(:)
  real(8), external     :: ran3

  ! NetCDF variables..................................................
  character (len = *), parameter :: FILE_NAME = "vector_x.nc"
  integer(4), parameter          :: NDIMS = 1
  integer(4)                     :: ncid
  integer(4)                     :: varid
  integer(4)                     :: dimids(NDIMS)
  integer(4)                     :: x_dimid

  iseed = -99
  ! Allocate memory...................................................
  if ( .not. allocated ( darr ) ) allocate ( darr(n) )

  ! Create random vector DARR.........................................
  do i = 1, n
     darr(i) = ran3(iseed)
  end do

  !do i = 1, n 
  !   write(6,'(1x, i4, 1x, f8.5)') i, darr(i)
  !end do

  ! Create the netCDF file. The nf90_clobber parameter tells netCDF to
  ! overwrite this file, if it already exists.
  call check( nf90_create(FILE_NAME, NF90_CLOBBER, ncid) )

  ! Define the dimensions. NetCDF will hand back an ID for each. 
  call check( nf90_def_dim(ncid, "x", n, x_dimid) )

  ! The dimids array is used to pass the IDs of the dimensions of
  ! the variables. Note that in fortran arrays are stored in
  ! column-major format.
  dimids =  (/ x_dimid /)
  
  ! Define the variable. The type of the variable in this case is
  ! NF90_DOUBLE
  call check( nf90_def_var(ncid, "data", NF90_DOUBLE, dimids, varid) )

  ! End define mode. This tells netCDF we are done defining metadata.
  call check( nf90_enddef(ncid) )

  ! Write the pretend data to the file. Although netCDF supports
  ! reading and writing subsets of data, in this case we write all the
  ! data in one operation.
  call check( nf90_put_var(ncid, varid, darr) )

  ! Close the file. This frees up any internal netCDF resources
  ! associated with the file, and flushes any buffers.
  call check( nf90_close(ncid) )

  write(6,*) '*** SUCCESS writing example file vector_x.nc! ***'

  ! Free memory.......................................................
  if ( allocated ( darr ) ) deallocate ( darr )

contains
  subroutine check(status)
    integer(4), intent ( in) :: status
    
    if( status /= nf90_noerr) then 
      write(6,*) trim(nf90_strerror(status))
      stop 'Stopped'
    end if
  end subroutine check

end program netcdf_test

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
