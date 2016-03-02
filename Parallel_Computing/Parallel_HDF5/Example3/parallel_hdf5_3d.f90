!=====================================================================
! Program: parallel_hdf5_3d.f90
!
!          This code generates a random 3D array and writes it in
!          parallel to a HDF5 file
!
! Author: Plamen G Krastev
!         Harvard University
!         Research Computing
!         Faculty of Arts and Sciences
!         38 Oxford Street, Room 117
!         Cambridge, MA 02138, USA
!         Email: plamenkrastev@fas.harvard.edu
!
!=====================================================================
! Modules
!.....................................................................
module nodeinfo
  implicit none
  integer(4) :: iproc
  integer(4) :: nproc
  integer(4) :: icomm
end module nodeinfo

! Main program........................................................
program main
  use mpi
  use hdf5
  use nodeinfo
  implicit none
  integer(4) :: i
  integer(4) :: j
  integer(4) :: k
  integer(4) :: m
  integer(4) :: n
  integer(4) :: l
  integer(4) :: nn
  integer(4) :: ierr
  integer(4) :: info
  integer(4) :: iseed
  integer(4) :: ista
  integer(4) :: iend
  real(8), external :: ran3

  real(8), allocatable :: a(:,:,:) ! Global
  real(8), allocatable :: b(:,:,:) ! Local

  ! HDF5 stuff........................................................
  character(len=10), parameter :: filename = "pset_3d.h5"  ! File name
  character(len=12), parameter  :: dsetname = "DoubleArray"! Dataset name
  integer(HID_T)    :: file_id                             ! File identifier 
  integer(HID_T)    :: dset_id                             ! Dataset identifier 
  integer(HID_T)    :: filespace                           ! Dataspace identifier in file 
  integer(HID_T)    :: memspace                            ! Dataspace identifier in memory
  integer(HID_T)    :: plist_id                            ! Property list identifier 
  integer(HSIZE_T)  :: dimsf(3)                            ! Dataset dimensions.
  integer(HSIZE_T)  :: count(3)  
  integer(HSSIZE_T) :: offset(3) 

  integer(4) :: rank = 3                                   ! Dataset rank
  integer(4) :: error                                      ! Error flag
  
  icomm = MPI_COMM_WORLD
  info  = MPI_INFO_NULL

  ! Start MPI.........................................................
  call MPI_INIT(ierr)
  call MPI_COMM_SIZE(icomm, nproc, ierr)
  call MPI_COMM_RANK(icomm, iproc, ierr)
  
  iseed = -99

  m = 10 ! dimension in i
  n = 30 ! dimension in j
  l = 8  ! dimension in k

  dimsf(1) = m
  dimsf(2) = n
  dimsf(3) = l

  ! Allocate and generate global array................................
  if ( .not. allocated(a) ) allocate( a(m,n,l) )
  do k = 1, l
     do j = 1, n
        do i = 1, m
           a(i,j,k) = ran3(iseed)
        end do
     end do
  end do

  ! Allocate and fill in local arrays.................................
  call para_range(1, l, nproc, iproc, ista, iend)
  if ( .not. allocated(b) ) allocate( b(m,n,ista:iend) )
  do k = 1, l
     if ( ( k >= ista ) .and. ( k <= iend ) ) then
        do j = 1, n
           do i = 1, m
              b(i,j,k) = a(i,j,k)
           end do
        end do
     end if
  end do

  ! Print out A.......................................................
!  call MPI_BARRIER(icomm,ierr)
!  if ( iproc == 0 ) then
!     nn = 0
!     write(6,*)''
!     write(6,'(1x,a)') 'Matrix A'
!     write(6,'(5(4x,a))')'#','i','j','k','A(i,j,k)'
!     do k = 1, l
!        do j = 1, n
!           do i = 1, m
!              nn = nn  + 1
!              write(6,'(4(2x,i3),3x,f8.5)') nn, i, j, k, a(i,j,k)
!           end do
!        end do
!     end do
!  end if

  ! Print out B.......................................................
!  call MPI_BARRIER(icomm,ierr)
!  if ( iproc == 3 ) then
!     nn = 0
!     write(6,*)''
!     write(6,'(1x,a, 2x,i3)') 'Matrix B on proc', iproc
!     write(6,'(5(4x,a))')'#','i','j','k','B(i,j,k)'
!     do k = ista, iend
!        do j = 1, n
!           do i = 1, m
!              nn = nn  + 1
!              write(6,'(4(2x,i3),3x,f8.5)') nn, i, j, k, b(i,j,k)
!           end do
!        end do
!     end do
!  end if

  !             ++++++ WRITE HDF5 FILE IN PARALLEL ++++++

  ! Initialize FORTRAN predefined datatypes...........................
  call h5open_f(error) 
  
  ! Setup file access property list with parallel I/O access..........
  call h5pcreate_f(H5P_FILE_ACCESS_F, plist_id, error)
  call h5pset_fapl_mpio_f(plist_id, icomm, info, error)
  
  ! Create the file collectively...................................... 
  call h5fcreate_f(filename, H5F_ACC_TRUNC_F, file_id, error, access_prp = plist_id)
  call h5pclose_f(plist_id, error)

  ! Create the data space for the  dataset............................
  call h5screate_simple_f(rank, dimsf, filespace, error)
  
  ! Create the dataset with default properties........................
  call h5dcreate_f(file_id, dsetname, H5T_NATIVE_DOUBLE, filespace, dset_id, error)
  call h5sclose_f(filespace, error)

  ! Each process defines dataset in memory and writes it to the  hyperslab in the file. 
  count(1)  = dimsf(1)
  count(2)  = dimsf(2) 
  count(3)  = iend - ista + 1
  offset(1) = 0
  offset(2) = 0
  offset(3) = ista - 1 
  call h5screate_simple_f(rank, count, memspace, error) 

  ! Select hyperslab in the file......................................
  call h5dget_space_f(dset_id, filespace, error)
  call h5sselect_hyperslab_f (filespace, H5S_SELECT_SET_F, offset, count, error)

  ! Create property list for collective dataset write.................
  call h5pcreate_f(H5P_DATASET_XFER_F, plist_id, error) 
  call h5pset_dxpl_mpio_f(plist_id, H5FD_MPIO_COLLECTIVE_F, error)
  
  ! Write the dataset collectively....................................
  call h5dwrite_f(dset_id, H5T_NATIVE_DOUBLE, b, dimsf, error, &
       file_space_id = filespace, mem_space_id = memspace, xfer_prp = plist_id)

  ! Close dataspaces..................................................
  call h5sclose_f(filespace, error)
  call h5sclose_f(memspace, error)
  
  ! Close the dataset and property list...............................
  call h5dclose_f(dset_id, error)
  call h5pclose_f(plist_id, error)
  
  ! Close the file....................................................
  call h5fclose_f(file_id, error)
  
  ! Close FORTRAN predefined datatypes................................
  call h5close_f(error)
  
  ! Shut down MPI interface...........................................
  call MPI_FINALIZE(ierr)
  stop
end program main

!=====================================================================
! Calculates iteration range and/or array dimension for each core
! Adapted from "RS/6000 SP: Practical MPI Programming", IBM red book
!=====================================================================
subroutine para_range(n1, n2, nprocs, irank, ista, iend)
  implicit none
  integer(4) :: n1        ! Lowest value of iteration variable
  integer(4) :: n2        ! Highest value of iteration variable
  integer(4) :: nprocs    ! Number of cores
  integer(4) :: irank     ! Process rank
  integer(4) :: ista      ! Start of iterations for rank iproc
  integer(4) :: iend      ! End of iterations for rank iproc
  integer(4) :: iwork1
  integer(4) :: iwork2
  iwork1 = ( n2 - n1 + 1 )  /  nprocs
  iwork2 = mod(n2 - n1 + 1, nprocs)
  ista = irank * iwork1 + n1 + min(irank, iwork2)
  iend = ista + iwork1 - 1
  if ( iwork2 > irank ) iend = iend + 1
  return
end subroutine para_range

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
