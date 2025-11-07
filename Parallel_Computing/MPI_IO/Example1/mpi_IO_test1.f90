!=====================================================================
! Program: mpi_IO_test1.f90
!
!          Program illustrates use of MPI-IO
!          It generates 3 random vectors of dimension 20 and
!          writes them to disk with MPI IO. Then, it reads them
!          with MPI IO and writes them out to screen.
!
! Author: Plamen G Krastev
!         Harvard University
!         FAS Research Computing
!         38 Oxford Street
!         Cambridge, MA 02138, USA
!         Email: plamenkrastev@fas.harvard.edu
!
!=====================================================================
module nodeinfo
  implicit none
  integer(4) :: icomm
  integer(4) :: nproc
  integer(4) :: iproc
end module nodeinfo

program main
  use nodeinfo
  implicit none
  include 'mpif.h'
  integer(4)            :: ierr
  integer(4)            :: i
  integer(4)            :: k
  integer(4)            :: iseed    ! Seed for random number generator
  real(8)               :: ran3     ! Random number generator
  real(8), allocatable  :: vecin(:) ! Input vector
  real(8), allocatable  :: vecout(:)! Output vector
  integer(4), parameter :: n = 20   ! Vector dimension
  integer(4), parameter :: nvec = 3 ! Number of vectors to read / write
  ! para_range........................................................
  integer(4) :: istart,iend
  integer(4) :: jstart,jend,jrank
  !...................................................................
  integer(4),allocatable :: jdisp(:),jlen(:)
  real(8), allocatable   :: v2(:)

  integer(4) :: fh
  logical :: IO_MPI

  call MPI_INIT(ierr)
  icomm = MPI_COMM_WORLD
  call MPI_COMM_SIZE(icomm,nproc,ierr)
  call MPI_COMM_RANK(icomm,iproc,ierr)

  IO_MPI = .true.
  fh = 11

  ! Allocate memory...................................................
  if ( .not. allocated(vecin) )  allocate( vecin(n) ) 
  if ( .not. allocated(vecout) ) allocate( vecout(n) )
  if ( .not. allocated(jdisp) )  allocate( jdisp(0:nproc-1) )
  if ( .not. allocated(jlen) )   allocate( jlen(0:nproc-1) )
  if ( (iproc == 0) .and. (.not. allocated(v2)) ) allocate( v2(n) )

  ! Calculate iteration range for each processor......................
  do jrank = 0, nproc - 1
     call para_range(1,n,nproc,jrank,jstart,jend)
     jlen(jrank) = jend - jstart + 1
     jdisp(jrank) = jstart - 1
  end do
  call para_range(1,n,nproc,iproc,istart,iend)

  call open_file(fh,IO_MPI)

  ! Loop over vectors.................................................
  do k = 1,nvec
     ! Create random vector of dimension n............................
     do i = 1, n
        vecin(i) = ran3(k)
     end do
     call write_file(fh,IO_MPI,k,n,istart,jlen(iproc),vecin(istart:iend))
  end do
  do k = 1, nvec
     vecout(:) = 0.0d0
     call read_file(fh,IO_MPI,k,n,istart,jlen(iproc),vecout(istart:iend))
     call MPI_REDUCE(vecout,v2,n,MPI_REAL8,MPI_SUM,0,icomm,ierr)
     if ( iproc == 0 ) then
        write(6,'(a,1x,i2,a)')'Vector',k,':'
        do i =  1, n
           write(6,'(i3,2x,f8.4)')i,v2(i)
        end do
     end if
  end do
  call close_file(fh,IO_MPI)
  
  ! Free memory.......................................................
  if ( allocated(vecin) ) deallocate( vecin )
  if ( allocated(vecout) ) deallocate( vecout )
  if ( allocated(jdisp) ) deallocate( jdisp )
  if ( allocated(jlen) ) deallocate( jlen )
  if ( allocated(v2) ) deallocate( v2 )

  call MPI_BARRIER(icomm,ierr)
  call MPI_FINALIZE(ierr)
  stop
end program main

!=====================================================================
! Subroutine calculates iteration range
!=====================================================================
subroutine para_range(n1, n2, nprocs, irank, ista, iend)
  implicit none
  include 'mpif.h'
  integer(4) :: iwork
  integer(4) :: ista
  integer(4) :: iend
  integer(4) :: n1
  integer(4) :: n2
  integer(4) :: nprocs
  integer(4) :: irank
  iwork = (n2 - n1) / nprocs + 1
  ista = MIN(irank * iwork + n1, n2 + 1)
  iend = MIN(ista + iwork - 1, n2)
  return
end subroutine para_range

!=====================================================================
! Subroutine opens MPI file
!=====================================================================
subroutine open_file(fh,IO_MPI)
  use nodeinfo
  implicit none
  include 'mpif.h'

  integer(4)         :: ierr
  integer(4)         :: ilast
  integer(4)         :: file_mode,file_info,ierror
  integer(4)         :: fh
  character(len=4)   :: proc_name
  character (len=25) :: filename
  logical            :: IO_MPI

  filename='vectors'
  proc_name='_000'
  if ( iproc < 10 ) then
     write(proc_name(4:4),'(i1)')iproc
  else if ( iproc < 100 ) then
     write(proc_name(3:4),'(i2)')iproc
  elseif(iproc < 1000)then
     write(proc_name(2:4),'(i3)')iproc
  end if      

  ilast = index(filename,' ')-1
  if ( nproc == 1 ) then  !  normal I/O 
     open(unit=fh,file=filename(1:ilast)//'.lvec', & 
          status = 'unknown',form='unformatted')
  else
     if ( IO_MPI ) then   !  use MPI-IO
        file_mode = MPI_MODE_CREATE + MPI_MODE_RDWR
        call MPI_INFO_CREATE(file_info,ierror)
        call MPI_INFO_SET(file_info,'access_style','write_mostly', &
             ierror)
        call MPI_FILE_OPEN(icomm,filename(1:ilast)//'.lvec', &
             file_mode,file_info,fh,ierror)
     else
        open(unit=fh,file=filename(1:ilast)//proc_name//'.lvec', & 
             status = 'unknown',form='unformatted')           
     end if
  end if

  return
end subroutine open_file

!=====================================================================
! Close MPI file
!=====================================================================
subroutine close_file(fh,IO_MPI)
  use nodeinfo
  implicit none
  include 'mpif.h'
  integer(4) :: ierror
  integer(4) :: fh
  logical IO_MPI
  if ( nproc == 1) then
     rewind(fh)
     write(fh)0
     close(fh)
  else
     if ( IO_MPI ) then
        call MPI_FILE_CLOSE(fh,ierror)
     else
        rewind(fh)
        write(fh)0
        close(fh)
     end if
  end if
  return
end subroutine close_file

!=====================================================================
! Write MPI file
!=====================================================================
subroutine write_file(fh,IO_MPI,k,kdim,istart,n,v)
  use nodeinfo
  implicit none
  include 'mpif.h'

  integer(4) :: ierr
  integer(4) :: ierror
  integer(4) :: fh
  integer(4) :: n
  integer(4) :: i,j,k,kdim
  integer(4) :: istart
  integer(4) :: status(MPI_STATUS_SIZE)
  real(8)    :: v(n)
  logical    :: IO_MPI
  integer(kind=MPI_OFFSET_KIND) :: disp,ista,ibyte,idim,iter
  ista = istart
  iter = k
  idim = kdim
  ibyte = 8
  disp = (ista - 1) * ibyte + ( iter - 1 ) * idim * ibyte
  if ( nproc == 1 ) then
     write(fh) ( v(j), j = 1, n )
  else
     if ( IO_MPI ) then
        call MPI_FILE_WRITE_AT_ALL(fh,disp,v,n,MPI_REAL8,status,ierr)
     else
        write(fh) ( v(j), j = 1, n )
     end if
  end if
  return
end subroutine write_file

!=====================================================================
! Read MPI file
!=====================================================================
subroutine read_file(fh,IO_MPI,k,kdim,istart,n,v)
  use nodeinfo
  implicit none
  include 'mpif.h'

  integer(4) :: ierr
  integer(4) :: ierror
  integer(4) :: fh
  integer(4) :: n
  integer(4) :: i,j,k,kdim
  integer(4) :: istart
  integer(4) :: status(MPI_STATUS_SIZE)
  real(8)    :: v(n)
  logical    :: IO_MPI
  integer(kind=MPI_OFFSET_KIND) :: disp, ista, ibyte, idim, iter
  ista = istart
  idim = kdim
  iter = k
  ibyte = 8
  disp = (ista - 1) * ibyte + ( iter - 1 ) * idim * ibyte
  if ( nproc == 1 ) then
     if ( iproc == 0 ) rewind(fh)
     read(fh) ( v(j), j = 1, n )
  else
     if ( IO_MPI ) then
        call MPI_FILE_READ_AT_ALL(fh,disp,v,n,MPI_REAL8,status,ierr)
     else
        if ( iproc == 0 ) rewind(fh)
        read(fh) ( v(j), j = 1, n )
     end if
  end if
  return
end subroutine read_file

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
