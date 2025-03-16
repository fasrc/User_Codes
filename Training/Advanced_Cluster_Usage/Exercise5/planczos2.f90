!=====================================================================
! Program: planczos2.f90
!          Lanczos diagonalization with reorthogonalization
!=====================================================================
! Autor: Plamen G Krastev
!        Harvard University
!        Faculty of Arts and Sciences
!        Research Computing
!        38 Oxford Street
!        Cambridge, MA 02138, USA
!        Email: plamenkrastev@fas.harvard.edu
!=====================================================================
! Modules.............................................................
module nodeinfo
  implicit none
  integer(4) :: nproc
  integer(4) :: iproc
  integer(4) :: icomm
end module nodeinfo

module lanczos_info
  implicit none
  integer(4) :: niter
  integer(4) :: nkeep
  integer(4) :: lvec_file = 11
  integer(4) :: coef_file = 22

  logical    :: reorthog
  logical    :: writetodisk
  logical    :: IO_MPI

  real(kind=8), allocatable :: lvec(:,:)
end module lanczos_info

module timing
  implicit none
  real(kind=8) :: timeorthog
  real(kind=8) :: timelast_ort
end module timing
!.....................................................................

! Main................................................................
program planczos2
  use ifport
  use nodeinfo
  use lanczos_info
  use timing
  implicit none
  include 'mpif.h'
  integer(4)           :: ierr
  integer(4)           :: n
  integer(4)           :: i
  integer(4)           :: j
  integer(4)           :: iseed
  integer(4)           :: ierror
  integer(4)           :: iter
  real(8)              :: dnorm
  real(8)              :: dsclrprod
  real(8), allocatable :: h(:,:)
  real(8), allocatable :: dh(:)
  real(8), allocatable :: eh(:)
  real(8), allocatable :: zh(:,:)
! Lanczos vectors
  real(8), allocatable :: v(:)
  real(8), allocatable :: w(:)
! Lanczos coefficients
  real(8), allocatable :: alpha(:)
  real(8), allocatable :: beta(:)
! arrays for diagonalization of tridiagonal matrix
  real(8),allocatable  :: d(:)
  real(8), allocatable :: e(:)
  real(8), allocatable :: z(:,:)
  real(8), external    :: ran3

  real(8)             :: da
  real(8)             :: db

! Used in reorthogonalization.........................................
  integer(4) :: nread
  integer(4) :: jvec
  real(8)    :: d_nproc

! Initialize MPI......................................................
  call MPI_INIT(ierr)
  icomm = MPI_COMM_WORLD
  call MPI_COMM_SIZE(icomm,nproc,ierr)
  call MPI_COMM_RANK(icomm,iproc,ierr)

  writetodisk = .true.
  reorthog    = .true.
  IO_MPI      = .true.

  if ( writetodisk ) call open_file(lvec_file,IO_MPI)
!.....................................................................
! initialize random number generator
  iseed = -99
  
!  if ( iproc == 0 ) then
!     write(6,*)'Enter matrix dimension:'
!     read(5,*)n
!     write(6,*)'Enter nkeep and niter:'
!     read(5,*)nkeep,niter
!  end if
!  call MPI_BCAST(n,1,MPI_INTEGER,0,icomm,ierr)
!  call MPI_BCAST(nkeep,1,MPI_INTEGER,0,icomm,ierr)
!  call MPI_BCAST(niter,1,MPI_INTEGER,0,icomm,ierr)

! +++ RUN PARAMETERS +++
  n = 30000   ! Matrix dimension
  nkeep = 5   ! Number of eigen-values to keep
  niter = 50  ! Number of iterations
! +++ END OF RUN PARAMETERS +++  

! Allocate............................................................
  if ( .not. writetodisk ) then
     allocate( lvec(n,niter) )
  end if
  allocate( h(n,n), dh(n), eh(n), zh(n,n) )
  allocate( v(n), w(n) )
  allocate( alpha(niter), beta(niter) )
  allocate( d(niter),e(niter),z(n,niter) )
! create random test matrix h.........................................
  do j = 1, n
     do i = 1, j
        h(i,j) = rand() !ran3(iseed)
        h(j,i) = h(i,j)
     end do
  end do
! diagonalize matrix h (via Householder)..............................
  !call tred2(n,n,h,dh,eh,zh) 
  !call tqli(n,n,dh,eh,zh,ierror)
  !call eigsrt(dh,zh,n,n)
  !if ( ierr .ne. 0 .and. iproc == 0 ) then
  !   write(6,*) 'h - diag ierr=',ierror
  !end if
! Lanczos diagonalization.............................................
! Create a random initial vector......................................
  dnorm = 1.d0 / dsqrt(real(n,kind=8))
  do i = 1, n
     v(i) = dnorm
  end do
  call dnormvec(n,v,da)
  iter = 0
! Start Lanczos iterations............................................
  do while ( iter < niter )
     iter = iter + 1
     call  write_file(lvec_file,v,n,iter,IO_MPI)
     call applyh(n,h,v,w) ! H v = w
     call dvecproj(n,w,v,da)
     alpha(iter) = da
! Reorhogonalize......................................................
     if ( reorthog ) then
        call reorthogonalize(w,v,n,iter)
     end if
!.....................................................................
     if ( iter < niter ) then
        call dnormvec(n,v,db)
        beta(iter+1) = db
     end if
! Prepare to diagonalize..............................................
     d(:) = 0.0d0
     e(:) = 0.0d0
     z(:,:) = 0.0d0
     do j = 1, niter
        z(j,j) = 1.0d0
     end do
     do j = 1, iter
        d(j) = alpha(j)
        e(j) = beta(j)
     end do
! Diagonalize tridiagonal matrix......................................
     do j = 1, iter
        d(j) = alpha(j)
        e(j) = beta(j)
     end do
     call tqli(n,iter,d,e,z,ierr)
     call eigsrt(d,z,niter,niter)
     if ( iproc == 0 ) then
        if ( ierr .ne. 0 ) write(6,*) 'diag ierr=',ierror
        write(6,*) nkeep,' lowest eigenvalues - Lanczos'
        write(6,*) 'iteration:',iter
        do i = 1, nkeep
           write(6,*) i,d(i)
        end do
     end if
  end do ! while
  if ( iproc == 0 )write(6,*)'Lanczos iterations finished...' 

  if ( writetodisk ) call close_file(lvec_file,IO_MPI)
  call MPI_FINALIZE(ierr)
  stop
end program planczos2
!.....................................................................

!=====================================================================
! Apply H
!=====================================================================
subroutine applyh(n,h,vecin,vecout)
  implicit none
  integer(4) :: i
  integer(4) :: j
  integer(4) :: n
  real(8) :: h(n,n)
  real(8) :: vecin(n)
  real(8) :: vecout(n)
  do j = 1, n
     vecout(j) = 0.0d0
     do i = 1, n
        vecout(j) = vecout(j) + h(i,j) * vecin(i)
     end do
  end do
  return                                                                    
end subroutine applyh

!=====================================================================
! Open Lanczos file
!=====================================================================
subroutine open_file(fh,IO_MPI)
  use nodeinfo
  implicit none
  include 'mpif.h'

  integer(4)         :: ierr
  integer(4)         :: ilast
  integer(4)         :: file_mode,file_info
  integer(4)         :: fh
  character(len=4)   :: proc_name
  character(len=25)  :: filename
  logical            :: IO_MPI

  filename='lanczosvector'
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
     open(unit=fh,file=filename(1:ilast)//'.lvec',status = 'unknown',form='unformatted')
  else
     if ( IO_MPI ) then   !  use MPI-IO
        file_mode = MPI_MODE_CREATE + MPI_MODE_RDWR
        call MPI_INFO_CREATE(file_info,ierr)
        call MPI_INFO_SET(file_info,'access_style','write_mostly,read_mostly',ierr)
        call MPI_FILE_OPEN(icomm,filename(1:ilast)//'.lvec',file_mode,file_info,fh,ierr)
     else
        open(unit=fh,file=filename(1:ilast)//proc_name//'.lvec',status = 'unknown',form='unformatted')
     end if
  end if

  return
end subroutine open_file

!=====================================================================
! Close file
!=====================================================================
subroutine close_file(fh,IO_MPI)
  use nodeinfo
  implicit none
  include 'mpif.h'
  integer(4) :: ierr
  integer(4) :: fh
  logical    :: IO_MPI
  if ( nproc == 1) then
     rewind(fh)
     write(fh)0
     close(fh)
  else
     if ( IO_MPI ) then
        call MPI_FILE_CLOSE(fh,ierr)
     else
        rewind(fh)
        write(fh)0
        close(fh)
     end if
  end if
  return
end subroutine close_file

!=====================================================================
! Write File
!=====================================================================
subroutine write_file(iunit,v,dimbasis,i,IO_MPI)
  use nodeinfo
  implicit none
  include 'mpif.h'
  integer(4) :: ierr
  integer(4) :: iunit
  integer(4) :: i,j,k
  integer(4) :: dimbasis
  integer(4) :: iwrite
  real(8)    :: v(dimbasis)
  logical    :: IO_MPI

  integer(4) ::status(MPI_STATUS_SIZE)
  integer(kind=MPI_OFFSET_KIND) :: jvector,jbyte,jdim,disp

  iwrite  = mod( i - int( 1, kind = 4 ), nproc )
  jdim    = dimbasis
  jvector = i
  jbyte   = int(8,kind=MPI_OFFSET_KIND)
  disp    = ( jvector - 1 ) * jbyte * jdim
  if ( nproc == 1 ) then
     if ( i == 1 ) rewind(iunit)
     write( iunit ) ( v(k), k = 1, dimbasis )
  else
     if ( IO_MPI ) then
        if ( iproc == iwrite ) then
           call MPI_FILE_WRITE_AT(iunit,disp,v,dimbasis,MPI_REAL8,status,ierr)
        end if
     else
        if ( iproc == iwrite ) then
           if ( i == iproc + 1 ) rewind(iunit)
           write( iunit ) ( v(k), k = 1, dimbasis )
        end if
     end if
  end if
  return
end subroutine write_file

!=====================================================================
! Read File
!=====================================================================
subroutine read_file(iunit,v,dimbasis,i,IO_MPI)
  use nodeinfo
  implicit none
  include 'mpif.h'
  integer(4) :: ierr
  integer(4) :: iunit
  integer(4) :: i,j,k
  integer(4) :: dimbasis
  integer(4) :: iread
  real(8)    :: v(dimbasis)
  logical    :: IO_MPI

  integer(4) ::status(MPI_STATUS_SIZE)
  integer(kind=MPI_OFFSET_KIND) :: jvector,jbyte,jdim,disp

  iread = mod( i - int(1, kind = 4 ), nproc )
  jdim    = dimbasis
  jvector = i
  jbyte   = int(8,kind=MPI_OFFSET_KIND)
  disp    = ( jvector - 1 ) * jbyte * jdim
  if ( nproc == 1 ) then
     if ( i == 1 ) rewind(iunit)
     read( iunit ) ( v(k), k = 1, dimbasis )
  else
     if ( IO_MPI ) then
        if ( iproc == iread ) then
           call MPI_FILE_READ_AT(iunit,disp,v,dimbasis,MPI_REAL8,status,ierr)
        end if
     else
        if ( iproc == iread ) then
           if ( i == iproc + 1 ) rewind(iunit)
           read( iunit ) ( v(k), k = 1, dimbasis )
        end if
     end if
  end if
  return
end subroutine read_file

!=====================================================================
! Reorthogonalize
!=====================================================================
subroutine reorthogonalize(w,v,n,iter)
  use nodeinfo
  use lanczos_info
  use timing
  implicit none
  include 'mpif.h'
  integer(4) :: ierr
  integer(4) :: i
  integer(4) :: j
  integer(4) :: nread
  integer(4) :: iter
  integer(4) :: jvec
  integer(4) :: n
  real(8)    :: d_nproc
  real(8)    :: dsclrprod
  real(8)    :: w(n)
  real(8)    :: v(n)

  d_nproc = real(nproc,kind=8)
  do i = 1,n
     w(i) = w(i) / d_nproc
  end do
  nread = iter / nproc
  if ( nread * nproc < iter ) then
     if ( nread * nproc + iproc + 1 <= iter ) nread = nread + 1
  end if
  do j = 1, nread
     jvec = (j-1)*nproc+iproc+1
     call read_file(lvec_file,v,n,jvec,IO_MPI)
     dsclrprod = real(0,kind=8)
     do i = 1,n
        dsclrprod = dsclrprod + ( w(i) * v(i) )
     end do
     do i = 1, n
        w(i) = w(i) - ( v(i) * dsclrprod * d_nproc)
     end do
  end do
  if ( nproc > 1 ) then
     call MPI_ALLREDUCE(w,v,n,MPI_REAL8,MPI_SUM,icomm,ierr)
  else
     do i = 1, n
        v(i) = w(i)
     end do
  end if
! Here we return v....................................................
  return
end subroutine reorthogonalize

!=====================================================================
! Double-precision normalization
!
! n     --> dimension of vector
! dvec  --> double-precision vector
!
! dnorm --> double-precision norm of dvec
!=====================================================================
subroutine dnormvec(n,dvec,dnorm)
  implicit none
  integer(8) :: n
  real(8)    :: dvec(n)
  real(8)    :: dnorm
  real(8)    :: d  
  integer(4) :: i
  dnorm = 0.d0
  do i = 1, n
     d = dvec(i)
     dnorm = dnorm + d * d
  end do
  dnorm = dsqrt(dnorm)
  d = 1.d0 / dnorm
  do i = 1,n
     dvec(i) = dvec(i) * d
  end do
  return
end subroutine dnormvec


!=====================================================================
! Double-precision projection
!
! n: dimension of vector
! dvec1: double-precision vector
! dvec2: double-preciscion vector
!
! dsclrprod  = dvec1*dvec2
!
! dvec1 -> dvec1 - dvec2*dsclrprod
!=====================================================================
subroutine dvecproj(n,dvec1,dvec2,dsclrprod)
  implicit none
  integer(4) :: n
  real(8)    :: dvec1(n),dvec2(n)
  real(8)    :: dsclrprod
  real(8)    :: d1,d2
  integer(4) :: i

  dsclrprod = 0.d0

  do i = 1, n
     d1 = dvec1(i)
     d2 = dvec2(i)
     dsclrprod = dsclrprod + ( d1 * d2 )
  end do

  do i = 1, n
     d1 = dvec1(i)
     d2 = dvec2(i)
     dvec1(i) = d1 - ( d2 * dsclrprod )
  end do
  return
end subroutine dvecproj
