!=====================================================================
! Program: planczos2.f90
!          Lanczos diagonalization with reorthogonalization using OpenMP
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
module lanczos_info
  implicit none
  integer(4) :: niter
  integer(4) :: nkeep
  integer(4) :: lvec_file = 11
  integer(4) :: coef_file = 22

  logical    :: reorthog
  logical    :: writetodisk

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
  use lanczos_info
  use timing
  implicit none
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
  real(8), allocatable :: d(:)
  real(8), allocatable :: e(:)
  real(8), allocatable :: z(:,:)
  real(8), external    :: ran3

  real(8)             :: da
  real(8)             :: db

! Used in reorthogonalization.........................................
  integer(4) :: nread
  integer(4) :: jvec

! OpenMP initialization (no MPI)
  writetodisk = .true.
  reorthog    = .true.

  if (writetodisk) call open_file(lvec_file)
!.....................................................................
! Initialize random number generator
  iseed = -99
  
! +++ RUN PARAMETERS +++
  n = 70000   ! Matrix dimension
  nkeep = 5   ! Number of eigenvalues to keep
  niter = 50  ! Number of iterations
! +++ END OF RUN PARAMETERS +++  

! Allocate............................................................
  if (.not. writetodisk) then
     allocate(lvec(n,niter))
  end if
  allocate(h(n,n), dh(n), eh(n), zh(n,n))
  allocate(v(n), w(n))
  allocate(alpha(niter), beta(niter))
  allocate(d(niter), e(niter), z(n,niter))
! Create random test matrix h.........................................
  do j = 1, n
     do i = 1, j
        h(i,j) = rand() !ran3(iseed)
        h(j,i) = h(i,j)
     end do
  end do
! Lanczos diagonalization.............................................
! Create a random initial vector......................................
  dnorm = 1.d0 / dsqrt(real(n,kind=8))
  do i = 1, n
     v(i) = dnorm
  end do
  call dnormvec(n,v,da)
  iter = 0
! Start Lanczos iterations............................................
  do while (iter < niter)
     iter = iter + 1
     call write_file(lvec_file,v,n,iter) ! Serial I/O
     call applyh(n,h,v,w) ! H v = w
     call dvecproj(n,w,v,da)
     alpha(iter) = da
! Reorthogonalize......................................................
     if (reorthog) then
        call reorthogonalize(w,v,n,iter)
     end if
!.....................................................................
     if (iter < niter) then
        call dnormvec(n,w,db)
        beta(iter+1) = db
        !$omp parallel do
        do i = 1, n
           v(i) = w(i)
        end do
        !$omp end parallel do
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
     if (ierr .ne. 0) write(6,*) 'diag ierr=',ierror
     write(6,*) nkeep,' lowest eigenvalues - Lanczos'
     write(6,*) 'iteration:',iter
     do i = 1, nkeep
        write(6,*) i,d(i)
     end do
  end do ! while
  write(6,*) 'Lanczos iterations finished...' 

  if (writetodisk) call close_file(lvec_file)
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
  !$omp parallel do private(i)
  do j = 1, n
     vecout(j) = 0.0d0
     do i = 1, n
        vecout(j) = vecout(j) + h(i,j) * vecin(i)
     end do
  end do
  !$omp end parallel do
  return                                                                    
end subroutine applyh

!=====================================================================
! Open Lanczos file (serial)
!=====================================================================
subroutine open_file(fh)
  implicit none
  integer(4)         :: fh
  character(len=25)  :: filename

  filename = 'lanczosvector'
  open(unit=fh, file=trim(filename)//'.lvec', status='unknown', form='unformatted')
  return
end subroutine open_file

!=====================================================================
! Close file (serial)
!=====================================================================
subroutine close_file(fh)
  implicit none
  integer(4) :: fh
  rewind(fh)
  write(fh) 0
  close(fh)
  return
end subroutine close_file

!=====================================================================
! Write File (serial)
!=====================================================================
subroutine write_file(iunit,v,dimbasis,i)
  implicit none
  integer(4) :: iunit
  integer(4) :: i, k
  integer(4) :: dimbasis
  real(8)    :: v(dimbasis)

  if (i == 1) rewind(iunit)
  write(iunit) (v(k), k=1,dimbasis)
  return
end subroutine write_file

!=====================================================================
! Read File (serial)
!=====================================================================
subroutine read_file(iunit,v,dimbasis,i)
  implicit none
  integer(4) :: iunit
  integer(4) :: i, k
  integer(4) :: dimbasis
  real(8)    :: v(dimbasis)

  if (i == 1) rewind(iunit)
  read(iunit) (v(k), k=1,dimbasis)
  return
end subroutine read_file

!=====================================================================
! Reorthogonalize
!=====================================================================
subroutine reorthogonalize(w,v,n,iter)
  use lanczos_info
  use timing
  implicit none
  integer(4) :: i
  integer(4) :: j
  integer(4) :: nread
  integer(4) :: iter
  integer(4) :: jvec
  integer(4) :: n
  real(8)    :: dsclrprod
  real(8)    :: w(n)
  real(8)    :: v(n)
  real(8), allocatable :: temp_v(:)

  allocate(temp_v(n))
  !$omp parallel do
  do i = 1, n
     w(i) = w(i) ! No nproc scaling since no MPI
  end do
  !$omp end parallel do
  nread = iter
  do j = 1, nread
     jvec = j
     call read_file(lvec_file,temp_v,n,jvec) ! Serial read
     dsclrprod = 0.0d0
     !$omp parallel do reduction(+:dsclrprod)
     do i = 1, n
        dsclrprod = dsclrprod + (w(i) * temp_v(i))
     end do
     !$omp end parallel do
     !$omp parallel do
     do i = 1, n
        w(i) = w(i) - (temp_v(i) * dsclrprod)
     end do
     !$omp end parallel do
  end do
  !$omp parallel do
  do i = 1, n
     v(i) = w(i)
  end do
  !$omp end parallel do
  deallocate(temp_v)
  return
end subroutine reorthogonalize

!=====================================================================
! Double-precision normalization
!=====================================================================
subroutine dnormvec(n,dvec,dnorm)
  implicit none
  integer(8) :: n
  real(8)    :: dvec(n)
  real(8)    :: dnorm
  real(8)    :: d  
  integer(4) :: i
  dnorm = 0.d0
  !$omp parallel do reduction(+:dnorm)
  do i = 1, n
     d = dvec(i)
     dnorm = dnorm + d * d
  end do
  !$omp end parallel do
  dnorm = dsqrt(dnorm)
  d = 1.d0 / dnorm
  !$omp parallel do
  do i = 1, n
     dvec(i) = dvec(i) * d
  end do
  !$omp end parallel do
  return
end subroutine dnormvec

!=====================================================================
! Double-precision projection
!=====================================================================
subroutine dvecproj(n,dvec1,dvec2,dsclrprod)
  implicit none
  integer(4) :: n
  real(8)    :: dvec1(n), dvec2(n)
  real(8)    :: dsclrprod
  real(8)    :: d1, d2
  integer(4) :: i

  dsclrprod = 0.d0
  !$omp parallel do reduction(+:dsclrprod)
  do i = 1, n
     d1 = dvec1(i)
     d2 = dvec2(i)
     dsclrprod = dsclrprod + (d1 * d2)
  end do
  !$omp end parallel do
  !$omp parallel do
  do i = 1, n
     d1 = dvec1(i)
     d2 = dvec2(i)
     dvec1(i) = d1 - (d2 * dsclrprod)
  end do
  !$omp end parallel do
  return
end subroutine dvecproj
