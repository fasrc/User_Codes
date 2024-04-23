!=====================================================
! Fortran 90 MPI example: mpitest.f90
!=====================================================
program mpitest
  implicit none
  include 'mpif.h'
  integer(4) :: ierr
  integer(4) :: iproc
  integer(4) :: nproc
  integer(4) :: i
  call MPI_INIT(ierr)
  call MPI_COMM_SIZE(MPI_COMM_WORLD,nproc,ierr)
  call MPI_COMM_RANK(MPI_COMM_WORLD,iproc,ierr)
  do i = 0, nproc-1
     call MPI_BARRIER(MPI_COMM_WORLD,ierr)
     if ( iproc == i ) then
        write (6,*) 'Rank',iproc,'out of',nproc
     end if
  end do
  call MPI_FINALIZE(ierr)
  if ( iproc == 0 ) write(6,*)'End of program.'
!  stop
end program mpitest
