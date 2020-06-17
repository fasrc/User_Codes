!===========================================================
! NAME:    sum.f90
! PURPOSE: Computes sum of integers from 1 to N
! COMPILE:
! GNU:     gfortran -o sum.x sum.f90
! Intel:   ifort -o sum.x sum.f90
!
! USAGE:   ./sum.x -n <N>
!===========================================================
program sum
  implicit none
  integer(4)        :: i
  integer(4)        :: n
  integer(4)        :: isum
  character(len=10) :: op_n
  character(len=10) :: str_n

  call get_command_argument(1,op_n)
  call get_command_argument(2,str_n)

  if ( op_n /= '-n' ) then
     write(6,*) 'USAGE:'
     write(6,*) './sumx.x -n <N>'
     stop
  end if

  str_n  = trim(adjustl(str_n))
  
  read(str_n,*) n
  n = int(n)

  isum = 0
  do i = 1, n
     isum = isum + i
  end do
  write(6,'(a,1x,i4,1x,a3,1x,i4,a1)') 'The sum of integers from 1 to', n, 'is', isum, '.'
  !stop "End of program."
end program sum
