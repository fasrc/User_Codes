!===========================================================
! NAME:    sum.f90
! PURPOSE: Computes sum of integers from 1 to 100
! COMPILE:
! GNU:     gfortran -o sum.x sum.f90
! Intel:   ifort -o sum.x sum.f90
!===========================================================
program sum
  implicit none
  integer(4) :: i
  integer(4) :: n
  integer(4) :: isum
  isum = 0
  do i = 1, 100
     isum = isum + i
  end do
  write(6,'(a,1x,i4,1a)') "The sum of integers from 1 to 100 is", isum, "."
  stop "End of program."
end program sum
