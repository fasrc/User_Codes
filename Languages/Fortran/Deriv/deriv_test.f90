!===========================================================
! Program: deriv_test.f90
!
! Example program calculating the first derivative of 
! sin(x) in the interval [0, PI] using three-point 
! Lagrange interpolation.
!
! Compile:
!
! (i) Intel Fortran Compiler (ifort)
!
! ifort -o deriv_test.x deriv_tab.f90 deriv_test.f90 -O2
! 
! (ii) GNU Fortran Compiler (gfortran)
!
! gfortran -o deriv_test.x deriv_tab.f90 deriv_test.f90 -O2
! 
!===========================================================
program deriv_test
  implicit none
  integer(4), parameter :: ni = 100
  integer(4)            :: i
  real(8), parameter    :: pi = 3.1415926535897932d0
  real(8)               :: xmin
  real(8)               :: xmax
  real(8)               :: step
  real(8)               :: x
  real(8)               :: y
  real(8)               :: xi(ni)
  real(8)               :: yi(ni)
  real(8)               :: dy(ni)
  real(8)               :: dex(ni)
  real(8), external     :: deriv_tab

  ! Define function interval and compute the step
  xmin = 0.0d0
  xmax = pi
  step = ( xmax - xmin ) / dfloat( ni - 1 )
  
  x = 0.0d0
  do i = 1, ni
     y = sin(x)
     xi(i) = x
     yi(i) = y
     dex(i) = cos(x) ! Exact derivative
     x = x + step
  end do

  ! Calculate derivative
  x = 0.0d0
  do i = 1, ni
     dy(i) = deriv_tab(x, xi, yi, ni, 1) ! Computed derivative
     x = x + step
  end do
 
  ! Print out results
  write(6,"(8x,a,11x,a,9x,a,7x,a)") &
       "x", "y", "dy/dx", "Exact"
  do i = 1, ni
     write(6,"(4(2x,f10.6))") xi(i), yi(i), dy(i), dex(i)
  end do

  stop
end program deriv_test
