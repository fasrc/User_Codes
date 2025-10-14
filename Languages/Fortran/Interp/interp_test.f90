!===========================================================
! Program: interp_test.f90
!
! Example program illustrating Langrange interpolation.
! It approximates a Legendre polynomial, P_5(x), on a set 
! of predefined abscissa points.
!
! Compile:
!
! (i) Intel Fortran Compiler:
!
! ifort -o interp_test.x lagint.f90 interp_test.f90 -O2
!
! (i) GNU Fortran Compiler
!
! gfortran -o interp_test.x lagint.f90 interp_test.f90 -O2
!
!===========================================================
program interp_test
  implicit none
  integer(4), parameter :: order = 3 ! Order of interpolation
  integer(4), parameter :: ni = 50   ! Size of arrays xi() and yi()
  integer(4)            :: i
  real(8)               :: step
  real(8)               :: xmin
  real(8)               :: xmax
  real(8)               :: x
  real(8)               :: xx
  real(8)               :: xi(ni)
  real(8)               :: yi(ni)
  real(8)               :: yex
  real(8)               :: yint
  real(8), external     :: lagint
  real(8), external     :: p5
  

  xmin = -1.0d0
  xmax = 1.0d0
  step = ( xmax - xmin) / dfloat( ni - 1 )

  ! Fill in the arrays xi() and yi()
  x = xmin
  do i = 1, ni
     xi(i) = x
     yi(i) = p5(x)
     x = x + step
  end do

  ! Define a set of abscissa points and interpolate
  step = ( xmax - xmin) / dfloat( 10 - 1 )
  x = xmin
  write(6,"(8x,a,9x,a,6x,a)")"x", "P_5(x)", "Exact"
  do i = 1, 10
     yint = lagint(x, xi, yi, ni, order+1) !Interpolated values
     yex = p5(x)                           !Exact values
     write(6,"(3(2x,f10.6))") x, yint, yex
     x = x + step
  end do
  
  stop
end program interp_test

!===========================================================
! Function p5: Legendre Polynomial P_n(x), n = 5
!===========================================================
function p5(x)
  implicit none
  real(8) :: p5
  real(8) :: x
  p5 = (1.0d0/8.0d0)*(63.0d0*x**5 - 70.0d0*x**3 + 15.0d0*x)
  return
end function p5
