!=====================================================================
! Function: deriv_tab.f90
!
! Evaluate first- or second-order derivatives, of tabulated function,
! using three-point Lagrange interpolation 
!
! Input: 
! xx   - the abscissa at which the interpolation is to be evaluated
! xi() - the arrays of data abscissas
! yi() - the arrays of data ordinates
! ni   - size of the arrays xi() and yi()
! m    - order of a derivative (1 or 2)
!
! Output: 
! deriv_tab  - interpolated value
!=====================================================================
real(8) function deriv_tab(xx, xi, yi, ni, m)
  implicit none
  integer(4), parameter :: n = 3
  integer(4)            :: ni
  integer(4)            :: m
  integer(4)            :: i
  integer(4)            :: j
  integer(4)            :: k
  integer(4)            :: ix
  real(8)               :: xx
  real(8)               :: xi(ni)
  real(8)               :: yi(ni)
  real(8)               :: x(n)
  real(8)               :: f(n)
  real(8)               :: dy

  ! Exit if too high-order derivative was needed,
  if ( m > 2 ) then
     deriv_tab = 0.0d0
     return
  end if

  ! if x is ouside the xi(1)-xi(ni) interval set deriv3=0.0
  if ( xx < xi(1) .or. xx > xi(ni) ) then
     deriv_tab = 0.0d0
     return
  end if

  ! a binary (bisectional) search to find i so that xi(i-1) < x < xi(i)
  i = 1
  j = ni
  do while ( j > i+1 )
     k = (i+j)/2
     if ( xx < xi(k) ) then
        j = k
     else
        i = k
     end if
  end do

  ! shift i that will correspond to n-th order of interpolation
  ! the search point will be in the middle in x_i, x_i+1, x_i+2 ...
  i = i + 1 - n/2

  ! check boundaries: if i is ouside of the range [1, ... n] -> shift i
  if ( i < 1 ) i = 1
  if ( i + n > ni ) i = ni - n + 1

  ! just wanted to use index i
  ix = i

  ! initialization of f(n) and x(n)
  do i = 1, n
     f(i) = yi(ix+i-1)
     x(i) = xi(ix+i-1)
  end do

  ! calculate the first-order derivative using Lagrange interpolation
  if ( m == 1 ) then
     dy =             (2.0*xx - (x(2)+x(3)))*f(1)/((x(1)-x(2))*(x(1)-x(3)))
     dy = dy + (2.0*xx - (x(1)+x(3)))*f(2)/((x(2)-x(1))*(x(2)-x(3)))
     dy = dy + (2.0*xx - (x(1)+x(2)))*f(3)/((x(3)-x(1))*(x(3)-x(2)))
     ! calculate the second-order derivative using Lagrange interpolation
  else
     dy =          2.0*f(1)/((x(1)-x(2))*(x(1)-x(3)))
     dy = dy + 2.0*f(2)/((x(2)-x(1))*(x(2)-x(3)))
     dy = dy + 2.0*f(3)/((x(3)-x(1))*(x(3)-x(2)))
  end if

  deriv_tab = dy

  return
end function deriv_tab
