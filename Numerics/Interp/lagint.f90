!=====================================================================
! FUNCTION LAGINT: Lagrange interpolation
!
! Input:
! xx    -- abscissa at which the interpolation is to be evaluated
! xi()  -- arrays of data abscissas
! yi()  -- arrays of data ordinates
! ni    -- size of the arrays xi() and yi()
! n     -- number of points for interpolation (order of interp. = n-1)
!
! Output:
! lagint- interpolated value
!
! Comments:
! if ( n > ni ) n = ni
! Program works for both equally and unequally spaced xi()
!=====================================================================
function lagint(xx, xi, yi, ni, n)
  implicit none
  integer(4) :: i
  integer(4) :: j
  integer(4) :: k
  integer(4) :: js
  integer(4) :: jl
  integer(4) :: ni
  integer(4) :: n
  real(8)    :: xi(ni)
  real(8)    :: yi(ni)
  real(8)    :: lambda(ni)
  real(8)    :: y
  real(8)    :: lagint
  real(8)    :: xx

  ! check order of interpolation
  if ( n > ni ) n = ni

  ! if x is ouside the xi(1)-xi(ni) interval take a boundary value
  if ( xx <= xi(1) ) then
     lagint = yi(1)
     return
  end if
  if ( xx >= xi(ni) ) then
     lagint = yi(ni)
     return
  end if

  ! a binary (bisectional) search to find i so that xi(i) < x < xi(i+1)
  i = 1
  j = ni
  do while ( j > i+1 )
     k = ( i + j ) / 2
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

  ! Lagrange interpolation
  y = 0.0
  do js =i, i + n - 1
     lambda(js) = 1.0
     do jl = i, i + n - 1
        if( jl /= js ) lambda(js)=lambda(js)*(xx-xi(jl))/(xi(js)-xi(jl))
     end do
     y = y + yi(js)*lambda(js)
  end do
  lagint = y
  return
end function lagint
