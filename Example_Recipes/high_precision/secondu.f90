function second ()

!   The permits one to obtain elapsed run time in seconds on most Unix systems.
!   Usage example:

!      double precision second, runtime, tm0, tm1
!      external second
!      tm0 = second ()
!      call sub
!      tm1 = second ()
!      runtime = tm1 - tm0

!   Note: On some systems, replace "etime" below with "etime_".

double precision second
real t(2)

second = etime_(t)
return
end

function secondwc ()
 
!   This routine, when used in conjunction with secondwe below, employs the
!   Fortran 2003 intrinsic system_clock to obtain elapsed wall clock run time.
!   Usage example:

!     double precision runtime, secondwc, secondwe
!     external secondwc, secondwe, tm0, tm1
!     tm0 = secondwc ()
!     call sub
!     tm1 = secondwc ()
!     runtime = secondwe (tm1 - tm0)

double precision secondwc
integer*4 itm1, itm2, itm3

call system_clock (itm1, itm2, itm3)
secondwc = itm1 / dble (itm2)
return
end

function secondwe (tm)

!   This corrects the result of secondwc when wrap-around has occurred.
!   The value "2147483.648d0" here should be the same as (itm3+1)/itm2, where
!   itm2 and itm3 are the second and third arguments output by system_clock.
!   See example above.

double precision secondwe, tm

if (tm >= 0.d0) then
  secondwe = tm
else
  secondwe = tm + 2147483.648d0
endif

return
end
