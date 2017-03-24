!   David H Bailey   20 Mar 2017

!  COPYRIGHT AND DISCLAIMER:
!    All software in this package (c) 2017 David H. Bailey.
!    By downloading or using this software you agree to the copyright, disclaimer
!    and license agreement in the accompanying file DISCLAIMER.txt.

!   This program demonstrates three quadrature routines:

!   quadts:  Implements the tanh-sinh quadrature scheme of Takahashi and Mori,
!     for functions on a finite interval such as (0,1).
!   quades:  Implements the exp-sinh scheme, for functions on a
!     semi-infinite interval such as (0, infinity).
!   quadss:  Implements the sinh-sinh scheme, for functions on the entire
!     real line.

!   These schemes have two very desirable properties:  (a) the cost of
!   precomputing abscissa-eight pairs only increases linearly (instead of
!   quadratically, as with Gaussian quadrature), and (b) the schemes often
!   work well when the function has a blow-up singularity at one or both
!   endpoints.  Each of these routines proceeds level-by-level, with the
!   computational cost (and, often, the accuracy) approximately doubling with
!   each iteration, until a target accuracy (500-digit precision in this case),
!   based on an accuracy estimate computed by the program, is achieved.  These 
!   schemes are all variants on the tanh-sinh scheme, which is described here:

!   David H. Bailey, Xiaoye S. Li and Karthik Jeyabalan, "A comparison of
!   three high-precision quadrature schemes," Experimental Mathematics, 
!   vol. 14 (2005), no. 3, pg. 317-329, preprint available at
!   http://www.davidhbailey.com/dhbpapers/quadrature-em.pdf.

!   The function to be integrated must be defined in an external function
!   subprogram (see samples below), and the name of the function must be
!   included in a "type (mp_real)" and an "external" statement.  Prior to
!   calling the quadrature routine, the corresponding initialization routine
!   must be called to initialize the abscissa and weight arrays:  initqts,
!   initqes and initqss, corresponding to the quadrature routines quadts,
!   quades and quadss, respectively.

!   All of these routines are 100% THREAD SAFE -- all requisite parameters
!   and arrays are passed through subroutine arguments. 

!   Here are some specific instructions for the individual quadrature schemes:

!   quadts:  

!   For this routine, the endpoints are specified in the variables x1 and
!   x2. For some integrand functions, it is important that these endpoints be
!   computed to seconday precision (nwds2 words) in the calling program, that
!   these secondary precision values be passed to quadts (where scaled
!   abscissas are calculated to  precision), and that the function definition
!   itself uses these secondary precision scaled abscissas in any initial
!   subtractions or other sensitive operations involving the input argument.
!   Otherwise the accuracy of the quadrature result might only be half as
!   high as it otherwise could be.  See the function definitions of fun07 and
!   fun09 below for examples on how this is done.  The function evaluation
!   itself can and should be performed with primary precision (nwds1 words)
!   for faster run times.

!   In the initialization routine for quadts, abscissa-weight pairs are
!   calculated until the weights are smaller than 10^(neps2), where neps2
!   is the secondary precision epsilon (typically twice the primary precision
!   epsilon -- see below).  In some problems it may be necessary to adjust
!   neps2 to a more negative value to obtain the best accuracy.

!   quades:

!   For this routine, it is assumed that the left endpoint is x1, and the
!   right endpoint is +infinity.  The comment about computing the endpoints
!   to high precision in note for quadts above also applies here, as does
!   the comment about computing abscissa-weight pairs based on neps2.

!   quadss:

!   No endpoints are specified here -- the integral is performed over the
!   entire real line.  However, the comment about computing abscissa-weight
!   pairs, based on neps2, also applies here.

!   These inputs are set in the parameter statement below:

!   ndp1   Primary ("low") precision in digits; this is the target accuracy
!          of quadrature results.
!   ndp2   Secondary ("high") precision in digits. By default, ndp2 = 2*ndp1.
!   neps1  Log10 of the primary tolerance. By default, neps1 = - ndp1.
!   neps2  Log10 of the secondary tolerance. By default, neps2 = -ndp2.
!   nq1    Max number of phases in quadrature routine; adding 1 increases
!          (possibly doubles) the number of accurate digits in the result,
!          but also roughly doubles the run time. nq1 must be at least 3.
!   nq2    Space parameter for wk and xk arrays in the calling program.  By
!          default it is set to 12 * 2^nq1. Increase nq2 if directed by a 
!          message produced in initqts.
!   nwds1  Low precision in words. By default nwds1 = int (ndp1 / mpdpw + 2).
!   nwds2  High precision in words. By default nwds2 = int (ndp2 / mpdpw + 2).

!   The endpoints x1 and x2 are set in executable statements below.

subroutine initqes (nq1, nq2, nwds1, neps2, wk, xk)

!   This subroutine initializes the quadrature arays xk and wk using the
!   function x(t) = exp (pi/2*sinh(t)).  The argument nq2 is the space
!   allocated for wk and xk in the calling program.  By default it is set to 
!   12 * 2^nq1.  Increase nq2 if directed by a message produced below.
!   neps2 controls termination of the loop below, which ends when 
!   wk(k) * 10^(neps2) > 1. If quadts outputs the message "Increase 
!   Quadlevel" or "Terms too large", adjust nq1 and neps2 as necessary.

!   Both initqes and quades are 100% THREAD SAFE -- all requisite parameters
!   and arrays are passed through subroutine arguments. 

!   David H Bailey   20 Mar 2017

use mpmodule
implicit none
integer i, ierror, iprint, j, k, k1, ndebug, neps2, nq1, nq2, nwds1
parameter (iprint = 1024, ndebug = 2)
type (mp_real) eps2, h, p2, t1, t2, t3, t4, u1, u2, &
  wk(-1:nq2), xk(-1:nq2)

write (6, 1)
1 format ('initqes: Exp-sinh quadrature initialization')

eps2 = mpreal (10.d0, nwds1) ** neps2
p2 = 0.5d0 * mppi (nwds1)
h = mpreal (0.5d0 ** nq1, nwds1)
wk(-1) = mpreal (dble (nq1), nwds1)

do k = 0, nq2
  if (ndebug >= 2 .and. mod (k, iprint) == 0) write (6, *) k, nq2
    t1 = mpreal (dble (k) * h, nwds1)

!   xk(k) = exp (u1)
!   wk(k) = exp (u1) * u2
!   where u1 = pi/2 * sinh (t1) and u2 = pi/2 * cosh (t1)

  t2 = exp (t1)
  u1 = 0.5d0 * p2 * (t2 - 1.d0 / t2)
  u2 = 0.5d0 * p2 * (t2 + 1.d0 / t2)
  xk(k) = exp (u1)
  wk(k) = xk(k) * u2

  if (wk(k) * eps2 > 1.d0) goto 100
enddo

write (6, 2) nq2
2 format ('initqes: Table space parameter is too small; value =',i8)
stop

100 continue

xk(-1) = mpreal (dble (k), nwds1)
if (ndebug >= 2) then
  write (6, 3) k
3 format ('initqes: Table spaced used =',i8)
endif

return
end subroutine initqes

function quades (fun, x1, nq1, nq2, nwds1, nwds2, neps1, wk, xk)

!   This routine computes the integral of the function fun on the interval
!   (x1, inf) with a target tolerance of 10^neps1.  The quadrature level is
!   progressively increased (approximately doubling the work with each level)
!   until level nq1 has been performed or the target tolerance has been met.
!   nq2 is the size of the wk and xk arrays, which must first be initialized
!   by calling initqes.  If quades outpues the message "Increase Quadlevel"
!   or "Terms too large", adjust nq1 and neps2 as necessary in the call to
!   initqes.

!   Both initqes and quades are 100% THREAD SAFE -- all requisite parameters
!   and arrays are passed through subroutine arguments. 

!   David H Bailey  20 Mar 2017

use mpmodule
implicit none
integer i, ierror, ip(0:100), iz1, iz2, izx, j, k, k1, k2, n, ndebug, &
  nds, neps1, nq1, nq2, nqq1, nwds1, nwds2
parameter (izx = 5, ndebug = 2)
logical log1
real (8) d1, d2, d3, d4, dplog10q
type (mp_real) c10, eps1, eps2, epsilon1, err, fun, h, &
  quades, tsum, s1, s2, s3, t1, t2, t3, tw1, tw2, twi1, twi2, twmx, &
  wk(-1:nq2), xk(-1:nq2)
type (mp_real) x1, xx1, xx2
external fun, dplog10q

epsilon1 = mpreal (10.d0, nwds1) ** neps1
tsum = mpreal (0.d0, nwds1)
s1 = mpreal (0.d0, nwds1)
s2 = mpreal (0.d0, nwds1)
h = mpreal (1.d0, nwds1)
c10 = mpreal (10.d0, nwds1)

if (wk(-1) < dble (nq1)) then
  write (6, 1) nq1
1 format ('quades: quadrature arrays have not been initialized; nq1 =',i6)
  goto 140
endif
nqq1 = dble (wk(-1))
n = dble (xk(-1))

do k = 0, nqq1
  ip(k) = 2 ** k
enddo

do k = 1, nq1
  h = 0.5d0 * h
  s3 = s2
  s2 = s1
  k1 = ip(nqq1-k)
  k2 = ip(nqq1-k+1)
  iz1 = 0
  iz2 = 0
  twmx = mpreal (0.d0, nwds1)

!   Evaluate function at level k in x, avoiding unnecessary computation.

  do i = 0, n, k1
    if (mod (i, k2) /= 0 .or. k == 1) then

!   These next few lines, which scale the abscissas, must be performed in
!   high precision (nwds2 words) to ensure full accuracy in the quadrature
!   results, even though the abscissas xk(i) were computed in low precision.

      xx1 = x1 + mpreal (xk(i), nwds2)
      xx2 = x1 + 1.d0 / mpreal (xk(i), nwds2)
      log1 = xx1 > x1
  
!   The remaining computations are performed in low precision (nwds1 words).

      if (iz1 < izx) then
        t1 = fun (xk(i), nwds1)
        tw1 = t1 * wk(i)
        twi1 = abs (tw1)
        if (twi1 < epsilon1) then
          iz1 = iz1 + 1
        else
          iz1 = 0
        endif
      else
        t1 = mpreal (0.d0, nwds1)
        tw1 = mpreal (0.d0, nwds1)
      endif

      if (i > 0 .and. log1 .and. iz2 < izx) then
        t2 = fun (xx2, nwds1)
        tw2 = t2 * wk(i) / xk(i)**2
        twi2 = abs (tw2)
        if (twi2 < epsilon1) then
          iz2 = iz2 + 1
        else
          iz2 = 0
        endif
      else
        t2 = mpreal (0.d0, nwds1)
        tw2 = mpreal (0.d0, nwds1)
      endif

      tsum = tsum + tw1 + tw2
      twmx = max (twmx, abs (tw1), abs (tw2))
    endif
  enddo

!   Compute s1 = current integral approximation and err = error estimate.
!   Tsum is the sum of all tw1 and tw2 from the loop above.
!   Twmx is the largest absolute value of tw1 and tw2 from the loop above.
!   Twi1 and twi2 are the final nonzero values of abs(tw1) and abs(tw2).

  s1 =  h * tsum
  eps1 = twmx * epsilon1
  eps2 = max (twi1, twi2)
  d1 = dplog10q (abs (s1 - s2))
  d2 = dplog10q (abs (s1 - s3))
  d3 = dplog10q (eps1) - 1.d0
  d4 = dplog10q (eps2) - 1.d0

  if (k <= 2) then
    err = mpreal (1.d0, nwds1)
  elseif (d1 .eq. -999999.d0) then
    err = mpreal (0.d0, nwds1)
  else
    err = c10 ** nint (min (0.d0, max (d1 ** 2 / d2, 2.d0 * d1, d3, d4)))
  endif

!   Output current integral approximation and error estimate, to 60 digits.

  if (ndebug >= 2) then
    write (6, 2) k, nq1, nint (dplog10q (abs (err)))
2   format ('quades: Iteration',i3,' of',i3,'; est error = 10^',i5, &
      '; approx value =')
    call mpwrite (6, 80, 60, s1)
  endif

  if (k >= 3 .and. iz1 == 0 .and. iz2 == 0) then
    write (6, 3)
3   format ('quades: Terms too large -- adjust neps2 in call to initqss.')
    goto 140
  endif

  if (k >= 3 .and. err < eps1) then
    write (6, 4) nint (dplog10q (abs (err)))
4   format ('quades: Estimated error = 10^',i5)
    goto 140
  endif

  if (k >= 3 .and. err < eps2) then
    write (6, 5) nint (dplog10q (abs (err)))
5   format ('quades: Estimated error = 10^',i5/&
    'Adjust nq1 and neps2 in initqss for greater accuracy.')
    goto 140
  endif
enddo

140 continue

quades = s1
return
end function quades

subroutine initqss (nq1, nq2, nwds1, neps2, wk, xk)

!   This subroutine initializes the quadrature arays xk and wk using the
!   function x(t) = sinh (pi/2*sinh(t)).  The argument nq2 is the space
!   allocated for wk and xk in the calling program.  By default it is set to 
!   12 * 2^nq1.  Increase nq2 if directed by a message produced below.
!   neps2 controls termination of the loop below, which ends when 
!   wk(k) * 10^(neps2) > 1. If quadss outputs the message "Increase 
!   Quadlevel" or "Terms too large", adjust nq1 and neps2 as necessary.

!   Both initqss and quadss are 100% THREAD SAFE -- all requisite parameters
!   and arrays are passed through subroutine arguments. 

!   David H Bailey   20 Mar 2017

use mpmodule
implicit none
integer i, ierror, iprint, j, k, k1, ndebug, neps2, nq1, nq2, nwds1
parameter (iprint = 1024, ndebug = 2)
type (mp_real) eps2, h, p2, t1, t2, t3, t4, u1, u2, &
  wk(-1:nq2), xk(-1:nq2)

write (6, 1)
1 format ('initqss: Sinh-sinh quadrature initialization')

eps2 = mpreal (10.d0, nwds1) ** neps2
p2 = 0.5d0 * mppi (nwds1)
h = mpreal (0.5d0 ** nq1, nwds1)
wk(-1) = mpreal (dble (nq1), nwds1)

do k = 0, nq2
  if (ndebug >= 2 .and. mod (k, iprint) == 0) write (6, *) k, nq2
    t1 = mpreal (dble (k) * h, nwds1)

!   xk(k) = sinh (u1)
!   wk(k) = cosh (u1) * u2
!   where u1 = pi/2 * sinh (t1) and u2 = pi/2 * cosh (t1)

  t2 = exp (t1)
  u1 = 0.5d0 * p2 * (t2 - 1.d0 / t2)
  u2 = 0.5d0 * p2 * (t2 + 1.d0 / t2)
  t3 = exp (u1)
  xk(k) = 0.5d0 * (t3 - 1.d0 / t3)
  wk(k) = 0.5d0 * (t3 + 1.d0 / t3) * u2

  if (wk(k) * eps2 > 1.d0) goto 100
enddo

write (6, 2) nq2
2 format ('initqss: Table space parameter is too small; value =',i8)
stop

100 continue

xk(-1) = mpreal (dble (k), nwds1)
if (ndebug >= 2) then
  write (6, 3) k
3 format ('initqss: Table spaced used =',i8)
endif

return
end subroutine initqss

function quadss (fun, nq1, nq2, nwds1, neps1, wk, xk)

!   This routine computes the integral of the function fun on the interval
!   (-inf, inf) with a target tolerance of 10^neps1.  The quadrature level is
!   progressively increased (approximately doubling the work with each level)
!   until level nq1 has been performed or the target tolerance has been met.
!   nq2 is the size of the wk and xk arrays, which must first be initialized
!   by calling initqss.  If quadss outputs the message "Increase Quadlevel"
!   or "Terms too large", adjust nq1 and neps2 as necessary in the call to
!   initqss.

!   Both initqss and quadss are 100% THREAD SAFE -- all requisite parameters
!   and arrays are passed through subroutine arguments. 

!   David H Bailey  20 Mar 2017

use mpmodule
implicit none
integer i, ierror, ip(0:100), iz1, iz2, izx, j, k, k1, k2, n, ndebug, &
  nds, neps1, nq1, nq2, nqq1, nwds1
parameter (izx = 5, ndebug = 2)
real (8) d1, d2, d3, d4, dplog10q
type (mp_real) c10, eps1, eps2, epsilon1, err, fun, h, &
  quadss, tsum, s1, s2, s3, t1, t2, t3, tw1, tw2, twi1, twi2, twmx, &
  wk(-1:nq2), xk(-1:nq2)
external fun, dplog10q

epsilon1 = mpreal (10.d0, nwds1) ** neps1
tsum = mpreal (0.d0, nwds1)
s1 = mpreal (0.d0, nwds1)
s2 = mpreal (0.d0, nwds1)
h = mpreal (1.d0, nwds1)
c10 = mpreal (10.d0, nwds1)

if (wk(-1) < dble (nq1)) then
  write (6, 1) nq1
1 format ('quadss: quadrature arrays have not been initialized; nq1 =',i6)
  goto 140
endif
nqq1 = dble (wk(-1))
n = dble (xk(-1))

do k = 0, nqq1
  ip(k) = 2 ** k
enddo

do k = 1, nq1
  h = 0.5d0 * h
  s3 = s2
  s2 = s1
  k1 = ip(nqq1-k)
  k2 = ip(nqq1-k+1)
  iz1 = 0
  iz2 = 0
  twmx = mpreal (0.d0, nwds1)

!   Evaluate function at level k in x, avoiding unnecessary computation.

  do i = 0, n, k1
    if (mod (i, k2) /= 0 .or. k == 1) then
      if (iz1 < izx) then
        t1 = fun (xk(i), nwds1)
        tw1 = t1 * wk(i)
        twi1 = abs (tw1)
        if (twi1 < epsilon1) then
          iz1 = iz1 + 1
        else
          iz1 = 0
        endif
      else
        t1 = mpreal (0.d0, nwds1)
        tw1 = mpreal (0.d0, nwds1)
      endif

      if (i > 0 .and. iz2 < izx) then
        t2 = fun (-xk(i), nwds1)
        tw2 = t2 * wk(i)
        twi2 = abs (tw2)
        if (twi2 < epsilon1) then
          iz2 = iz2 + 1
        else
          iz2 = 0
        endif
      else
        t2 = mpreal (0.d0, nwds1)
        tw2 = mpreal (0.d0, nwds1)
      endif

      tsum = tsum + tw1 + tw2
      twmx = max (twmx, abs (tw1), abs (tw2))
    endif
  enddo

!   Compute s1 = current integral approximation and err = error estimate.
!   Tsum is the sum of all tw1 and tw2 from the loop above.
!   Twmx is the largest absolute value of tw1 and tw2 from the loop above.
!   Twi1 and twi2 are the final nonzero values of abs(tw1) and abs(tw2).

  s1 =  h * tsum
  eps1 = twmx * epsilon1
  eps2 = max (twi1, twi2)
  d1 = dplog10q (abs (s1 - s2))
  d2 = dplog10q (abs (s1 - s3))
  d3 = dplog10q (eps1) - 1.d0
  d4 = dplog10q (eps2) - 1.d0

  if (k <= 2) then
    err = mpreal (1.d0, nwds1)
  elseif (d1 .eq. -999999.d0) then
    err = mpreal (0.d0, nwds1)
  else
    err = c10 ** nint (min (0.d0, max (d1 ** 2 / d2, 2.d0 * d1, d3, d4)))
  endif

!   Output current integral approximation and error estimate, to 60 digits.

  if (ndebug >= 2) then
    write (6, 2) k, nq1, nint (dplog10q (abs (err)))
2   format ('quadss: Iteration',i3,' of',i3,'; est error = 10^',i5, &
      '; approx value =')
    call mpwrite (6, 80, 60, s1)
  endif

  if (k >= 3 .and. iz1 == 0 .and. iz2 == 0) then
    write (6, 3)
3   format ('quadss: Terms too large -- adjust neps2 in call to initqss.')
    goto 140
  endif

  if (k >= 3 .and. err < eps1) then
    write (6, 4) nint (dplog10q (abs (err)))
4   format ('quadss: Estimated error = 10^',i5)
    goto 140
  endif

  if (k >= 3 .and. err < eps2) then
    write (6, 5) nint (dplog10q (abs (err)))
5   format ('quadss: Estimated error = 10^',i5/&
    'Adjust nq1 and neps2 in initqss for greater accuracy.')
    goto 140
  endif
enddo

140 continue

quadss = s1
return
end function quadss

subroutine initqts (nq1, nq2, nwds1, neps2, wk, xk)

!   This subroutine initializes the quadrature arays xk and wk using the
!   function x(t) = tanh (pi/2*sinh(t)).  The argument nq2 is the space
!   allocated for wk and xk in the calling program.  By default it is set to 
!   12 * 2^nq1.  Increase nq2 if directed by a message produced below.
!   Upon completion, wk(-1) = nq1, and xk(-1) = n, the maximum space parameter
!   for these arrays.  In other words, the arrays occupy (wk(i), i = -1 to n)
!   and (xk(i), i = -1 to n), where n = xk(-1).   The array x_k contains 
!   1 minus the abscissas; the wk array contains the weights at these abscissas.
!   If quadts outputs the message "Increase Quadlevel" or "Terms too large",
!   adjust nq1 and neps2 as necessary in the call to initqss.

!   Both initqts and quadts are 100% THREAD SAFE -- all requisite parameters
!   and arrays are passed through subroutine arguments. 

!   These computations are performed in low precision (nwds1), although
!   computations continue until weights are smaller than 10^neps2. 

!   David H Bailey   20 Mar 2017

use mpmodule
implicit none
integer i, ierror, iprint, j, k, k1, ndebug, neps2, nq1, nq2, nwds1
parameter (iprint = 1024, ndebug = 2)
type (mp_real) eps2, h, p2, t1, t2, t3, t4, u1, u2, wk(-1:nq2), xk(-1:nq2)

write (6, 1)
1 format ('initqts: Tanh-sinh quadrature initialization')

eps2 = mpreal (10.d0, nwds1) ** neps2
p2 = 0.5d0 * mppi (nwds1)
h = mpreal (0.5d0 ** nq1, nwds1)
wk(-1) = mpreal (dble (nq1), nwds1)

do k = 0, nq2
  if (ndebug >= 2 .and. mod (k, iprint) == 0) write (6, *) k, nq2
  t1 = mpreal (dble (k) * h, nwds1)

!   xk(k) = 1 - tanh (u1) = 1 /(e^u1 * cosh (u1))
!   wk(k) = u2 / cosh (u1)^2
!   where u1 = pi/2 * cosh (t1), u2 = pi/2 * sinh (t1)

  t2 = exp (t1)
  u1 = 0.5d0 * p2 * (t2 + 1.d0 / t2)
  u2 = 0.5d0 * p2 * (t2 - 1.d0 / t2)
  t3 = exp (u2)
  t4 = 0.5d0 * (t3 + 1.d0 / t3)
  xk(k) = 1.d0 / (t3 * t4)
  wk(k) = u1 / t4 ** 2

  if (wk(k) < eps2) goto 100
enddo

write (6, 2) nq2
2 format ('initqts: Table space parameter is too small; value =',i8)
stop

100 continue

xk(-1) = mpreal (dble (k), nwds1)
if (ndebug >= 2) then
  write (6, 3) k
3 format ('initqts: Table spaced used =',i8)
endif

return
end subroutine initqts


function quadts (fun, x1, x2, nq1, nq2, nwds1, nwds2, neps1, wk, xk)

!   This routine computes the integral of the function fun on the interval
!   (x1, x2) with a target tolerance of 10^neps1.  The quadrature level is
!   progressively increased (approximately doubling the work with each level)
!   until level nq1 has been performed or the target tolerance has been met.
!   nq2 is the size of the wk and xk arrays, which must first be initialized
!   by calling initqts. The function fun is not evaluated at the endpoints
!   x1 and x2.  If quadts outputs the message "Increase Quadlevel" or "Terms
!   too large", adjust nq1 and neps2 as necessary in the call to initqss.

!   Both initqts and quadts are 100% THREAD SAFE -- all requisite parameters
!   and arrays are passed through subroutine arguments. 

!   For some functions, it is important that the endpoints x1 and x2 be
!   computed to high precision (nwds2 words) in the calling program, that
!   these high-precision values be passed to quadts (where scaled abscissas
!   are calculated to high precision), and that the function definition
!   itself uses these high-precision scaled abscissas in any initial
!   subtractions or other sensitive operations involving the input argument.
!   Otherwise the accuracy of the quadrature result might only be half as
!   high as it otherwise could be.  See the function definitions of fun06,
!   fun07, fun09 and fun10 for examples on how this is done.  Otherwise the
!   function evaluation can and should be performed with low precision
!   (nwds1 words) for faster run times.  The two precision levels (nwds1
!   and nwds2) are specified by the user in the calling program.

!   David H Bailey  20 Mar 2017

use mpmodule
implicit none
integer i, ierror, ip(0:100), iz1, iz2, izx, j, k, k1, k2, n, ndebug, &
  nds, neps1, nq1, nq2, nqq1, nwds1, nwds2
parameter (izx = 5, ndebug = 2)
logical log1, log2
double precision d1, d2, d3, d4, dplog10q
type (mp_real) c10, eps1, eps2, epsilon1, err, fun, h, &
  quadts, tsum, s1, s2, s3, t1, t2, t3, tw1, tw2, twi1, twi2, twmx, &
  wk(-1:nq2), xk(-1:nq2)
type (mp_real) ax, bx, x1, x2, xki, xt1, xx1, xx2

external fun, dplog10q

!  These two lines are performed in high precision (nwds2 words).

ax = 0.5d0 * (x2 - x1)
bx = 0.5d0 * (x2 + x1)

!  The remaining initialization is performed in low precision (nwds1 words).

epsilon1 = mpreal (10.d0, nwds1) ** neps1
tsum = mpreal (0.d0, nwds1)
s1 = mpreal (0.d0, nwds1)
s2 = mpreal (0.d0, nwds1)
h = mpreal (1.d0, nwds1)
c10 = mpreal (10.d0, nwds1)

if (wk(-1) < dble (nq1)) then
  write (6, 1) nq1
1 format ('quadts: Quadrature arrays have not been initialized; nq1 =',i6)
  goto 140
endif
nqq1 = dble (wk(-1))
n = dble (xk(-1))

do k = 0, nqq1
  ip(k) = 2 ** k
enddo

do k = 1, nq1
  h = 0.5d0 * h
  s3 = s2
  s2 = s1
  k1 = ip(nqq1-k)
  k2 = ip(nqq1-k+1)
  iz1 = 0
  iz2 = 0
  twmx = mpreal (0.d0, nwds1)

!   Evaluate function at level k in x, avoiding unnecessary computation.

  do i = 0, n, k1  
    if (mod (i, k2) /= 0 .or. k == 1) then

!   These next few lines, which scale the abscissas, must be performed in
!   high precision (nwds2 words) to ensure full accuracy in the quadrature
!   results, even though the abscissas xk(i) were computed in low precision.

      xki = xk(i)
      xt1 = 1.d0 - mpreal (xki, nwds2)
      xx1 = - ax * xt1 + bx
      xx2 = ax * xt1 + bx
      log1 = xx1 > x1
      log2 = xx2 < x2      

!   The remaining computations are performed in low precision (nwds1 words).

      if (log1 .and. iz1 < izx) then
        t1 = fun (xx1, nwds1, nwds2)
        tw1 = t1 * wk(i)
        twi1 = abs (tw1)
        if (twi1 < epsilon1) then
          iz1 = iz1 + 1
        else
          iz1 = 0
        endif
      else
        t1 = mpreal (0.d0, nwds1)
        tw1 = mpreal (0.d0, nwds1)
      endif

      if (i > 0 .and. log2 .and. iz2 < izx) then
        t2 = fun (xx2, nwds1, nwds2)
        tw2 = t2 * wk(i)
        twi2 = abs (tw2)
        if (twi2 < epsilon1) then
          iz2 = iz2 + 1
        else
          iz2 = 0
        endif
      else
        t2 = mpreal (0.d0, nwds1)
        tw2 = mpreal (0.d0, nwds1)
      endif

      tsum = tsum + tw1 + tw2
      twmx = max (twmx, abs (tw1), abs (tw2))
    endif
  enddo

!   Compute s1 = current integral approximation and err = error estimate.
!   Tsum is the sum of all tw1 and tw2 from the loop above.
!   Twmx is the largest absolute value of tw1 and tw2 from the loop above.
!   Twi1 and twi2 are the final nonzero values of abs(tw1) and abs(tw2).

  s1 =  mpreal (ax, nwds1) * h * tsum
  eps1 = twmx * epsilon1
  eps2 = max (twi1, twi2)
  d1 = dplog10q (abs (s1 - s2))
  d2 = dplog10q (abs (s1 - s3))
  d3 = dplog10q (eps1) - 1.d0
  d4 = dplog10q (eps2) - 1.d0

  if (k <= 2) then
    err = mpreal (1.d0, nwds1)
  elseif (d1 .eq. -999999.d0) then
    err = mpreal (0.d0, nwds1)
  else
    err = c10 ** nint (min (0.d0, max (d1 ** 2 / d2, 2.d0 * d1, d3, d4)))
  endif

!   Output current integral approximation and error estimate, to 60 digits.

  if (ndebug >= 2) then
    write (6, 2) k, nq1, nint (dplog10q (abs (err)))
2   format ('quadts: Iteration',i3,' of',i3,'; est error = 10^',i5, &
      '; approx value =')
    call mpwrite (6, 80, 60, s1)
  endif

  if (k >= 3 .and. iz1 == 0 .and. iz2 == 0) then
    write (6, 3)
3   format ('quadts: Terms too large -- adjust neps2 in call to initqss.')
    goto 140
  endif

  if (k >= 3 .and. err < eps1) then
    write (6, 4) nint (dplog10q (abs (err)))
4   format ('quadts: Estimated error = 10^',i5)
    goto 140
  endif

  if (k >= 3 .and. err < eps2) then
    write (6, 5) nint (dplog10q (abs (err)))
5   format ('quadts: Estimated error = 10^',i5/&
    'Adjust nq1 and neps2 in initqts for greater accuracy.')
    goto 140
  endif
enddo

140 continue

quadts = s1
return
end function quadts

function dplog10q (a)

!   For input MP value a, this routine returns a DP approximation to log10 (a).

use mpmodule
implicit none
integer ia
double precision da, dplog10q, t1
type (mp_real) a

call mpmdi (a, da, ia)
if (da .eq. 0.d0) then
  dplog10q = -999999.d0
else
  dplog10q = log10 (abs (da)) + ia * log10 (2.d0)
endif

100 continue
return
end function dplog10q

subroutine decmdq (a, b, ib)

!   For input MP value a, this routine returns DP b and integer ib such that 
!   a = b * 10^ib, with 1 <= abs (b) < 10 for nonzero a.

use mpmodule
implicit none
integer ia, ib
double precision da, b, t1, xlt
parameter (xlt = 0.3010299956639812d0)
type (mp_real) a

call mpmdi (a, da, ia)
if (da .ne. 0.d0) then
  t1 = xlt * ia + log10 (abs (da))
  ib = t1
  if (t1 .lt. 0.d0) ib = ib - 1
  b = sign (10.d0 ** (t1 - ib), da)
else
  b = 0.d0
  ib = 0
endif

return
end subroutine decmdq
