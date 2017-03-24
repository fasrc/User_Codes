program mpint_test_v2

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

  use mpmodule
  implicit none
  integer i, ndp1, ndp2, neps1, neps2, nq1, nq2, nwds1, nwds2, n1
  parameter (ndp1 = 500, ndp2 = 1000, neps1 = -ndp1, neps2 = -ndp2, &
       nq1 = 11, nq2 = 12 * 2 ** nq1, nwds1 = int (ndp1 / mpdpw + 2), &
       nwds2 = int (ndp2 / mpdpw + 2))
  double precision dplog10q, d1, d2, second, tm0, tm1, tm2
  type (mp_real) err, quades, quadss, quadts, fun01, fun02, fun03, fun04, &
       fun05, fun06, fun07, fun08, fun09, fun10, fun11, fun12, fun13, fun14, &
       fun15, fun16, fun17, fun18, one, t1, t2, t3, t4, wkes(-1:nq2), &
       xkes(-1:nq2), wkss(-1:nq2), xkss(-1:nq2), wkts(-1:nq2), xkts(-1:nq2), zero
  type (mp_real) mppic, mpl02, x1, x2
  external fun01, fun02, fun03, fun04, fun05, fun06, fun07, fun08, &
       fun09, fun10, fun11, fun12, fun13, fun14, fun15, fun16, fun17, fun18, &
       quades, quadss, quadts, second

  !   Check to see if default precision is high enough.

  if (ndp2 > mpipl) then
     write (6, '("Increase default precision in module MPFUNF.")')
     stop
  endif

  !   Compute pi and log(2) to high precision (nwds2 words).

  one = mpreal (1.d0, nwds2)
  zero = mpreal (0.d0, nwds2)
  mppic = mppi (nwds2)
  mpl02 = mplog2 (nwds2)

  write (6, 1) nq1, ndp1, ndp2, neps1, neps2
1 format ('Quadts test:  Quadlevel =',i6/'Digits1 =',i6,'  Digits2 =',i6, &
       '  Epsilon1 =',i6,'  Epsilon2 =',i6)

  !   Initialize quadrature tables wk and xk (weights and abscissas).

  tm0 = second ()
  call initqes (nq1, nq2, nwds1, neps2, wkes, xkes)
  call initqss (nq1, nq2, nwds1, neps2, wkss, xkss)
  call initqts (nq1, nq2, nwds1, neps2, wkts, xkts)
  tm1 = second ()
  tm2 = tm1 - tm0
  write (6, 2) tm1 - tm0
2 format ('Quadrature initialization completed: cpu time =',f12.6)

  !   Begin quadrature tests.

  ! +++ Problem 1 +++
  write (6, 11)
11 format (/'Continuous functions on finite intervals:'//&
        'Problem 1: Int_0^1 t*log(1+t) dt = 1/4'/)
  x1 = zero
  x2 = one
  tm0 = second ()
  t1 = quadts (fun01, x1, x2, nq1, nq2, nwds1, nwds2, neps1, wkts, xkts)
  tm1 = second ()
  tm2 = tm2 + (tm1 - tm0)
  write (6, 3) tm1 - tm0
3 format ('Quadrature completed: CPU time =',f12.6/'Result =')
  call mpwrite (6, ndp1 + 20, ndp1, t1)
  t2 = mpreal (0.25d0, nwds1)
  call decmdq (t2 - t1, d1, n1)
  write (6, 4) d1, n1
4 format ('Actual error =',f10.6,'x10^',i6)

  write(6,*)' '
  write (6, *) 'Exact result: 1/4 to ', ndp1, ' digits ='
  call mpwrite (6, ndp1 + 20, ndp1, t2)
  
  ! +++ Problem 2 +++
  write (6, 12)
12 format (/'Problem 2: Int_0^1 t^2*arctan(t) dt = (pi - 2 + 2*log(2))/12'/)
  x1 = zero
  x2 = one
  tm0 = second ()
  t1 = quadts (fun02, x1, x2, nq1, nq2, nwds1, nwds2, neps1, wkts, xkts)
  tm1 = second ()
  tm2 = tm2 + (tm1 - tm0)
  write (6, 3) tm1 - tm0
  call mpwrite (6, ndp1 + 20, ndp1, t1)
  t2 = (mppic - 2.d0 + 2.d0 * mpl02) / 12.d0
  call decmdq (t2 - t1, d1, n1)
  write (6, 4) d1, n1

  write(6,*)' '
  write (6, *) 'Exact result: (pi - 2 + 2*log(2))/12 to ', ndp1, ' digits ='
  call mpwrite (6, ndp1 + 20, ndp1, t2)

  ! +++ Problem 3 +++
  write (6, 13)
13 format (/'Problem 3: Int_0^(pi/2) e^t*cos(t) dt = 1/2*(e^(pi/2) - 1)'/)
  x1 = zero
  x2 = 0.5d0 * mppic
  tm0 = second ()
  t1 = quadts (fun03, x1, x2, nq1, nq2, nwds1, nwds2, neps1, wkts, xkts)
  tm1 = second ()
  tm2 = tm2 + (tm1 - tm0)
  write (6, 3) tm1 - tm0
  call mpwrite (6, ndp1 + 20, ndp1, t1)
  t2 = 0.5d0 * (exp (0.5d0 * mppic) - 1.d0)
  call decmdq (t2 - t1, d1, n1)
  write (6, 4) d1, n1

  write(6,*)' '
  write (6, *) 'Exact result: 1/2*(e^(pi/2) - 1) to ', ndp1, ' digits ='
  call mpwrite (6, ndp1 + 20, ndp1, t2)

  ! +++ Problem 4 +++
  write (6, 14)
14 format (/ &
        'Problem 4: Int_0^1 arctan(sqrt(2+t^2))/((1+t^2)sqrt(2+t^2)) dt = 5*Pi^2/96'/)
  x1 = zero
  x2 = one
  tm0 = second ()
  t1 = quadts (fun04, x1, x2, nq1, nq2, nwds1, nwds2, neps1, wkts, xkts)
  tm1 = second ()
  tm2 = tm2 + (tm1 - tm0)
  write (6, 3) tm1 - tm0
  call mpwrite (6, ndp1 + 20, ndp1, t1)
  t2 = 5.d0 * mppic**2 / 96.d0
  call decmdq (t2 - t1, d1, n1)
  write (6, 4) d1, n1

  write(6,*)' '
  write (6, *) 'Exact result: 5*Pi^2/96 to ', ndp1, ' digits ='
  call mpwrite (6, ndp1 + 20, ndp1, t2)

  ! +++ Problem 5 +++
  write (6, 15)
15 format (/&
        'Continuous functions on finite intervals, but non-diff at an endpoint:'// &
        'Problem 5: Int_0^1 sqrt(t)*log(t) dt = -4/9'/)
  x1 = zero
  x2 = one
  tm0 = second ()
  t1 = quadts (fun05, x1, x2, nq1, nq2, nwds1, nwds2, neps1, wkts, xkts)
  tm1 = second ()
  tm2 = tm2 + (tm1 - tm0)
  write (6, 3) tm1 - tm0
  call mpwrite (6, ndp1 + 20, ndp1, t1)
  t2 = mpreal (-4.d0, nwds1) / 9.d0
  call decmdq (t2 - t1, d1, n1)
  write (6, 4) d1, n1

  write(6,*)' '
  write (6, *) 'Exact result: -4/9 to ', ndp1, ' digits ='
  call mpwrite (6, ndp1 + 20, ndp1, t2)

  ! +++ Problem 6 +++
  write (6, 16)
16 format (/'Problem 6: Int_0^1 sqrt(1-t^2) dt = pi/4'/)
  x1 = zero
  x2 = one
  tm0 = second ()
  t1 = quadts (fun06, x1, x2, nq1, nq2, nwds1, nwds2, neps1, wkts, xkts)
  tm1 = second ()
  tm2 = tm2 + (tm1 - tm0)
  write (6, 3) tm1 - tm0
  call mpwrite (6, ndp1 + 20, ndp1, t1)
  t2 = 0.25d0 * mppic
  call decmdq (t2 - t1, d1, n1)
  write (6, 4) d1, n1

  write(6,*)' '
  write (6, *) 'Exact result: pi/4 to ', ndp1, ' digits ='
  call mpwrite (6, ndp1 + 20, ndp1, t2)

  ! +++ Problem 7 +++
  write (6, 17)
17 format (/&
        'Functions on finite intervals with integrable singularity at an endpoint:'//&
        'Problem 7: Int_0^1 sqrt(t)/sqrt(1-t^2) dt = 2*sqrt(pi)*gamma(3/4)/gamma(1/4)'/)
  x1 = zero
  x2 = one
  tm0 = second ()
  t1 = quadts (fun07, x1, x2, nq1, nq2, nwds1, nwds2, neps1, wkts, xkts)
  tm1 = second ()
  tm2 = tm2 + (tm1 - tm0)
  write (6, 3) tm1 - tm0
  call mpwrite (6, ndp1 + 20, ndp1, t1)
  t2 = 2.d0 * sqrt (mpreal (mppic, nwds1)) * gamma (mpreal (0.75d0, nwds1)) &
       / gamma (mpreal (0.25d0, nwds1))
  call decmdq (t2 - t1, d1, n1)
  write (6, 4) d1, n1
  
  write(6,*)' '
  write (6, *) 'Exact result: 2*sqrt(pi)*gamma(3/4)/gamma(1/4) to ', ndp1, ' digits ='
  call mpwrite (6, ndp1 + 20, ndp1, t2)

  ! +++ Problem 8 +++
  write (6, 18)
18 format (/'Problem 8: Int_0^1 log(t)^2 dt = 2'/)
  x1 = zero
  x2 = one
  tm0 = second ()
  t1 = quadts (fun08, x1, x2, nq1, nq2, nwds1, nwds2, neps1, wkts, xkts)
  tm1 = second ()
  tm2 = tm2 + (tm1 - tm0)
  write (6, 3) tm1 - tm0
  call mpwrite (6, ndp1 + 20, ndp1, t1)
  t2 = mpreal (2.d0, nwds1)
  call decmdq (t2 - t1, d1, n1)
  write (6, 4) d1, n1

  write(6,*)' '
  write (6, *) 'Exact result: 2 to ', ndp1, ' digits ='
  call mpwrite (6, ndp1 + 20, ndp1, t2)

  ! +++ Problem 9 +++
  write (6, 19)
19 format (/'Problem 9: Int_0^(pi/2) log(cos(t)) dt = -pi*log(2)/2'/)
  x1 = zero
  x2 = 0.5d0 * mppic
  tm0 = second ()
  t1 = quadts (fun09, x1, x2, nq1, nq2, nwds1, nwds2, neps1, wkts, xkts)
  tm1 = second ()
  tm2 = tm2 + (tm1 - tm0)
  write (6, 3) tm1 - tm0
  call mpwrite (6, ndp1 + 20, ndp1, t1)
  t2 = -0.5d0 * mppic * mpl02
  call decmdq (t2 - t1, d1, n1)
  write (6, 4) d1, n1

  write(6,*)' '
  write (6, *) 'Exact result: -pi*log(2)/2 to ', ndp1, ' digits ='
  call mpwrite (6, ndp1 + 20, ndp1, t2)

  ! +++ Problem 10 +++
  write (6, 20)
20 format (/'Problem 10: Int_0^(pi/2) sqrt(tan(t)) dt = pi*sqrt(2)/2'/)
  x1 = zero
  x2 = 0.5d0 * mppic
  tm0 = second ()
  t1 = quadts (fun10, x1, x2, nq1, nq2, nwds1, nwds2, neps1, wkts, xkts)
  tm1 = second ()
  tm2 = tm2 + (tm1 - tm0)
  write (6, 3) tm1 - tm0
  call mpwrite (6, ndp1 + 20, ndp1, t1)
  t2 = 0.5d0 * mppic * sqrt (mpreal (2.d0, nwds1))
  call decmdq (t2 - t1, d1, n1)
  write (6, 4) d1, n1

  write(6,*)' '
  write (6, *) 'Exact result: pi*sqrt(2)/2 to ', ndp1, ' digits ='
  call mpwrite (6, ndp1 + 20, ndp1, t2)

  ! +++ Problem 11 +++
  write (6, 21)
21 format (/&
        'Functions on a semi-infinite interval:'//&
        'Problem 11: Int_0^inf 1/(1+t^2) dt = pi/2'/)
  x1 = zero
  tm0 = second ()
  t1 = quades (fun11, x1, nq1, nq2, nwds1, nwds2, neps1, wkes, xkes)
  tm1 = second ()
  tm2 = tm2 + (tm1 - tm0)
  write (6, 3) tm1 - tm0
  call mpwrite (6, ndp1 + 20, ndp1, t1)
  t2 = 0.5d0 * mppic
  call decmdq (t2 - t1, d1, n1)
  write (6, 4) d1, n1

  write(6,*)' '
  write (6, *) 'Exact result: pi/2 to ', ndp1, ' digits ='
  call mpwrite (6, ndp1 + 20, ndp1, t2)


  ! +++ Problem 12 +++
  write (6, 22)
22 format (/'Problem 12: Int_0^inf e^(-t)/sqrt(t) dt = sqrt(pi)'/)
  x1 = zero
  tm0 = second ()
  t1 = quades (fun12, x1, nq1, nq2, nwds1, nwds2, neps1, wkes, xkes)
  tm1 = second ()
  tm2 = tm2 + (tm1 - tm0)
  write (6, 3) tm1 - tm0
  call mpwrite (6, ndp1 + 20, ndp1, t1)
  t2 = sqrt (mppic)
  call decmdq (t2 - t1, d1, n1)
  write (6, 4) d1, n1

  write(6,*)' '
  write (6, *) 'Exact result: sqrt(pi) to ', ndp1, ' digits ='
  call mpwrite (6, ndp1 + 20, ndp1, t2)


  ! +++ Problem 13 +++
  write (6, 23)
23 format (/'Problem 13: Int_0^inf e^(-t^2/2) dt = sqrt(pi/2)'/)
  x1 = zero
  tm0 = second ()
  t1 = quades (fun13, x1, nq1, nq2, nwds1, nwds2, neps1, wkes, xkes)
  tm1 = second ()
  tm2 = tm2 + (tm1 - tm0)
  write (6, 3) tm1 - tm0
  call mpwrite (6, ndp1 + 20, ndp1, t1)
  t2 = sqrt (0.5d0 * mppic)
  call decmdq (t2 - t1, d1, n1)
  write (6, 4) d1, n1

  write(6,*)' '
  write (6, *) 'Exact result: sqrt(pi/2) to ', ndp1, ' digits ='
  call mpwrite (6, ndp1 + 20, ndp1, t2)

  ! +++Problem 14 +++
  write (6, 24)
24 format (/'Problem 14: Int_0^inf e^(-t)*cos(t) dt = 1/2'/)
  x1 = zero
  tm0 = second ()
  t1 = quades (fun14, x1, nq1, nq2, nwds1, nwds2, neps1, wkes, xkes)
  tm1 = second ()
  tm2 = tm2 + (tm1 - tm0)
  write (6, 3) tm1 - tm0
  call mpwrite (6, ndp1 + 20, ndp1, t1)
  t2 = mpreal (0.5d0, nwds1)
  call decmdq (t2 - t1, d1, n1)
  write (6, 4) d1, n1

  write(6,*)' '
  write (6, *) 'Exact result: 1/2 to ', ndp1, ' digits ='
  call mpwrite (6, ndp1 + 20, ndp1, t2)

  ! +++ Problem 15 +++
  write (6, 25)
25 format (/ &
        'Functions on the entire real line:'// &
        'Problem 15: Int_-inf^inf 1/(1+t^2) dt = Pi'/)
  tm0 = second ()
  t1 = quadss (fun15, nq1, nq2, nwds1, neps1, wkss, xkss)
  tm1 = second ()
  tm2 = tm2 + (tm1 - tm0)
  write (6, 3) tm1 - tm0
  call mpwrite (6, ndp1 + 20, ndp1, t1)
  t2 = mppic
  call decmdq (t2 - t1, d1, n1)
  write (6, 4) d1, n1

  write(6,*)' '
  write (6, *) 'Exact result: pi to ', ndp1, ' digits ='
  call mpwrite (6, ndp1 + 20, ndp1, t2)


  ! +++ Problem 16 +++
  write (6, 26)
26 format (/'Problem 16: Int_-inf^inf 1/(1+t^4) dt = Pi/Sqrt(2)'/)
  tm0 = second ()
  t1 = quadss (fun16, nq1, nq2, nwds1, neps1, wkss, xkss)
  tm1 = second ()
  tm2 = tm2 + (tm1 - tm0)
  write (6, 3) tm1 - tm0
  call mpwrite (6, ndp1 + 20, ndp1, t1)
  t2 = mppic / sqrt (mpreal (2.d0, nwds1))
  call decmdq (t2 - t1, d1, n1)
  write (6, 4) d1, n1

  write(6,*)' '
  write (6, *) 'Exact result: pi/sqrt(2) to ', ndp1, ' digits ='
  call mpwrite (6, ndp1 + 20, ndp1, t2)

  ! +++ problem 17 +++
  write (6, 27)
27 format (/'Problem 17: Int_-inf^inf e^(-t^2/2) dt = sqrt (2*Pi)'/)
  tm0 = second ()
  t1 = quadss (fun17, nq1, nq2, nwds1, neps1, wkss, xkss)
  tm1 = second ()
  tm2 = tm2 + (tm1 - tm0)
  write (6, 3) tm1 - tm0
  call mpwrite (6, ndp1 + 20, ndp1, t1)
  t2 = sqrt (2.d0 * mppic)
  call decmdq (t2 - t1, d1, n1)
  write (6, 4) d1, n1

  write(6,*)' '
  write (6, *) 'Exact result: sqrt(2*pi) to ', ndp1, ' digits ='
  call mpwrite (6, ndp1 + 20, ndp1, t2)

  !+++ Problem 18 +++
  write (6, 28)
28 format (/'Problem 18: Int_-inf^inf e^(-t^2/2) cos(t) dt = sqrt (2*Pi/e)'/)
  tm0 = second ()
  t1 = quadss (fun18, nq1, nq2, nwds1, neps1, wkss, xkss)
  tm1 = second ()
  tm2 = tm2 + (tm1 - tm0)
  write (6, 3) tm1 - tm0
  call mpwrite (6, ndp1 + 20, ndp1, t1)
  t2 = sqrt (2.d0 * mppic / exp (mpreal (1.d0, nwds1)))
  call decmdq (t2 - t1, d1, n1)
  write (6, 4) d1, n1

  write(6,*)' '
  write (6, *) 'Exact result: sqrt(2*pi/e) to ', ndp1, ' digits ='
  call mpwrite (6, ndp1 + 20, ndp1, t2)

  write (6, 99) tm2
99 format ('Total CPU time =',f12.6)

  stop 'End of program.'
end program mpint_test_v2


! +++ FUNCTIONS TO INTIGRATE +++
function fun01 (t, nwds1, nwds2)
  !   fun01(t) = t * log(1+t)
  use mpmodule
  implicit none
  integer nwds1, nwds2
  type (mp_real) fun01, t1, t2
  type (mp_real) t
  t1 = mpreal (t, nwds1)
  fun01 = t1 * log (1.d0 + t1)
  return
end function fun01

function fun02 (t, nwds1, nwds2)
  !   fun02(t) = t^2 * arctan(t)
  use mpmodule
  implicit none
  integer nwds1, nwds2
  type (mp_real) fun02, t1
  type (mp_real) t
  t1 = mpreal (t, nwds1)
  fun02 = t1 ** 2 * atan (t1)
  return
end function fun02

function fun03 (t, nwds1, nwds2)
!   fun03(t) = e^t * cos(t)
  use mpmodule
  implicit none
  integer nwds1, nwds2
  type (mp_real) fun03, t1
  type (mp_real) t

  t1 = mpreal (t, nwds1)
  fun03 = exp(t1) * cos(t1)
  return
end function fun03

function fun04 (t, nwds1, nwds2)
  !   fun04(t) = arctan(sqrt(2+t^2))/((1+t^2)sqrt(2+t^2))
  
  use mpmodule
  implicit none
  integer nwds1, nwds2
  type (mp_real) fun04, t1, t2
  type (mp_real) t
  t1 = mpreal (t, nwds1)
  t2 = sqrt (2.d0 + t1**2)
  fun04 = atan(t2) / ((1.d0 + t1**2) * t2)
  return
end function fun04

function fun05 (t, nwds1, nwds2)
  !    fun05(t) = sqrt(t)*log(t)
  use mpmodule
  implicit none
  integer nwds1, nwds2
  type (mp_real) fun05, t1
  type (mp_real) t
  t1 = mpreal (t, nwds1)
  fun05 = sqrt (t1) * log (t1)
  return
end function fun05

function fun06 (t, nwds1, nwds2)
  !    fun06(t) = sqrt(1-t^2)
  
  use mpmodule
  implicit none
  integer nwds1, nwds2
  type (mp_real) fun06, t1, t2
  type (mp_real) t
  t1 = mpreal (t, nwds1)
  t2 = mpreal (1.d0 - t**2, nwds1)
  fun06 = sqrt (t2)
  return
end function fun06

function fun07 (t, nwds1, nwds2)
  !   fun07(t) = sqrt (t) / sqrt(1-t^2)
  use mpmodule
  implicit none
  integer nwds1, nwds2
  type (mp_real) fun07, t1, t2
  type (mp_real) t
  !   The subtraction to compute t2 must be performed using high precision
  !   (nwds2), but after the subtraction its low precision value is fine.
  t1 = mpreal (t, nwds1)
  t2 = mpreal (1.d0 - t, nwds1)
  fun07 = sqrt (t1) / sqrt (t2 * (1.d0 + t1))
  return
end function fun07

function fun08 (t, nwds1, nwds2)
  !   fun08(t) = log(t)^2
  use mpmodule
  implicit none
  integer nwds1, nwds2
  type (mp_real) fun08, t1
  type (mp_real) t
  t1 = mpreal (t, nwds1)
  fun08 = log (t1) ** 2
  return
end function fun08

function fun09 (t, nwds1, nwds2)
  !   fun09(t) = log (cos (t))
  use mpmodule
  implicit none
  integer nwds1, nwds2
  type (mp_real) fun09, pi, t1, t2, t3, t4
  type (mp_real) t
  t1 = mpreal (t, nwds1)
  pi = mppi (nwds2)
  t3 = mpreal (0.25d0 * pi, nwds1)
  t2 = mpreal (0.5d0 * pi - t, nwds1)
  if (t1 < t3) then
     t4 = cos (t1)
  else
     t4 = sin (t2)
  endif
  fun09 = log (t4)
  return
end function fun09

function fun10 (t, nwds1, nwds2)
  !   fun10(t) = sqrt(tan(t))
  use mpmodule
  implicit none
  integer nwds1, nwds2
  type (mp_real) fun10, pi, t1, t2, t3, t4
  type (mp_real) t
  t1 = mpreal (t, nwds1)
  pi = mppi (nwds2)
  t3 = mpreal (0.25d0 * pi, nwds1)
  t2 = mpreal (0.5d0 * pi - t, nwds1)
  if (t1 < t3) then
     fun10 = sqrt (tan (t1))
  else
     fun10 = 1.d0 / sqrt (tan (t2))
  endif
  return
end function fun10

function fun11 (t, nwds1, nwds2)  
  !   1/(1 + t^2) on (0, Inf).
  use mpmodule
  implicit none
  integer nwds1, nwds2
  type (mp_real) fun11, t1
  type (mp_real) t
  t1 = mpreal (t, nwds1)
  fun11 = 1.d0 / (1.d0 + t1 ** 2)
  return
end function fun11

function fun12 (t, nwds1, nwds2)
  !   e^(-t)/sqrt(t) on (0, inf).  
  use mpmodule
  implicit none
  integer nwds1, nwds2
  type (mp_real) fun12, t1, t2
  type (mp_real) t
  !   The subtraction to compute t2 must be performed using high precision
  !   (nwds2), but after the subtraction its low precision value is fine.
  t1 = mpreal (t, nwds1)
  fun12 = exp (-t1) / sqrt (t1)
  return
end function fun12

function fun13 (t, nwds1, nwds2)
  !   e^(-t^2/2) on (0, inf).
  use mpmodule
  implicit none
  integer nwds1, nwds2
  double precision dig1
  type (mp_real) fun13, t1, t2
  type (mp_real) t
  t1 = mpreal (t, nwds1)
  fun13 = exp (-0.5d0 * t1 ** 2)
  return
end function fun13

function fun14 (t, nwds1, nwds2)  
  !  e^(-t) cos(t) on (0, inf).
  use mpmodule
  implicit none
  integer nwds1, nwds2
  double precision dig1
  type (mp_real) fun14, t1, t2
  type (mp_real) t
  t1 = mpreal (t, nwds1)
  fun14 = exp (-t1) * cos (t1)
  return
end function fun14

function fun15 (t, nwds1)
  use mpmodule
  implicit none
  integer nwds1
  type (mp_real) t, fun15
  fun15 = 1.d0 / (1.d0 + t**2)
  return
end function fun15

function fun16 (t, nwds1)
  use mpmodule
  implicit none
  integer nwds1
  type (mp_real) t, fun16
  fun16 = 1.d0 / (1.d0 + t**4)
  return
end function fun16

function fun17 (t, nwds1)
  use mpmodule
  implicit none
  integer nwds1
  type (mp_real) t, fun17  
  fun17 = exp (-0.5d0 * t**2)
  return
end function fun17

function fun18 (t, nwds1)
  use mpmodule
  implicit none
  integer nwds1
  type (mp_real) t, fun18
  fun18 = exp (-0.5d0 * t**2) * cos (t)
  return
end function fun18
