!===============================================================================
! Program: mpint_test.f90
!===============================================================================
program mpint_test
  use mpmodule
  implicit none
  integer i, ndp1, ndp2, neps1, neps2, nq1, nq2, nwds1, nwds2, n1
  parameter (ndp1 = 1500, ndp2 = 1800, neps1 = -ndp1, neps2 = -ndp2,    &
       nq1 = 11, nq2 = 12 * 2 ** nq1, nwds1 = int (ndp1 / mpdpw + 2),  &
       nwds2 = int (ndp2 / mpdpw + 2))
  double precision dplog10q, d1, d2, second, tm0, tm1, tm2
  type (mp_real) err, quades, quadss, quadts, fun01, fun02, fun03, fun04,    &
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
3 format ('Quadrature completed: CPU time =',f12.6/'Result =')
4 format ('Actual error =',f10.6,'x10^',i6)
  
  !   Begin quadrature tests.

  ! +++ Problem 1 +++
  write (6, 16)
16 format (/'Problem 1: Int_0^1 sqrt(1-t^2) dt = pi/4'/)
  x1 = zero
  x2 = one
  tm0 = second ()
  t1 = quadts (fun01, x1, x2, nq1, nq2, nwds1, nwds2, neps1, wkts, xkts)
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

  stop
end program mpint_test

! +++ Functions to integrate +++

function fun01 (t, nwds1, nwds2)
  !    fun01(t) = sqrt(1-t^2)
  use mpmodule
  implicit none
  integer nwds1, nwds2
  type (mp_real) fun01, t1, t2
  type (mp_real) t
  t1 = mpreal (t, nwds1)
  t2 = mpreal (1.d0 - t**2, nwds1)
  fun01 = sqrt (t2)
  return
end function fun01

