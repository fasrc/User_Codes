!*****************************************************************************

!  MPFUN-MPFR: An MPFR-based arbitrary precision computation package
!  Language interface module (module MPFUNG)
!  Variant 1: Precision level specifications are *optional*; no real*16 support.
!  Search for !> for variant differences.

!  Revision date:  6 Feb 2017

!  AUTHOR:
!     David H. Bailey
!     Lawrence Berkeley National Lab (retired) and University of California, Davis
!     Email: dhbailey@lbl.gov

!  COPYRIGHT AND DISCLAIMER:
!  All software in this package (c) 2017 David H. Bailey.
!  By downloading or using this software you agree to the copyright, disclaimer
!  and license agreement in the accompanying file DISCLAIMER.txt.

!  PURPOSE OF PACKAGE:
!    This package permits one to perform floating-point computations (real and
!    complex) to arbitrarily high numeric precision, by making only relatively
!    minor changes to existing Fortran-90 programs.  All basic arithmetic
!    operations and transcendental functions are supported, together with several
!    special functions.

!    This version differs from the MPFUN-Fort version by the same author in that
!    it is based on MPFR, which presently is the fastest available low-level
!    package for high-precision floating-point computation.  Thus most user
!    applications typically run 3X faster.  In addition, the developers of the
!    MPFR package have taken considerable pains to ensure that the many functions
!    return correctly rounded (to the last bit) results for each input.  At the
!    Fortran user level, application codes written for MPFUN-Fort may be compiled
!    and executed with MPFUN-MPFR --i.e., MPFUN-MPFR is "plug compatible" with
!    MPFUN-Fort.

!  DOCUMENTATION:
!    A detailed description of this package, and instructions for compiling
!    and testing this program on various specific systems are included in the
!    README file accompanying this package, and, in more detail, in the
!    following technical paper:

!    David H. Bailey, "MPFUN2015: A thread-safe arbitrary precision package," 
!    available at http://www.davidhbailey.com/dhbpapers/mpfun2015.pdf.
  
!  DESCRIPTION OF THIS MODULE (MPFUNG):
!    This module contains all high-level Fortran-90 language interfaces.
!    There are two variants of this module, which take different approaches
!    to dynamically changing working precision within an application.  Variant
!    1 allows mixed-mode assignments, and does not require precision level
!    specifications in certain functions, whereas variant 2 does not permit
!    mixed-mode assignments, and requires precision level specifications in
!    certain functions.  See documentation for details.
 
module mpfung
use mpfuna
use mpfunf
implicit none
  
!   The mp_real and mp_complex datatypes are defined here:

type mp_real
  sequence
  integer(8) mpr(0:mpwds+5)
end type
type mp_complex
  sequence
  integer(8) mpc(0:2*mpwds+11)
end type

!  Assignments and the five arithmetic operators:

private &
  mp_eqrr, mp_eqdr, mp_eqrd, mp_eqir, mp_eqri, mp_eqra, mp_eqrz, &
  mp_eqzr, mp_eqzz, mp_eqdz, mp_eqzd, mp_eqdcz, mp_eqzdc, &
  mp_addrr, mp_adddr, mp_addrd, mp_addir, mp_addri, mp_addzz, &
  mp_adddz, mp_addzd, mp_adddcz, mp_addzdc, mp_addrz, mp_addzr, &
  mp_subrr, mp_subdr, mp_subrd, mp_subir, mp_subri, mp_subzz, &
  mp_subdz, mp_subzd, mp_subdcz, mp_subzdc, mp_subrz, mp_subzr, &
  mp_negr, mp_negz, &
  mp_mulrr, mp_muldr, mp_mulrd, mp_mulir, mp_mulri, mp_mulzz, &
  mp_muldz, mp_mulzd, mp_muldcz, mp_mulzdc, mp_mulrz, mp_mulzr, &
  mp_divrr, mp_divdr, mp_divrd, mp_divir, mp_divri, mp_divzz, &
  mp_divdz, mp_divzd, mp_divdcz, mp_divzdc, mp_divrz, mp_divzr, &
  mp_expri, mp_exprr, mp_expzi, mp_expzz, mp_exprz, mp_expzr

!  The six comparison tests:

private &
  mp_eqtrr, mp_eqtdr, mp_eqtrd, mp_eqtir, mp_eqtri, mp_eqtzz, &
  mp_eqtdz, mp_eqtzd, mp_eqtdcz, mp_eqtzdc, mp_eqtrz, mp_eqtzr, &
  mp_netrr, mp_netdr, mp_netrd, mp_netir, mp_netri, mp_netzz, &
  mp_netdz, mp_netzd, mp_netdcz, mp_netzdc, mp_netrz, mp_netzr, &
  mp_letrr, mp_letdr, mp_letrd, mp_letir, mp_letri, &
  mp_getrr, mp_getdr, mp_getrd, mp_getir, mp_getri, &
  mp_lttrr, mp_lttdr, mp_lttrd, mp_lttir, mp_lttri, &
  mp_gttrr, mp_gttdr, mp_gttrd, mp_gttir, mp_gttri

!  Algebraic, transcendental and type conversion functions:
  
private &
  mp_abrt, mp_absr, mp_absz, mp_acos, mp_acosh, mp_agm, mp_aimag, &
  mp_aint, mp_airy, mp_anint, mp_asin, mp_asinh, mp_atan, &
  mp_atanh, mp_atan2, mp_ator1, mp_atorn, mp_berne, mp_besselj, &
  mp_bessel_j0, mp_bessel_j1, mp_bessel_jn, mp_bessel_y0, mp_bessel_y1, &
  mp_bessel_yn, mp_checkdp, mp_ccos, mp_cexp, mp_clog, mp_conjg, &
  mp_cos, mp_cosh, mp_csin, mp_csqrt, mp_cssh, mp_cssn, mp_dctoz, &
  mp_dctoz2, mp_digamma, mp_dtor, mp_dtor2, mp_eform, mp_egamma, &
  mp_erf, mp_erfc, mp_exp, mp_expint, mp_fixlocr, mp_fixlocz, &
  mp_fform, mp_gamma, mp_gammainc, mp_hypot, mp_init, mp_initvr, &
  mp_initvz, mp_inpr, mp_inpz, mp_log, mp_log_gamma, mp_log2, mp_max, &
  mp_mdi, mp_min, mp_nrt, mp_outr, mp_outz, mp_pi, mp_polylog, &
  mp_prodd, mp_qtor, mp_quotd, mp_readr1, mp_readr2, mp_readr3, &
  mp_readr4, mp_readr5, mp_readz1, mp_readz2, mp_readz3, mp_readz4, &
  mp_readz5, mp_rtod, mp_rtoq, mp_rtor, mp_rtoz, mp_setwp, mp_sign, &
  mp_sin, mp_sinh, mp_sqrt, mp_tan, mp_tanh, mp_wprec, mp_wprecz, &
  mp_writer, mp_writez, mp_zeta, mp_zetaem, mp_ztodc, mp_ztor, &
  mp_ztor2, mp_ztoz

!  Operator extension interface blocks:

interface assignment (=)
  module procedure mp_eqrr
  module procedure mp_eqdr
  module procedure mp_eqir
  module procedure mp_eqrz
  module procedure mp_eqzr
  module procedure mp_eqzz
  module procedure mp_eqdz
  module procedure mp_eqdcz

!>  In variant #1, the next five module procedure lines are uncommented;
!>  In variant #2 they are commented out.

  module procedure mp_eqrd
  module procedure mp_eqri
  module procedure mp_eqra
  module procedure mp_eqzd
  module procedure mp_eqzdc
end interface

interface operator (+)
  module procedure mp_addrr
  module procedure mp_adddr
  module procedure mp_addrd
  module procedure mp_addir
  module procedure mp_addri
  module procedure mp_addzz
  module procedure mp_adddz
  module procedure mp_addzd
  module procedure mp_adddcz
  module procedure mp_addzdc
  module procedure mp_addrz
  module procedure mp_addzr
end interface

interface operator (-)
  module procedure mp_subrr
  module procedure mp_subdr
  module procedure mp_subrd
  module procedure mp_subir
  module procedure mp_subri
  module procedure mp_subzz
  module procedure mp_subdz
  module procedure mp_subzd
  module procedure mp_subdcz
  module procedure mp_subzdc
  module procedure mp_subrz
  module procedure mp_subzr
  module procedure mp_negr
  module procedure mp_negz
end interface

interface operator (*)
  module procedure mp_mulrr
  module procedure mp_muldr
  module procedure mp_mulrd
  module procedure mp_mulir
  module procedure mp_mulri
  module procedure mp_mulzz
  module procedure mp_muldz
  module procedure mp_mulzd
  module procedure mp_muldcz
  module procedure mp_mulzdc
  module procedure mp_mulrz
  module procedure mp_mulzr
end interface

interface operator (/)
  module procedure mp_divrr
  module procedure mp_divdr
  module procedure mp_divrd
  module procedure mp_divir
  module procedure mp_divri
  module procedure mp_divzz
  module procedure mp_divdz
  module procedure mp_divzd
  module procedure mp_divdcz
  module procedure mp_divzdc
  module procedure mp_divrz
  module procedure mp_divzr
end interface

interface operator (**)
   module procedure mp_expri
   module procedure mp_exprr
   module procedure mp_expzi
   module procedure mp_expzz
   module procedure mp_exprz
   module procedure mp_expzr
end interface

interface operator (.eq.)
  module procedure mp_eqtrr
  module procedure mp_eqtdr
  module procedure mp_eqtrd
  module procedure mp_eqtir
  module procedure mp_eqtri
  module procedure mp_eqtzz
  module procedure mp_eqtdz
  module procedure mp_eqtzd
  module procedure mp_eqtdcz
  module procedure mp_eqtzdc
  module procedure mp_eqtrz
  module procedure mp_eqtzr
end interface

interface operator (.ne.)
  module procedure mp_netrr
  module procedure mp_netdr
  module procedure mp_netrd
  module procedure mp_netir
  module procedure mp_netri
  module procedure mp_netzz
  module procedure mp_netdz
  module procedure mp_netzd
  module procedure mp_netdcz
  module procedure mp_netzdc
  module procedure mp_netrz
  module procedure mp_netzr
end interface

interface operator (.le.)
  module procedure mp_letrr
  module procedure mp_letdr
  module procedure mp_letrd
  module procedure mp_letir
  module procedure mp_letri
end interface

interface operator (.ge.)
  module procedure mp_getrr
  module procedure mp_getdr
  module procedure mp_getrd
  module procedure mp_getir
  module procedure mp_getri
end interface

interface operator (.lt.)
  module procedure mp_lttrr
  module procedure mp_lttdr
  module procedure mp_lttrd
  module procedure mp_lttir
  module procedure mp_lttri
end interface

interface operator (.gt.)
  module procedure mp_gttrr
  module procedure mp_gttdr
  module procedure mp_gttrd
  module procedure mp_gttir
  module procedure mp_gttri
end interface

!  MP generic function interface blogs, listed alphabetically by interface name:

interface abs
  module procedure mp_absr
  module procedure mp_absz
end interface

interface acos
  module procedure mp_acos
end interface

interface acosh
  module procedure mp_acosh
end interface

interface agm
  module procedure mp_agm
end interface

interface aimag
  module procedure mp_aimag
end interface

interface aint
  module procedure mp_aint
end interface

interface airy
  module procedure mp_airy
end interface

interface anint
  module procedure mp_anint
end interface

interface asin
  module procedure mp_asin
end interface

interface asinh
  module procedure mp_asinh
end interface

interface atan
  module procedure mp_atan
end interface

interface atanh
  module procedure mp_atanh
end interface

interface atan2
  module procedure mp_atan2
end interface

interface berne
  module procedure mp_berne
end interface

interface besselj
  module procedure mp_besselj
end interface

interface bessel_j0
  module procedure mp_bessel_j0
end interface

interface bessel_j1
  module procedure mp_bessel_j1
end interface

interface bessel_jn
  module procedure mp_bessel_jn
end interface

interface bessel_y0
  module procedure mp_bessel_y0
end interface

interface bessel_y1
  module procedure mp_bessel_y1
end interface

interface bessel_yn
  module procedure mp_bessel_yn
end interface

interface conjg
  module procedure mp_conjg
end interface

interface cos
  module procedure mp_cos
  module procedure mp_ccos
end interface

interface cosh
  module procedure mp_cosh
end interface

interface dble
  module procedure mp_rtod
end interface

interface dcmplx
  module procedure mp_ztodc
end interface

interface digamma
  module procedure mp_digamma
end interface

interface erf
  module procedure mp_erf
end interface

interface erfc
  module procedure mp_erfc
end interface

interface exp
  module procedure mp_exp 
  module procedure mp_cexp
end interface

interface expint
  module procedure mp_expint
end interface

interface gamma
  module procedure mp_gamma
end interface

interface gammainc
  module procedure mp_gammainc
end interface

interface hypot
  module procedure mp_hypot
end interface

interface log
  module procedure mp_log
  module procedure mp_clog
end interface

interface log_gamma
  module procedure mp_log_gamma
end interface

interface max
  module procedure mp_max
end interface

interface min
  module procedure mp_min
end interface

interface mpcmplx
  module procedure mp_dctoz
  module procedure mp_rtoz
  module procedure mp_ztoz
end interface

interface mpcmplxdc
  module procedure mp_dctoz2
end interface

interface mpcssh
  module procedure mp_cssh
end interface

interface mpcssn
  module procedure mp_cssn
end interface

interface mpeform
  module procedure mp_eform
end interface

interface mpegamma
  module procedure mp_egamma
end interface

interface mpfform
  module procedure mp_fform
end interface

interface mpinit
  module procedure mp_init
end interface

interface mplog2
  module procedure mp_log2
end interface

interface mpmdi
  module procedure mp_mdi
end interface

interface mpnrt
  module procedure mp_nrt
end interface

interface mppi
  module procedure mp_pi
end interface

interface mpprodd
  module procedure mp_prodd
end interface

interface mpquotd
  module procedure mp_quotd
end interface

interface mpread
  module procedure mp_readr1
  module procedure mp_readr2
  module procedure mp_readr3
  module procedure mp_readr4
  module procedure mp_readr5
  module procedure mp_readz1
  module procedure mp_readz2
  module procedure mp_readz3
  module procedure mp_readz4
  module procedure mp_readz5
end interface

interface mpreal
  module procedure mp_ator1
  module procedure mp_atorn
  module procedure mp_dtor
  module procedure mp_rtor
  module procedure mp_ztor

!>  If real*16 is supported, uncomment this line; otherwise commented.
!>  Real*16 support is not available yet with MPFR.

!  module procedure mp_qtor
end interface

interface mpreald
  module procedure mp_dtor2
end interface

interface mpwprec
  module procedure mp_wprec
  module procedure mp_wprecz
end interface

interface mpwrite
  module procedure mp_writer
  module procedure mp_writez
end interface

interface polylog
  module procedure mp_polylog
end interface

!>  If real*16 is supported, uncomment these three lines; otherwise commented.
!>  Real*16 support is not available yet with MPFR.

! interface qreal
!   module procedure mp_rtoq
! end interface

interface sign
  module procedure mp_sign
end interface

interface sin
  module procedure mp_sin
  module procedure mp_csin
end interface

interface sinh
  module procedure mp_sinh
end interface

interface sqrt
  module procedure mp_sqrt
  module procedure mp_csqrt
end interface

interface tan
  module procedure mp_tan
end interface

interface tanh
  module procedure mp_tanh
end interface

interface zeta
  module procedure mp_zeta
end interface

interface zetaem
  module procedure mp_zetaem
end interface

contains

!   This routine terminates execution.  Users may wish to replace the
!   default STOP with a call to a system routine that provides a traceback.

  subroutine mp_abrt (ier)
    implicit none
    integer ier
    write (mpldb, 1) ier
1   format ('*** MP_ABRT: Execution terminated, error code =',i4)
    stop
  end subroutine

!  This routine outputs an error message if iprec exceeds mpwds.

  function mp_setwp (iprec)
    integer mp_setwp
    integer, intent (in):: iprec
    if (iprec > mpwds) then
      write (mpldb, 1)
1       format ( &
        '*** MP_SETWP: requested precision level exceeds default precision.'/ &
        'Increase default precision in module MPFUNF.')
      call mp_abrt (98)
    endif
    mp_setwp = iprec
  end function

!  This routine checks if the input double precision variable has more than
!  40 significant bits; if so, an error message is output and mp_abrt is called.

  subroutine mp_checkdp (da)
    double precision, intent (in):: da
    double precision d1, d2
    d1 = mpb13x * abs (da)
    d2 = abs (abs (da) + d1) - abs (d1)
    if (d2 /= abs (da)) then
      write (mpldb, 1) da
1     format ('*** MP_CHECKDP: DP value has more than 40 significant bits:', &
      1p,d25.15/'and thus very likely represents an unintended loss of accuracy.'/ &
      'Fix the issue, or else use functions mpprodd, mpquotd, mpreald or mpcmplxdc.'/ &
      'See documentation for details.')
      call mp_abrt (82)
    endif
  end subroutine

!  These two routines are used to initialize scratch and output MP variables
!  in the routines below. The working precision level of the variable to nbt bits,
!  and the value is set to the "NAN" of MPFR.

  subroutine mp_initvr (ra, nbt)
    implicit none
    type (mp_real), intent (out):: ra
    integer, intent (in):: nbt
    ra%mpr(0) = mpwds6
    ra%mpr(1) = nbt
    ra%mpr(2) = 1
    ra%mpr(3) = mpnan
    ra%mpr(4) = loc (ra%mpr(4)) + 8
    return
  end subroutine
  
  subroutine mp_initvz (za, nbt)
    implicit none
    type (mp_complex), intent (out):: za
    integer, intent (in):: nbt
    integer l1
    l1 = mpwds6
    za%mpc(0) = mpwds6
    za%mpc(1) = nbt
    za%mpc(2) = 1
    za%mpc(3) = mpnan
    za%mpc(4) = loc (za%mpc(4)) + 8
    za%mpc(l1) = mpwds6
    za%mpc(l1+1) = nbt
    za%mpc(l1+2) = 1
    za%mpc(l1+3) = mpnan
    za%mpc(l1+4) = loc (za%mpc(l1+4)) + 8
    return
  end subroutine

!  The next two subroutines are needed for most of the routines below, because
!  temporary multiprecision variables generated by Fortran compiler merely 
!  copy the array, but do not correct the pointer in index 4 of the array.
!  Furthermore, calling one of these two subroutines, rather than simply
!  including the line or two of code below directly in the routine, avoids
!  error messages resulting from a conflict with the "intent (in)" attribute.

  subroutine mp_fixlocr (ra)
    implicit none
    type (mp_real):: ra
    ra%mpr(4) = loc (ra%mpr(4)) + 8
    return
  end subroutine
    
  subroutine mp_fixlocz (za)
    implicit none
    integer l1
    type (mp_complex):: za
    l1 = za%mpc(0)
    za%mpc(4) = loc (za%mpc(4)) + 8
    za%mpc(l1+4) = loc (za%mpc(l1+4)) + 8
    return
  end subroutine

!  Assignment routines:

  subroutine mp_eqrr (ra, rb)
    implicit none
    type (mp_real), intent (out):: ra
    type (mp_real), intent (in):: rb
    integer mpnwbt
    call mp_fixlocr (rb)
    mpnwbt = min (rb%mpr(1), mpwdsbt)
    call mp_initvr (ra, mpnwbt)
    call mpfr_set (ra%mpr(1), rb%mpr(1), %val(mprnd))
    return
  end subroutine

  subroutine mp_eqdr (da, rb)
    implicit none
    double precision, intent (out):: da
    type (mp_real), intent (in):: rb
    double precision mpfr_get_d
    external mpfr_get_d
    call mp_fixlocr (rb)
    da = mpfr_get_d (rb%mpr(1), %val(mprnd))
    return
  end subroutine

  subroutine mp_eqrd (ra, db)
    implicit none
    type (mp_real), intent (out):: ra
    double precision, intent (in):: db
    integer mpnwbt
    mpnwbt = mpwdsbt
    call mp_checkdp (db)
    call mp_initvr (ra, mpnwbt)
    call mpfrsetd (ra%mpr(1), db, mprnd)
    return
  end subroutine

  subroutine mp_eqir (ia, rb)
    implicit none
    integer, intent (out):: ia
    type (mp_real), intent (in):: rb
    double precision da
    double precision mpfr_get_d
    external mpfr_get_d
    call mp_fixlocr (rb)
    da = mpfr_get_d (rb%mpr(1), %val(mprnd))
    ia = da
    return
  end subroutine

  subroutine mp_eqri (ra, ib)
    implicit none
    type (mp_real), intent (out):: ra
    integer, intent (in):: ib
    double precision db
    integer mpnwbt
    mpnwbt = mpwdsbt
    db = ib
    call mp_checkdp (db)
    call mp_initvr (ra, mpnwbt)
    call mpfrsetd (ra%mpr(1), db, mprnd)
    return
  end subroutine

  subroutine mp_eqra (ra, ab)
    implicit none
    type (mp_real), intent (out):: ra
    character(*), intent (in):: ab
    character(1) :: chr1(len(ab)+1)
    integer i, l1, mpnw
    integer mpnwbt
    mpnwbt = mpwdsbt
    call mp_initvr (ra, mpnwbt)
    l1 = len (ab)
    do i = 1, l1
      if (ab(i:i) == 'D' .or. ab(i:i) == 'd') then
        chr1(i) = 'e'
      else
        chr1(i) = ab(i:i)
      endif
    enddo
    chr1(l1+1) = char(0)
    call mpfrsetstr (ra%mpr(1), chr1, mprnd)
    return
  end subroutine

  subroutine mp_eqzz (za, zb)
    implicit none
    type (mp_complex), intent (out):: za
    type (mp_complex), intent (in):: zb
    integer l1, l2, mpnwbt
    l1 = zb%mpc(0)
    call mp_fixlocz (zb)
    mpnwbt = max (zb%mpc(1), zb%mpc(l1+1))
    mpnwbt = min (mpnwbt, mpwdsbt)
    l2 = mpwds6
    call mp_initvz (za, mpnwbt)
    call mpfr_set (za%mpc(1), zb%mpc(1), %val(mprnd))
    call mpfr_set (za%mpc(l2+1), zb%mpc(l1+1), %val(mprnd))
    return
  end subroutine

  subroutine mp_eqdz (da, zb)
    implicit none
    double precision, intent (out):: da
    type (mp_complex), intent (in):: zb
    double precision mpfr_get_d
    call mp_fixlocz (zb)
    da = mpfr_get_d (zb%mpc(1), %val(mprnd))
    return
  end subroutine

  subroutine mp_eqzd (za, db)
    implicit none
    type (mp_complex), intent (out):: za
    double precision, intent (in):: db
    double precision d1
    integer l2, mpnwbt
    mpnwbt = mpwdsbt
    l2 = mpwds6
    d1 = 0.d0
    call mp_initvz (za, mpnwbt)
    call mp_checkdp (db)
    call mpfrsetd (za%mpc(1), db, mprnd)
    call mpfrsetd (za%mpc(l2+1), d1, mprnd)
    return
  end subroutine

  subroutine mp_eqdcz (dca, zb)
    implicit none
    complex (kind (0.d0)), intent (out):: dca
    type (mp_complex), intent (in):: zb
    integer l1, n1, n2
    double precision d1, d2
    double precision mpfr_get_d
    l1 = zb%mpc(0)
    call mp_fixlocz (zb)
    d1 = mpfr_get_d (zb%mpc(1), %val(mprnd))
    d2 = mpfr_get_d (zb%mpc(l1+1), %val(mprnd))
    dca = dcmplx (d1, d2)
    return
  end subroutine

  subroutine mp_eqzdc (za, dcb)
    implicit none
    type (mp_complex), intent (out):: za
    complex (kind (0.d0)), intent (in):: dcb
    integer l2, mpnwbt
    mpnwbt = mpwdsbt
    l2 = mpwds6
    call mp_initvz (za, mpnwbt)
    call mp_checkdp (dble (dcb))
    call mp_checkdp (aimag (dcb))
    call mpfrsetd (za%mpc(1), dble (dcb), mprnd)
    call mpfrsetd (za%mpc(l2+1), aimag (dcb), mprnd)
    return
  end subroutine

  subroutine mp_eqrz (ra, zb)
    implicit none
    type (mp_real), intent (out):: ra
    type (mp_complex), intent (in):: zb
    integer l1, mpnwbt
    l1 = zb%mpc(0)
    call mp_fixlocz (zb)
    mpnwbt = max (zb%mpc(1), zb%mpc(l1+1))
    mpnwbt = min (mpnwbt, mpwdsbt)
    call mp_initvr (ra, mpnwbt)
    call mpfr_set (ra%mpr(1), zb%mpc(1), %val(mprnd))
    return
  end subroutine

  subroutine mp_eqzr (za, rb)
    implicit none
    type (mp_complex), intent (out):: za
    type (mp_real), intent (in):: rb
    type (mp_real) r1
    integer l2, mpnwbt
    call mp_fixlocr (rb)
    mpnwbt = min (rb%mpr(1), mpwdsbt)
    l2 = mpwds6
    call mp_initvz (za, mpnwbt)
    call mpfr_set (za%mpc(1), rb%mpr(1), %val(mprnd))
    call mpfr_set_zero (za%mpc(l2+1), %val(0))    
    return
  end subroutine

!  Addition routines:

  function mp_addrr (ra, rb)
    implicit none
    type (mp_real):: mp_addrr
    type (mp_real), intent (in):: ra, rb
    integer mpnwbt
    call mp_fixlocr (ra)
    call mp_fixlocr (rb)
    mpnwbt = max (ra%mpr(1), rb%mpr(1))
    mpnwbt = min (mpnwbt, mpwdsbt)
    call mp_initvr (mp_addrr, mpnwbt)
    call mpfr_add (mp_addrr%mpr(1), ra%mpr(1), rb%mpr(1), %val(mprnd))
    return
  end function

  function mp_adddr (da, rb)
    implicit none
    type (mp_real):: mp_adddr
    double precision, intent (in):: da
    type (mp_real), intent (in):: rb
    integer mpnwbt
    mpnwbt = min (rb%mpr(1), mpwdsbt)
    call mp_checkdp (da)
    call mp_fixlocr (rb)
    call mp_initvr (mp_adddr, mpnwbt)
    call mpfraddd (mp_adddr%mpr(1), rb%mpr(1), da, mprnd)
    return
  end function

  function mp_addrd (ra, db)
    implicit none
    type (mp_real):: mp_addrd
    type (mp_real), intent (in):: ra
    double precision, intent (in):: db
    integer mpnwbt
    mpnwbt = min (ra%mpr(1), mpwdsbt)
    call mp_checkdp (db)
    call mp_fixlocr (ra)
    call mp_initvr (mp_addrd, mpnwbt)
    call mpfraddd (mp_addrd%mpr(1), ra%mpr(1), db, mprnd)
    return
  end function

  function mp_addir (ia, rb)
    implicit none
    type (mp_real):: mp_addir
    integer, intent (in):: ia
    type (mp_real), intent (in):: rb
    double precision da
    integer mpnwbt
    mpnwbt = min (rb%mpr(1), mpwdsbt)
    da = ia
    call mp_checkdp (da)
    call mp_fixlocr (rb)
    call mp_initvr (mp_addir, mpnwbt)
    call mpfraddd (mp_addir%mpr(1), rb%mpr(1), da, mprnd)
    return
  end function

  function mp_addri (ra, ib)
    implicit none
    type (mp_real):: mp_addri
    type (mp_real), intent (in):: ra
    integer, intent (in):: ib
    double precision db
    integer mpnwbt
    mpnwbt = min (ra%mpr(1), mpwdsbt)
    db = ib
    call mp_checkdp (db)
    call mp_fixlocr (ra)
    call mp_initvr (mp_addri, mpnwbt)
    call mpfraddd (mp_addri%mpr(1), ra%mpr(1), db, mprnd)
    return
  end function

  function mp_addzz (za, zb)
    implicit none
    type (mp_complex):: mp_addzz
    type (mp_complex), intent (in):: za, zb
    integer l1, l2, l3, mpnwbt
    l1 = za%mpc(0)
    call mp_fixlocz (za)
    l2 = zb%mpc(0)
    call mp_fixlocz (zb)
    mpnwbt = max (za%mpc(1), za%mpc(l1+1), zb%mpc(1), zb%mpc(l2+1))
    mpnwbt = min (mpnwbt, mpwdsbt)
    l3 = mpwds6
    call mp_initvz (mp_addzz, mpnwbt)
    call mpfr_add (mp_addzz%mpc(1), za%mpc(1), zb%mpc(1), %val(mprnd))
    call mpfr_add (mp_addzz%mpc(l3+1), za%mpc(l1+1), zb%mpc(l2+1), %val(mprnd))
    return
  end function

  function mp_adddz (da, zb)
    implicit none
    type (mp_complex):: mp_adddz
    double precision, intent (in):: da
    type (mp_complex), intent (in):: zb
    integer l1, l2, l3, mpnwbt
    l2 = zb%mpc(0)
    call mp_checkdp (da)
    call mp_fixlocz (zb)
    mpnwbt = max (zb%mpc(1), zb%mpc(l2+1))
    mpnwbt = min (mpnwbt, mpwdsbt)
    l3 = mpwds6
    call mp_initvz (mp_adddz, mpnwbt)
    call mpfraddd (mp_adddz%mpc(1), zb%mpc(1), da, mprnd)
    call mpfr_set (mp_adddz%mpc(l3+1), zb%mpc(l2+1), %val(mprnd))
    return
  end function

  function mp_addzd (za, db)
    implicit none
    type (mp_complex):: mp_addzd
    type (mp_complex), intent (in):: za
    double precision, intent (in):: db
    integer l1, l2, l3, mpnwbt
    l2 = za%mpc(0)
    call mp_checkdp (db)
    call mp_fixlocz (za)
    mpnwbt = max (za%mpc(1), za%mpc(l2+1))
    mpnwbt = min (mpnwbt, mpwdsbt)
    l3 = mpwds6
    call mp_initvz (mp_addzd, mpnwbt)
    call mpfraddd (mp_addzd%mpc(1), za%mpc(1), db, mprnd)
    call mpfr_set (mp_addzd%mpc(l3+1), za%mpc(l2+1), %val(mprnd))
    return
  end function

  function mp_adddcz (dca, zb)
    implicit none
    type (mp_complex):: mp_adddcz
    complex (kind (0.d0)), intent (in):: dca
    type (mp_complex), intent (in):: zb
    integer l1, l2, l3, mpnwbt
    l2 = zb%mpc(0)
    call mp_checkdp (dble (dca))
    call mp_checkdp (aimag (dca))
    call mp_fixlocz (zb)
    mpnwbt = max (zb%mpc(1), zb%mpc(l2+1))
    mpnwbt = min (mpnwbt, mpwdsbt)
    l3 = mpwds6
    call mp_initvz (mp_adddcz, mpnwbt)
    call mpfraddd (mp_adddcz%mpc(1), zb%mpc(1), dble (dca), mprnd)
    call mpfraddd (mp_adddcz%mpc(l3+1), zb%mpc(l2+1), aimag (dca), mprnd)
    return
  end function

  function mp_addzdc (za, dcb)
    implicit none
    type (mp_complex):: mp_addzdc
    type (mp_complex), intent (in):: za
    complex (kind (0.d0)), intent (in):: dcb
    integer l1, l2, l3, mpnwbt
    l2 = za%mpc(0)
    call mp_checkdp (dble (dcb))
    call mp_checkdp (aimag (dcb))
    call mp_fixlocz (za)
    mpnwbt = max (za%mpc(1), za%mpc(l2+1))
    mpnwbt = min (mpnwbt, mpwdsbt)
    l3 = mpwds6
    call mp_initvz (mp_addzdc, mpnwbt)
    call mpfraddd (mp_addzdc%mpc(1), za%mpc(1), dble (dcb), mprnd)
    call mpfraddd (mp_addzdc%mpc(l3+1), za%mpc(l2+1), aimag (dcb), mprnd)
    return
  end function

  function mp_addrz (ra, zb)
    implicit none
    type (mp_complex):: mp_addrz
    type (mp_real), intent (in):: ra
    type (mp_complex), intent (in):: zb
    integer l1, l2, l3, mpnwbt
    call mp_fixlocr (ra)
    l2 = zb%mpc(0)
    call mp_fixlocz (zb)
    mpnwbt = max (ra%mpr(1), zb%mpc(1), zb%mpc(l2+1))
    mpnwbt = min (mpnwbt, mpwdsbt)
    l3 = mpwds6
    call mp_initvz (mp_addrz, mpnwbt)
    call mpfr_add (mp_addrz%mpc(1), ra%mpr(1), zb%mpc(1), %val(mprnd))
    call mpfr_set (mp_addrz%mpc(l3+1), zb%mpc(l2+1), %val(mprnd))
    return
  end function

  function mp_addzr (za, rb)
    implicit none
    type (mp_complex):: mp_addzr
    type (mp_complex), intent (in):: za
    type (mp_real), intent (in):: rb
    integer l1, l2, l3, mpnwbt
    l1 = za%mpc(0)
    call mp_fixlocz (za)
    call mp_fixlocr (rb)
    mpnwbt = max (za%mpc(1), za%mpc(l1+1), rb%mpr(1))
    mpnwbt = min (mpnwbt, mpwdsbt)
    l3 = mpwds6
    call mp_initvz (mp_addzr, mpnwbt)
    call mpfr_add (mp_addzr%mpc(1), za%mpc(1), rb%mpr(1), %val(mprnd))
    call mpfr_set (mp_addzr%mpc(l3+1), za%mpc(l1+1), %val(mprnd))
    return
  end function

!  Subtraction routines:

  function mp_subrr (ra, rb)
    implicit none
    type (mp_real):: mp_subrr
    type (mp_real), intent (in):: ra, rb
    integer mpnwbt
    call mp_fixlocr (ra)
    call mp_fixlocr (rb)
    mpnwbt = max (ra%mpr(1), rb%mpr(1))
    mpnwbt = min (mpnwbt, mpwdsbt)
    call mp_initvr (mp_subrr, mpnwbt)
    call mpfr_sub (mp_subrr%mpr(1), ra%mpr(1), rb%mpr(1), %val(mprnd))   
    return
  end function

  function mp_subdr (da, rb)
    implicit none
    type (mp_real):: mp_subdr
    double precision, intent (in):: da
    type (mp_real), intent (in):: rb
    integer mpnwbt
    call mp_checkdp (da)
    call mp_fixlocr (rb)
    mpnwbt = min (rb%mpr(1), mpwdsbt)
    call mp_initvr (mp_subdr, mpnwbt)
    call mpfrdsub (mp_subdr%mpr(1), da, rb%mpr(1), mprnd)
    return
  end function

  function mp_subrd (ra, db)
    implicit none
    type (mp_real):: mp_subrd
    type (mp_real), intent (in):: ra
    double precision, intent (in):: db
    integer mpnwbt
    call mp_checkdp (db)
    call mp_fixlocr (ra)
    mpnwbt = min (ra%mpr(1), mpwdsbt)
    call mp_initvr (mp_subrd, mpnwbt)
    call mpfrsubd (mp_subrd%mpr(1), ra%mpr(1), db, mprnd)
    return
  end function

  function mp_subir (ia, rb)
    implicit none
    type (mp_real):: mp_subir
    integer, intent (in):: ia
    type (mp_real), intent (in):: rb
    double precision da
    integer mpnwbt
    da = ia
    call mp_checkdp (da)
    call mp_fixlocr (rb)
    mpnwbt = min (rb%mpr(1), mpwdsbt)
    call mp_initvr (mp_subir, mpnwbt)
    call mpfrdsub (mp_subir%mpr(1), da, rb%mpr(1), mprnd)
    return
  end function

  function mp_subri (ra, ib)
    implicit none
    type (mp_real):: mp_subri
    type (mp_real), intent (in):: ra
    integer, intent (in):: ib
    double precision db
    integer mpnwbt
    db = ib
    call mp_checkdp (db)
    call mp_fixlocr (ra)
    mpnwbt = min (ra%mpr(1), mpwdsbt)
    call mp_initvr (mp_subri, mpnwbt)
    call mpfrsubd (mp_subri%mpr(1), ra%mpr(1), db, mprnd)
    return
  end function

  function mp_subzz (za, zb)
    implicit none
    type (mp_complex):: mp_subzz
    type (mp_complex), intent (in):: za, zb
    integer l1, l2, l3, mpnwbt
    l1 = za%mpc(0)
    call mp_fixlocz (za)
    l2 = zb%mpc(0)
    call mp_fixlocz (zb)
    mpnwbt = max (za%mpc(1), za%mpc(l1+1), zb%mpc(1), zb%mpc(l2+1))
    mpnwbt = min (mpnwbt, mpwdsbt)
    l3 = mpwds6
    call mp_initvz (mp_subzz, mpnwbt)
    call mpfr_sub (mp_subzz%mpc(1), za%mpc(1), zb%mpc(1), %val(mprnd))
    call mpfr_sub (mp_subzz%mpc(l3+1), za%mpc(l1+1), zb%mpc(l2+1), %val(mprnd))
    return
  end function

  function mp_subdz (da, zb)
    implicit none
    type (mp_complex):: mp_subdz
    double precision, intent (in):: da
    type (mp_complex), intent (in):: zb
    integer l1, l2, l3, mpnwbt
    l2 = zb%mpc(0)
    call mp_checkdp (da)
    call mp_fixlocz (zb)
    mpnwbt = max (zb%mpc(1), zb%mpc(l2+1))
    mpnwbt = min (mpnwbt, mpwdsbt)
    l3 = mpwds6
    call mp_initvz (mp_subdz, mpnwbt)
    call mpfrdsub (mp_subdz%mpc(1), da, zb%mpc(1), mprnd)
    call mpfr_neg (mp_subdz%mpc(l3+1), zb%mpc(l2+1), %val(mprnd))
    return
  end function

  function mp_subzd (za, db)
    implicit none
    type (mp_complex):: mp_subzd
    type (mp_complex), intent (in):: za
    double precision, intent (in):: db
    integer l1, l2, l3, mpnwbt
    l2 = za%mpc(0)
    call mp_checkdp (db)
    call mp_fixlocz (za)
    mpnwbt = max (za%mpc(1), za%mpc(l2+1))
    mpnwbt = min (mpnwbt, mpwdsbt)
    l3 = mpwds6
    call mp_initvz (mp_subzd, mpnwbt)
    call mpfrsubd (mp_subzd%mpc(1), za%mpc(1), db, mprnd)
    call mpfr_set (mp_subzd%mpc(l3+1), za%mpc(l2+1), %val(mprnd))
    return
  end function

  function mp_subdcz (dca, zb)
    implicit none
    type (mp_complex):: mp_subdcz
    complex (kind (0.d0)), intent (in):: dca
    type (mp_complex), intent (in):: zb
    integer l1, l2, l3, mpnwbt
    l2 = zb%mpc(0)
    call mp_checkdp (dble (dca))
    call mp_checkdp (aimag (dca))
    call mp_fixlocz (zb)
    mpnwbt = max (zb%mpc(1), zb%mpc(l2+1))
    mpnwbt = min (mpnwbt, mpwdsbt)
    l3 = mpwds6
    call mp_initvz (mp_subdcz, mpnwbt)
    call mpfrdsub (mp_subdcz%mpc(1), dble (dca), zb%mpc(1), mprnd)
    call mpfrdsub (mp_subdcz%mpc(l3+1), aimag (dca), zb%mpc(l2+1), mprnd)
    return
  end function

  function mp_subzdc (za, dcb)
    implicit none
    type (mp_complex):: mp_subzdc
    type (mp_complex), intent (in):: za
    complex (kind (0.d0)), intent (in):: dcb
    integer l1, l2, l3, mpnwbt
    l2 = za%mpc(0)
    call mp_checkdp (dble (dcb))
    call mp_checkdp (aimag (dcb))
    call mp_fixlocz (za)
    mpnwbt = max (za%mpc(1), za%mpc(l2+1))
    mpnwbt = min (mpnwbt, mpwdsbt)
    l3 = mpwds6
    call mp_initvz (mp_subzdc, mpnwbt)
    call mpfrsubd (mp_subzdc%mpc(1), za%mpc(1), dble (dcb), mprnd)
    call mpfrsubd (mp_subzdc%mpc(l3+1), za%mpc(l2+1), aimag (dcb), mprnd)
    return
  end function

  function mp_subrz (ra, zb)
    implicit none
    type (mp_complex):: mp_subrz
    type (mp_real), intent (in):: ra
    type (mp_complex), intent (in):: zb
    integer l1, l2, l3, mpnwbt
    call mp_fixlocr (ra)
    l2 = zb%mpc(0)
    call mp_fixlocz (zb)
    mpnwbt = max (ra%mpr(1), zb%mpc(1), zb%mpc(l2+1))
    mpnwbt = min (mpnwbt, mpwdsbt)
    l3 = mpwds6
    call mp_initvz (mp_subrz, mpnwbt)
    call mpfr_sub (mp_subrz%mpc(1), ra%mpr(1), zb%mpc(1), %val(mprnd))
    call mpfr_set (mp_subrz%mpc(l3+1), zb%mpc(l2+1), %val(mprnd))
    call mpfr_neg (mp_subrz%mpc(l3+1), mp_subrz%mpc(l3+1), %val(mprnd))
    return
  end function

  function mp_subzr (za, rb)
    implicit none
    type (mp_complex):: mp_subzr
    type (mp_complex), intent (in):: za
    type (mp_real), intent (in):: rb
    integer l1, l2, l3, mpnwbt
    l1 = za%mpc(0)
    call mp_fixlocz (za)
    call mp_fixlocr (rb)
    mpnwbt = max (za%mpc(1), za%mpc(l1+1), rb%mpr(1))
    mpnwbt = min (mpnwbt, mpwdsbt)
    l3 = mpwds6
    call mp_initvz (mp_subzr, mpnwbt)
    call mpfr_sub (mp_subzr%mpc(1), za%mpc(1), rb%mpr(1), %val(mprnd))
    call mpfr_set (mp_subzr%mpc(l3+1), za%mpc(l1+1), %val(mprnd))
    return
  end function

!  Negation routines:

  function mp_negr (ra)
    implicit none
    type (mp_real):: mp_negr
    type (mp_real), intent (in):: ra
    integer mpnwbt
    call mp_fixlocr (ra)
    mpnwbt = min (ra%mpr(1), mpwdsbt)
    call mp_initvr (mp_negr, mpnwbt)
    call mpfr_neg (mp_negr%mpr(1), ra%mpr(1), %val(mprnd))
    return
  end function

  function mp_negz (za)
    implicit none
    type (mp_complex):: mp_negz
    type (mp_complex), intent (in):: za
    integer l1, l2, l3, mpnwbt
    l1 = za%mpc(0)
    call mp_fixlocz (za)
    mpnwbt = max (za%mpc(1), za%mpc(l1+1))
    mpnwbt = min (mpnwbt, mpwdsbt)
    l3 = mpwds6
    call mp_initvz (mp_negz, mpnwbt)
    call mpfr_neg (mp_negz%mpc(1), za%mpc(1), %val(mprnd))
    call mpfr_neg (mp_negz%mpc(l3+1), za%mpc(l1+1), %val(mprnd))
    return
  end function

!  Multiplication routines:

  function mp_mulrr (ra, rb)
    implicit none
    type (mp_real):: mp_mulrr
    type (mp_real), intent (in):: ra, rb
    integer mpnwbt
    call mp_fixlocr (ra)
    call mp_fixlocr (rb)
    mpnwbt = max (ra%mpr(1), rb%mpr(1))
    mpnwbt = min (mpnwbt, mpwdsbt)
    call mp_initvr (mp_mulrr, mpnwbt)
    call mpfr_mul (mp_mulrr%mpr(1), ra%mpr(1), rb%mpr(1), %val(mprnd))
    return
  end function

  function mp_muldr (da, rb)
    implicit none
    type (mp_real):: mp_muldr
    double precision, intent (in):: da
    type (mp_real), intent (in):: rb
    integer mpnwbt
    call mp_checkdp (da)
    call mp_fixlocr (rb)
    mpnwbt = min (rb%mpr(1), mpwdsbt)
    call mp_initvr (mp_muldr, mpnwbt)
    call mpfrmuld (mp_muldr%mpr(1), rb%mpr(1), da, mprnd)
    return
  end function

  function mp_mulrd (ra, db)
    implicit none
    type (mp_real):: mp_mulrd
    type (mp_real), intent (in):: ra
    double precision, intent (in):: db
    integer mpnwbt
    call mp_checkdp (db)
    call mp_fixlocr (ra)
    mpnwbt = min (ra%mpr(1), mpwdsbt)
    call mp_initvr (mp_mulrd, mpnwbt)
    call mpfrmuld (mp_mulrd%mpr(1), ra%mpr(1), db, mprnd)
    return
  end function

  function mp_mulir (ia, rb)
    implicit none
    type (mp_real):: mp_mulir
    integer, intent (in):: ia
    type (mp_real), intent (in):: rb
    double precision da
    integer mpnwbt
    da = ia
    call mp_checkdp (da)
    call mp_fixlocr (rb)
    mpnwbt = min (rb%mpr(1), mpwdsbt)
    call mp_initvr (mp_mulir, mpnwbt)
    call mpfrmuld (mp_mulir%mpr(1), rb%mpr(1), da, mprnd)
    return
  end function

  function mp_mulri (ra, ib)
    implicit none
    type (mp_real):: mp_mulri
    type (mp_real), intent (in):: ra
    integer, intent (in):: ib
    double precision db
    integer mpnwbt
    db = ib
    call mp_checkdp (db)
    call mp_fixlocr (ra)
    mpnwbt = min (ra%mpr(1), mpwdsbt)
    call mp_initvr (mp_mulri, mpnwbt)
    call mpfrmuld (mp_mulri%mpr(1), ra%mpr(1), db, mprnd)
    return
  end function

  function mp_mulzz (za, zb)
    implicit none
    type (mp_complex):: mp_mulzz
    type (mp_complex), intent (in):: za, zb
    type (mp_real) r1, r2
    integer l1, l2, l3, mpnwbt
    l1 = za%mpc(0)
    call mp_fixlocz (za)
    l2 = zb%mpc(0)
    call mp_fixlocz (zb)
    mpnwbt = max (za%mpc(1), za%mpc(l1+1), zb%mpc(1), zb%mpc(l2+1))
    mpnwbt = min (mpnwbt, mpwdsbt)
    call mp_initvr (r1, mpnwbt)
    call mp_initvr (r2, mpnwbt)
    l3 = mpwds6
    call mp_initvz (mp_mulzz, mpnwbt)
    call mpfr_mul (r1%mpr(1), za%mpc(1), zb%mpc(1), %val(mprnd))
    call mpfr_mul (r2%mpr(1), za%mpc(l1+1), zb%mpc(l2+1), %val(mprnd))
    call mpfr_sub (mp_mulzz%mpc(1), r1%mpr(1), r2%mpr(1), %val(mprnd))
    call mpfr_mul (r1%mpr(1), za%mpc(1), zb%mpc(l2+1), %val(mprnd))
    call mpfr_mul (r2%mpr(1), za%mpc(l1+1), zb%mpc(1), %val(mprnd))
    call mpfr_add (mp_mulzz%mpc(l3+1), r1%mpr(1), r2%mpr(1), %val(mprnd))
    return
  end function

  function mp_muldz (da, zb)
    implicit none
    type (mp_complex):: mp_muldz
    double precision, intent (in):: da
    type (mp_complex), intent (in):: zb
    integer l1, l2, l3, mpnwbt
    l2 = zb%mpc(0)
    call mp_checkdp (da)
    call mp_fixlocz (zb)
    mpnwbt = max (zb%mpc(1), zb%mpc(l2+1))
    mpnwbt = min (mpnwbt, mpwdsbt)
    l3 = mpwds6
    call mp_initvz (mp_muldz, mpnwbt)
    call mpfrmuld (mp_muldz%mpc(1), zb%mpc(1), da, mprnd)
    call mpfrmuld (mp_muldz%mpc(l3+1), zb%mpc(l2+1), da, mprnd)
    return
  end function

  function mp_mulzd (za, db)
    implicit none
    type (mp_complex):: mp_mulzd
    type (mp_complex), intent (in):: za
    double precision, intent (in):: db
    integer l1, l2, l3, mpnwbt
    l2 = za%mpc(0)
    call mp_checkdp (db)
    call mp_fixlocz (za)
    mpnwbt = max (za%mpc(1), za%mpc(l2+1))
    mpnwbt = min (mpnwbt, mpwdsbt)
    l3 = mpwds6
    call mp_initvz (mp_mulzd, mpnwbt)
    call mpfrmuld (mp_mulzd%mpc(1), za%mpc(1), db, mprnd)
    call mpfrmuld (mp_mulzd%mpc(l3+1), za%mpc(l2+1), db, mprnd)
    return
  end function

  function mp_muldcz (dca, zb)
    implicit none
    type (mp_complex):: mp_muldcz
    complex (kind (0.d0)), intent (in):: dca
    type (mp_complex), intent (in):: zb
    type (mp_real) r1, r2
    integer l1, l2, l3, mpnwbt
    l2 = zb%mpc(0)
    call mp_fixlocz (zb)
    mpnwbt = max (zb%mpc(1), zb%mpc(l2+1))
    mpnwbt = min (mpnwbt, mpwdsbt)
    call mp_checkdp (dble (dca))
    call mp_checkdp (aimag (dca))
    call mp_initvr (r1, mpnwbt)
    call mp_initvr (r2, mpnwbt)
    l3 = mpwds6
    call mp_initvz (mp_muldcz, mpnwbt)
    call mpfrmuld (r1%mpr(1), zb%mpc(1), dble (dca), mprnd)
    call mpfrmuld (r2%mpr(1), zb%mpc(l2+1), aimag (dca), mprnd)
    call mpfr_sub (mp_muldcz%mpc(1), r1%mpr(1), r2%mpr(1), %val(mprnd))
    call mpfrmuld (r1%mpr(1), zb%mpc(l2+1), dble (dca), mprnd)
    call mpfrmuld (r2%mpr(1), zb%mpc(1), aimag (dca), mprnd)
    call mpfr_add (mp_muldcz%mpc(l3+1), r1%mpr(1), r2%mpr(1), %val(mprnd))
    return
  end function

  function mp_mulzdc (za, dcb)
    implicit none
    type (mp_complex):: mp_mulzdc
    type (mp_complex), intent (in):: za
    complex (kind (0.d0)), intent (in):: dcb
    type (mp_real) r1, r2
    integer l1, l2, l3, mpnwbt
    l1 = za%mpc(0)
    call mp_fixlocz (za)
    mpnwbt = max (za%mpc(1), za%mpc(l1+1))
    mpnwbt = min (mpnwbt, mpwdsbt)
    call mp_checkdp (dble (dcb))
    call mp_checkdp (aimag (dcb))
    call mp_initvr (r1, mpnwbt)
    call mp_initvr (r2, mpnwbt)
    l3 = mpwds6
    call mp_initvz (mp_mulzdc, mpnwbt)
    call mpfrmuld (r1%mpr(1), za%mpc(1), dble (dcb), mprnd)
    call mpfrmuld (r2%mpr(1), za%mpc(l1+1), aimag (dcb), mprnd)
    call mpfr_sub (mp_mulzdc%mpc(1), r1%mpr(1), r2%mpr(1), %val(mprnd))
    call mpfrmuld (r1%mpr(1), za%mpc(l1+1), dble (dcb), mprnd)
    call mpfrmuld (r2%mpr(1), za%mpc(1), aimag (dcb), mprnd)
    call mpfr_add (mp_mulzdc%mpc(l3+1), r1%mpr(1), r2%mpr(1), %val(mprnd))
    return
  end function

  function mp_mulrz (ra, zb)
    implicit none
    type (mp_complex):: mp_mulrz
    type (mp_real), intent (in):: ra
    type (mp_complex), intent (in):: zb
    integer l1, l2, l3, mpnwbt
    l1 = ra%mpr(0)
    call mp_fixlocr (ra)
    l2 = zb%mpc(0)
    call mp_fixlocz (zb)
    mpnwbt = max (ra%mpr(1), zb%mpc(1), zb%mpc(l2+1))
    mpnwbt = min (mpnwbt, mpwdsbt)
    l3 = mpwds6
    call mp_initvz (mp_mulrz, mpnwbt)
    call mpfr_mul (mp_mulrz%mpc(1), ra%mpr(1), zb%mpc(1), %val(mprnd))
    call mpfr_mul (mp_mulrz%mpc(l3+1), ra%mpr(1), zb%mpc(l2+1), %val(mprnd))
    return
  end function

  function mp_mulzr (za, rb)
    implicit none
    type (mp_complex):: mp_mulzr
    type (mp_complex), intent (in):: za
    type (mp_real), intent (in):: rb
    integer l1, l2, l3, mpnwbt
    l1 = za%mpc(0)
    call mp_fixlocz (za)
    l2 = rb%mpr(0)
    call mp_fixlocr (rb)
    mpnwbt = max (za%mpc(1), za%mpc(l1+1), rb%mpr(1))
    mpnwbt = min (mpnwbt, mpwdsbt)
    l3 = mpwds6
    call mp_initvz (mp_mulzr, mpnwbt)
    call mpfr_mul (mp_mulzr%mpc(1), za%mpc(1), rb%mpr(1), %val(mprnd))
    call mpfr_mul (mp_mulzr%mpc(l3+1), za%mpc(l1+1), rb%mpr(1), %val(mprnd))
    return
  end function
  
!  Division routines:

  function mp_divrr (ra, rb)
    implicit none
    type (mp_real):: mp_divrr
    type (mp_real), intent (in):: ra, rb
    integer mpnwbt
    call mp_fixlocr (ra)
    call mp_fixlocr (rb)
    mpnwbt = max (ra%mpr(1), rb%mpr(1))
    mpnwbt = min (mpnwbt, mpwdsbt)
    call mp_initvr (mp_divrr, mpnwbt)
    call mpfr_div (mp_divrr%mpr(1), ra%mpr(1), rb%mpr(1), %val(mprnd))
    return
  end function

  function mp_divdr (da, rb)
    implicit none
    type (mp_real):: mp_divdr
    double precision, intent (in):: da
    type (mp_real), intent (in):: rb
    integer mpnwbt
    call mp_checkdp (da)
    call mp_fixlocr (rb)
    mpnwbt = min (rb%mpr(1), mpwdsbt)
    call mp_initvr (mp_divdr, mpnwbt)
    call mpfrddiv (mp_divdr%mpr(1), da, rb%mpr(1), mprnd)   
    return
  end function

  function mp_divrd (ra, db)
    implicit none
    type (mp_real):: mp_divrd
    type (mp_real), intent (in):: ra
    double precision, intent (in):: db
    integer mpnwbt
    call mp_checkdp (db)
    call mp_fixlocr (ra)
    mpnwbt = min (ra%mpr(1), mpwdsbt)
    call mp_initvr (mp_divrd, mpnwbt)
    call mpfrdivd (mp_divrd%mpr(1), ra%mpr(1), db, mprnd)
    return
  end function

  function mp_divir (ia, rb)
    implicit none
    type (mp_real):: mp_divir
    integer, intent (in):: ia
    type (mp_real), intent (in):: rb
    double precision da
    integer mpnwbt
    da = ia
    call mp_checkdp (da)
    call mp_fixlocr (rb)
    mpnwbt = min (rb%mpr(1), mpwdsbt)
    call mp_initvr (mp_divir, mpnwbt)
    call mpfrddiv (mp_divir%mpr(1), da, rb%mpr(1), mprnd)   
    return
  end function

  function mp_divri (ra, ib)
    implicit none
    type (mp_real):: mp_divri
    type (mp_real), intent (in):: ra
    integer, intent (in):: ib
    double precision db
    integer mpnwbt
    db = ib
    call mp_checkdp (db)
    call mp_fixlocr (ra)
    mpnwbt = min (ra%mpr(1), mpwdsbt)
    call mp_initvr (mp_divri, mpnwbt)
    call mpfrdivd (mp_divri%mpr(1), ra%mpr(1), db, mprnd)
    return
  end function

  function mp_divzz (za, zb)
    implicit none
    type (mp_complex):: mp_divzz
    type (mp_complex), intent (in):: za, zb
    type (mp_real) r1, r2, r3
    integer l1, l2, l3, mpnwbt
    l1 = za%mpc(0)
    call mp_fixlocz (za)
    l2 = zb%mpc(0)
    call mp_fixlocz (zb)
    mpnwbt = max (za%mpc(1), za%mpc(l1+1), zb%mpc(1), zb%mpc(l2+1))
    mpnwbt = min (mpnwbt, mpwdsbt)
    call mp_initvr (r1, mpnwbt)
    call mp_initvr (r2, mpnwbt)
    call mp_initvr (r3, mpnwbt)
    l3 = mpwds6
    call mp_initvz (mp_divzz, mpnwbt)
    call mpfr_mul (r1%mpr(1), za%mpc(1), zb%mpc(1), %val(mprnd))
    call mpfr_mul (r2%mpr(1), za%mpc(l1+1), zb%mpc(l2+1), %val(mprnd))
    call mpfr_add (mp_divzz%mpc(1), r1%mpr(1), r2%mpr(1), %val(mprnd))
    call mpfr_mul (r1%mpr(1), za%mpc(1), zb%mpc(l2+1), %val(mprnd))
    call mpfr_mul (r2%mpr(1), za%mpc(l1+1), zb%mpc(1), %val(mprnd))
    call mpfr_sub (mp_divzz%mpc(l3+1), r2%mpr(1), r1%mpr(1), %val(mprnd))
    call mpfr_mul (r1%mpr(1), zb%mpc(1), zb%mpc(1), %val(mprnd))
    call mpfr_mul (r2%mpr(1), zb%mpc(l2+1), zb%mpc(l2+1), %val(mprnd))
    call mpfr_add (r3%mpr(1), r1%mpr(1), r2%mpr(1), %val(mprnd))
    call mpfr_div (mp_divzz%mpc(1), mp_divzz%mpc(1), r3%mpr(1), %val(mprnd))
    call mpfr_div (mp_divzz%mpc(l3+1), mp_divzz%mpc(l3+1), r3%mpr(1), %val(mprnd))
    return
  end function

  function mp_divdz (da, zb)
    implicit none
    type (mp_complex):: mp_divdz
    double precision, intent (in):: da
    type (mp_complex), intent (in):: zb
    double precision d1, d2
    type (mp_real) r1, r2, r3, r4, r5
    integer l1, l2, l3, mpnwbt
    l2 = zb%mpc(0)
    call mp_checkdp (da)
    call mp_fixlocz (zb)
    mpnwbt = max (zb%mpc(1), zb%mpc(l2+1))
    mpnwbt = min (mpnwbt, mpwdsbt)
    l3 = mpwds6
    call mp_initvr (r1, mpnwbt)
    call mp_initvr (r2, mpnwbt)
    call mp_initvr (r3, mpnwbt)
    call mp_initvr (r4, mpnwbt)
    call mp_initvr (r5, mpnwbt)
    call mp_initvz (mp_divdz, mpnwbt)
    call mpfr_mul (r1%mpr(1), zb%mpc(1), zb%mpc(1), %val(mprnd))
    call mpfr_mul (r2%mpr(1), zb%mpc(l2+1), zb%mpc(l2+1), %val(mprnd))
    call mpfr_add (r3%mpr(1), r1%mpr(1), r2%mpr(1), %val(mprnd))
    call mpfrmuld (r4%mpr(1), zb%mpc(1), da, mprnd)
    call mpfrmuld (r5%mpr(1), zb%mpc(l2+1), da, mprnd)
    call mpfr_div (mp_divdz%mpc(1), r4%mpr(1), r3%mpr(1), %val(mprnd))
    call mpfr_div (mp_divdz%mpc(l3+1), r5%mpr(1), r3%mpr(1), %val(mprnd))
    call mpfr_neg (mp_divdz%mpc(l3+1), mp_divdz%mpc(l3+1), %val(mprnd))
    return
  end function

  function mp_divzd (za, db)
    implicit none
    type (mp_complex):: mp_divzd
    type (mp_complex), intent (in):: za
    double precision, intent (in):: db
    integer l1, l2, l3, mpnwbt
    l2 = za%mpc(0)
    call mp_checkdp (db)
    call mp_fixlocz (za)
    mpnwbt = max (za%mpc(1), za%mpc(l2+1))
    mpnwbt = min (mpnwbt, mpwdsbt)
    l3 = mpwds6
    call mp_initvz (mp_divzd, mpnwbt)
    call mpfrdivd (mp_divzd%mpc(1), za%mpc(1), db, mprnd)
    call mpfrdivd (mp_divzd%mpc(l3+1), za%mpc(l2+1), db, mprnd)
    return
  end function

  function mp_divdcz (dca, zb)
    implicit none
    type (mp_complex):: mp_divdcz
    complex (kind (0.d0)), intent (in):: dca
    type (mp_complex), intent (in):: zb
    type (mp_real) r1, r2, r3
    integer l1, l2, l3, mpnwbt
    l2 = zb%mpc(0)
    call mp_fixlocz (zb)
    mpnwbt = max (zb%mpc(1), zb%mpc(l2+1))
    mpnwbt = min (mpnwbt, mpwdsbt)
    call mp_checkdp (dble (dca))
    call mp_checkdp (aimag (dca))
    call mp_initvr (r1, mpnwbt)
    call mp_initvr (r2, mpnwbt)
    call mp_initvr (r3, mpnwbt)
    l3 = mpwds6
    call mp_initvz (mp_divdcz, mpnwbt)
    call mpfrmuld (r1%mpr(1), zb%mpc(1), dble (dca), mprnd)
    call mpfrmuld (r2%mpr(1), zb%mpc(l2+1), aimag (dca), mprnd)
    call mpfr_add (mp_divdcz%mpc(1), r1%mpr(1), r2%mpr(1), %val(mprnd))
    call mpfrmuld (r1%mpr(1), zb%mpc(l2+1), dble (dca), mprnd)
    call mpfrmuld (r2%mpr(1), zb%mpc(1), aimag (dca), mprnd)
    call mpfr_sub (mp_divdcz%mpc(l3+1), r2%mpr(1), r1%mpr(1), %val(mprnd))
    call mpfr_mul (r1%mpr(1), zb%mpc(1), zb%mpc(1), %val(mprnd))
    call mpfr_mul (r2%mpr(1), zb%mpc(l2+1), zb%mpc(l2+1), %val(mprnd))
    call mpfr_add (r3%mpr(1), r1%mpr(1), r2%mpr(1), %val(mprnd))
    call mpfr_div (mp_divdcz%mpc(1), mp_divdcz%mpc(1), r3%mpr(1), %val(mprnd))
    call mpfr_div (mp_divdcz%mpc(l3+1), mp_divdcz%mpc(l3+1), r3%mpr(1), %val(mprnd))
    return
  end function

  function mp_divzdc (za, dcb)
    implicit none
    type (mp_complex):: mp_divzdc
    type (mp_complex), intent (in):: za
    complex (kind (0.d0)), intent (in):: dcb
    type (mp_real) r1, r2
    double precision d3
    integer l1, l2, l3, mpnwbt
    l1 = za%mpc(0)
    call mp_fixlocz (za)
    mpnwbt = max (za%mpc(1), za%mpc(l1+1))
    mpnwbt = min (mpnwbt, mpwdsbt)
    call mp_checkdp (dble (dcb))
    call mp_checkdp (aimag (dcb))
    call mp_initvr (r1, mpnwbt)
    call mp_initvr (r2, mpnwbt)
    l3 = mpwds6
    call mp_initvz (mp_divzdc, mpnwbt)
    call mpfrmuld (r1%mpr(1), za%mpc(1), dble (dcb), mprnd)
    call mpfrmuld (r2%mpr(1), za%mpc(l1+1), aimag (dcb), mprnd)
    call mpfr_add (mp_divzdc%mpc(1), r1%mpr(1), r2%mpr(1), %val(mprnd))
    call mpfrmuld (r1%mpr(1), za%mpc(l1+1), dble (dcb), mprnd)
    call mpfrmuld (r2%mpr(1), za%mpc(1), aimag (dcb), mprnd)
    call mpfr_sub (mp_divzdc%mpc(l3+1), r1%mpr(1), r2%mpr(1), %val(mprnd))
    d3 = dble (dcb) ** 2 + aimag (dcb) ** 2
    call mpfrdivd (mp_divzdc%mpc(1), mp_divzdc%mpc(1), d3, mprnd)
    call mpfrdivd (mp_divzdc%mpc(l3+1), mp_divzdc%mpc(l3+1), d3, mprnd)
    return
  end function

  function mp_divrz (ra, zb)
    implicit none
    type (mp_complex):: mp_divrz
    type (mp_real), intent (in):: ra
    type (mp_complex), intent (in):: zb
    type (mp_real) r1, r2, r3
    integer l1, l2, l3, mpnwbt
    l1 = ra%mpr(0)
    call mp_fixlocr (ra)
    l2 = zb%mpc(0)
    call mp_fixlocz (zb)
    mpnwbt = max (ra%mpr(1), zb%mpc(1), zb%mpc(l2+1))
    mpnwbt = min (mpnwbt, mpwdsbt)
    call mp_initvr (r1, mpnwbt)
    call mp_initvr (r2, mpnwbt)
    call mp_initvr (r3, mpnwbt)
    l3 = mpwds6
    call mp_initvz (mp_divrz, mpnwbt)
    call mpfr_mul (mp_divrz%mpc(1), ra%mpr(1), zb%mpc(1), %val(mprnd))
    call mpfr_mul (r1%mpr(1), ra%mpr(1), zb%mpc(l2+1), %val(mprnd))
    call mpfr_neg (mp_divrz%mpc(l3+1), r1%mpr(1), %val(mprnd))
    call mpfr_mul (r1%mpr(1), zb%mpc(1), zb%mpc(1), %val(mprnd))
    call mpfr_mul (r2%mpr(1), zb%mpc(l2+1), zb%mpc(l2+1), %val(mprnd))
    call mpfr_add (r3%mpr(1), r1%mpr(1), r2%mpr(1), %val(mprnd))
    call mpfr_div (mp_divrz%mpc(1), mp_divrz%mpc(1), r3%mpr(1), %val(mprnd))
    call mpfr_div (mp_divrz%mpc(l3+1), mp_divrz%mpc(l3+1), r3%mpr(1), %val(mprnd))
    return
  end function

  function mp_divzr (za, rb)
    implicit none
    type (mp_complex):: mp_divzr
    type (mp_complex), intent (in):: za
    type (mp_real), intent (in):: rb
    integer l1, l2, l3, mpnwbt
    l1 = za%mpc(0)
    call mp_fixlocz (za)
    l2 = rb%mpr(0)
    call mp_fixlocr (rb)
    mpnwbt = max (za%mpc(1), za%mpc(l1+1), rb%mpr(1))
    mpnwbt = min (mpnwbt, mpwdsbt)
    l3 = mpwds6
    call mp_initvz (mp_divzr, mpnwbt)
    call mpfr_div (mp_divzr%mpc(1), za%mpc(1), rb%mpr(1), %val(mprnd))
    call mpfr_div (mp_divzr%mpc(l3+1), za%mpc(l1+1), rb%mpr(1), %val(mprnd))
    return
  end function

!  Exponentiation routines:

  function mp_expri (ra, ib)
    implicit none
    type (mp_real):: mp_expri
    type (mp_real), intent (in):: ra
    integer, intent (in):: ib
    integer mpnwbt
    call mp_fixlocr (ra)
    mpnwbt = min (ra%mpr(1), mpwdsbt)
    call mp_initvr (mp_expri, mpnwbt)
    call mpfrpowsi (mp_expri%mpr(1), ra%mpr(1), ib, mprnd)
    return
  end function

  function mp_exprr (ra, rb)
    implicit none
    type (mp_real):: mp_exprr
    type (mp_real), intent (in):: ra, rb
    integer mpnwbt
    call mp_fixlocr (ra)
    call mp_fixlocr (rb)
    mpnwbt = max (ra%mpr(1), rb%mpr(1))
    mpnwbt = min (mpnwbt, mpwdsbt)
    call mp_initvr (mp_exprr, mpnwbt)
    call mpfr_pow (mp_exprr%mpr(1), ra%mpr(1), rb%mpr(1), %val(mprnd))
    return
  end function

  function mp_expzi (za, ib)
    implicit none
    type (mp_complex):: mp_expzi
    type (mp_complex), intent (in):: za
    integer, intent (in):: ib
    integer j, kk, kn, l1, l2, l3, mpnwbt, mn, nn
    type (mp_complex) z0, z1, z2
    call mp_fixlocz (za)
    l1 = za%mpc(0)
    mpnwbt = max (za%mpc(1), za%mpc(l1+1))
    mpnwbt = min (mpnwbt, mpwdsbt)
    l2 = mpwds6
    call mp_initvz (z0, mpnwbt)
    call mp_initvz (z1, mpnwbt)
    call mp_initvz (z2, mpnwbt)
    l3 = mpwds6
    call mp_initvz (mp_expzi, mpnwbt)
    nn = abs (ib)

!   Handle nn = 0, 1, 2 as special cases.

    if (nn == 0) then
      call mpfrsetd (mp_expzi%mpc(1), 1.d0, mprnd)
      call mpfrsetd (mp_expzi%mpc(l3+1), 0.d0, mprnd)
      goto 120
    elseif (nn == 1) then
      call mpfr_set (z2%mpc(1), za%mpc(1), %val(mprnd))
      call mpfr_set (z2%mpc(l2+1), za%mpc(l1+1), %val(mprnd))
      goto 110
    elseif (nn == 2) then
      z2 = mp_mulzz (za, za)
      goto 110
    endif

!   Determine the least integer mn such that 2^mn > nn.

    mn = log (dble (nn)) / log(2.d0) + 1.d0 + 1.d-14
    call mpfrsetd (z2%mpc(1), 1.d0, mprnd)
    call mpfrsetd (z2%mpc(l2+1), 0.d0, mprnd)
    call mpfr_set (z0%mpc(1), za%mpc(1), %val(mprnd))
    call mpfr_set (z0%mpc(l2+1), za%mpc(l1+1), %val(mprnd))
    kn = nn

!   Compute za^nn using the binary rule for exponentiation.

    do j = 1, mn
      kk = kn / 2
      if (kn /= 2 * kk) then
        z1 = mp_mulzz (z2, z0)
        call mpfr_set (z2%mpc(1), z1%mpc(1), %val(mprnd))
        call mpfr_set (z2%mpc(l2+1), z1%mpc(l2+1), %val(mprnd))
      endif
      kn = kk
      if (j < mn) then
        z1 = mp_mulzz (z0, z0)
        call mpfr_set (z0%mpc(1), z1%mpc(1), %val(mprnd))
        call mpfr_set (z0%mpc(l2+1), z1%mpc(l2+1), %val(mprnd))
      endif
    enddo

!   Compute reciprocal if ib is negative.

110 continue

    if (ib < 0) then
      call mpfrsetd (z1%mpc(1), 1.d0, mprnd)
      call mpfrsetd (z1%mpc(l2+1), 0.d0, mprnd)
      z0 = mp_divzz (z1, z2)
      call mpfr_set (z2%mpc(1), z0%mpc(1), %val(mprnd))
      call mpfr_set (z2%mpc(l2+1), z0%mpc(l2+1), %val(mprnd))
    endif

    call mpfr_set (mp_expzi%mpc(1), z2%mpc(1), %val(mprnd))
    call mpfr_set (mp_expzi%mpc(l3+1), z2%mpc(l2+1), %val(mprnd))

120 continue
    return
  end function

  function mp_expzz (za, zb)
    implicit none
    type (mp_complex):: mp_expzz
    type (mp_complex), intent (in):: za, zb
    type (mp_real):: r1, r2, r3, r4, r5, r6
    integer l1, l2, l3, mpnwbt
    call mp_fixlocz (za)
    call mp_fixlocz (zb)
    l1 = za%mpc(0)
    l2 = zb%mpc(0)
    mpnwbt = max (za%mpc(1), za%mpc(l1+1), zb%mpc(1), zb%mpc(l2+1))
    mpnwbt = min (mpnwbt, mpwdsbt)
    l3 = mpwds6
    call mp_initvr (r1, mpnwbt)
    call mp_initvr (r2, mpnwbt)
    call mp_initvr (r3, mpnwbt)
    call mp_initvr (r4, mpnwbt)
    call mp_initvr (r5, mpnwbt)
    call mp_initvr (r6, mpnwbt)
    call mp_initvz (mp_expzz, mpnwbt)    
    call mpfr_mul (r1%mpr(1), za%mpc(1), za%mpc(1), %val(mprnd))
    call mpfr_mul (r2%mpr(1), za%mpc(l1+1), za%mpc(l1+1), %val(mprnd))
    call mpfr_add (r3%mpr(1), r1%mpr(1), r2%mpr(1), %val(mprnd))
    call mpfr_log (r4%mpr(1), r3%mpr(1), %val(mprnd))
    call mpfrmuld (r5%mpr(1), r4%mpr(1), 0.5d0, mprnd)
    call mpfr_mul (r1%mpr(1), zb%mpc(1), r5%mpr(1), %val(mprnd))
    call mpfr_atan2 (r2%mpr(1), za%mpc(l1+1), za%mpc(1), %val(mprnd))
    call mpfr_mul (r3%mpr(1), r2%mpr(1), zb%mpc(l2+1), %val(mprnd))
    call mpfr_sub (r4%mpr(1), r1%mpr(1), r3%mpr(1), %val(mprnd))
    call mpfr_exp (r1%mpr(1), r4%mpr(1), %val(mprnd))
    call mpfr_mul (r3%mpr(1), zb%mpc(l2+1), r5%mpr(1), %val(mprnd))
    call mpfr_mul (r4%mpr(1), zb%mpc(1), r2%mpr(1), %val(mprnd))
    call mpfr_add (r6%mpr(1), r3%mpr(1), r4%mpr(1), %val(mprnd))
    call mpfr_sin_cos (r4%mpr(1), r3%mpr(1), r6%mpr(1), %val(mprnd))
    call mpfr_mul (mp_expzz%mpc(1), r1%mpr(1), r3%mpr(1), %val(mprnd))
    call mpfr_mul (mp_expzz%mpc(l3+1), r1%mpr(1), r4%mpr(1), %val(mprnd))
  end function

  function mp_exprz (ra, zb)
    implicit none
    type (mp_complex):: mp_exprz
    type (mp_real), intent (in):: ra
    type (mp_complex), intent (in):: zb
    type (mp_real):: r1, r2, r3, r4, r5
    integer l1, l2, mpnwbt
    call mp_fixlocr (ra)
    call mp_fixlocz (zb)
    l1 = zb%mpc(0)
    mpnwbt = max (ra%mpr(1), zb%mpc(1), zb%mpc(l1+1))
    mpnwbt = min (mpnwbt, mpwdsbt)
    l2 = mpwds6
    call mp_initvr (r1, mpnwbt)
    call mp_initvr (r2, mpnwbt)
    call mp_initvr (r3, mpnwbt)
    call mp_initvr (r4, mpnwbt)
    call mp_initvr (r5, mpnwbt)
    call mp_initvz (mp_exprz, mpnwbt)
    call mpfr_pow (r1%mpr(1), ra%mpr(1), zb%mpc(1), %val(mprnd))
    call mpfr_log (r2%mpr(1), ra%mpr(1), %val(mprnd))
    call mpfr_mul (r3%mpr(1), r2%mpr(1), zb%mpc(l1+1), %val(mprnd))
    call mpfr_sin_cos (r5%mpr(1), r4%mpr(1), r3%mpr(1), %val(mprnd))
    call mpfr_mul (mp_exprz%mpc(1), r1%mpr(1), r4%mpr(1), %val(mprnd))
    call mpfr_mul (mp_exprz%mpc(l2+1), r1%mpr(1), r5%mpr(1), %val(mprnd))
    return
  end function

  function mp_expzr (za, rb)
    implicit none
    type (mp_complex):: mp_expzr
    type (mp_complex), intent (in):: za
    type (mp_real), intent (in):: rb
    type (mp_real):: r1, r2, r3, r4, r5
    integer l1, l2, mpnwbt
    call mp_fixlocz (za)
    call mp_fixlocr (rb)
    l1 = za%mpc(0)
    mpnwbt = max (za%mpc(1), za%mpc(l1+1), rb%mpr(1))
    mpnwbt = min (mpnwbt, mpwdsbt)
    l2 = mpwds6
    call mp_initvr (r1, mpnwbt)
    call mp_initvr (r2, mpnwbt)
    call mp_initvr (r3, mpnwbt)
    call mp_initvr (r4, mpnwbt)
    call mp_initvr (r5, mpnwbt)
    call mp_initvz (mp_expzr, mpnwbt)
    call mpfr_mul (r1%mpr(1), za%mpc(1), za%mpc(1), %val(mprnd))
    call mpfr_mul (r2%mpr(1), za%mpc(l1+1), za%mpc(l1+1), %val(mprnd))
    call mpfr_add (r3%mpr(1), r1%mpr(1), r2%mpr(1), %val(mprnd))
    call mpfr_log (r4%mpr(1), r3%mpr(1), %val(mprnd))
    call mpfrmuld (r5%mpr(1), r4%mpr(1), 0.5d0, mprnd)
    call mpfr_mul (r1%mpr(1), r5%mpr(1), rb%mpr(1), %val(mprnd))
    call mpfr_exp (r2%mpr(1), r1%mpr(1), %val(mprnd))
    call mpfr_atan2 (r3%mpr(1), za%mpc(l1+1), za%mpc(1), %val(mprnd))
    call mpfr_mul (r1%mpr(1), rb%mpr(1), r3%mpr(1), %val(mprnd))
    call mpfr_sin_cos (r5%mpr(1), r4%mpr(1), r1%mpr(1), %val(mprnd)) 
    call mpfr_mul (mp_expzr%mpc(1), r2%mpr(1), r4%mpr(1), %val(mprnd))
    call mpfr_mul (mp_expzr%mpc(l2+1), r2%mpr(1), r5%mpr(1), %val(mprnd))
    return
  end function

!  Equality test routines:

  function mp_eqtrr (ra, rb)
    implicit none
    logical mp_eqtrr
    type (mp_real), intent (in):: ra, rb
    integer ic, mpfr_cmp
    external mpfr_cmp
    call mp_fixlocr (ra)
    call mp_fixlocr (rb)
    ic = mpfr_cmp (ra%mpr(1), rb%mpr(1))
    if (ic == 0) then
      mp_eqtrr = .true.
    else
      mp_eqtrr = .false.
    endif
    return
  end function

  function mp_eqtdr (da, rb)
    implicit none
    logical mp_eqtdr
    double precision, intent (in):: da
    type (mp_real), intent (in):: rb
    integer ic, mpfr_cmp_d
    external mpfr_cmp_d
    call mp_checkdp (da)
    call mp_fixlocr (rb)
    ic = - mpfr_cmp_d (rb%mpr(1), %val(da))
    if (ic == 0) then
      mp_eqtdr = .true.
    else
      mp_eqtdr = .false.
    endif
    return
  end function

  function mp_eqtrd (ra, db)
    implicit none
    logical mp_eqtrd
    type (mp_real), intent (in):: ra
    double precision, intent (in):: db
    integer ic, mpfr_cmp_d
    external mpfr_cmp_d
    call mp_checkdp (db)
    call mp_fixlocr (ra)
    ic = mpfr_cmp_d (ra%mpr(1), %val(db))
    if (ic == 0) then
      mp_eqtrd = .true.
    else
      mp_eqtrd = .false.
    endif
    return
  end function

  function mp_eqtir (ia, rb)
    implicit none
    logical mp_eqtir
    integer, intent (in):: ia
    type (mp_real), intent (in):: rb
    double precision da
    integer ic, mpfr_cmp_d
    external mpfr_cmp_d
    da = ia
    call mp_checkdp (da)
    call mp_fixlocr (rb)
    ic = - mpfr_cmp_d (rb%mpr(1), %val(da))
    if (ic == 0) then
      mp_eqtir = .true.
    else
      mp_eqtir = .false.
    endif
    return
  end function

  function mp_eqtri (ra, ib)
    implicit none
    logical mp_eqtri
    type (mp_real), intent (in):: ra
    integer, intent (in):: ib
    double precision db
    integer ic, mpfr_cmp_d
    external mpfr_cmp_d
    db = ib
    call mp_checkdp (db)
    call mp_fixlocr (ra)
    ic = mpfr_cmp_d (ra%mpr(1), %val(db))
    if (ic == 0) then
      mp_eqtri = .true.
    else
      mp_eqtri = .false.
    endif
    return
  end function

  function mp_eqtzz (za, zb)
    implicit none
    logical mp_eqtzz
    type (mp_complex), intent (in):: za, zb
    integer ic1, ic2, l1, l2
    integer mpfr_cmp
    external mpfr_cmp
    call mp_fixlocz (za)
    call mp_fixlocz (zb)
    l1 = za%mpc(0)
    l2 = zb%mpc(0)
    ic1 = mpfr_cmp (za%mpc(1), zb%mpc(1))
    ic2 = mpfr_cmp (za%mpc(l1+1), zb%mpc(l2+1))
    if (ic1 == 0 .and. ic2 == 0) then
      mp_eqtzz = .true.
    else
      mp_eqtzz = .false.
    endif
    return
  end function

  function mp_eqtdz (da, zb)
    implicit none
    logical mp_eqtdz
    double precision, intent (in):: da
    type (mp_complex), intent (in):: zb
    integer ic1, ic2, l2
    integer mpfr_cmp_d
    external mpfr_cmp_d
    call mp_checkdp (da)
    call mp_fixlocz (zb)
    l2 = zb%mpc(0)
    ic1 = - mpfr_cmp_d (zb%mpc(1), %val(da))
    if (zb%mpc(l2+3) == mpzero) then
      ic2 = 0
    else
      ic2 = 1
    endif
    if (ic1 == 0 .and. ic2 == 0) then
      mp_eqtdz = .true.
    else
      mp_eqtdz = .false.
    endif
    return
  end function

  function mp_eqtzd (za, db)
    implicit none
    logical mp_eqtzd
    type (mp_complex), intent (in):: za
    double precision, intent (in):: db
    integer ic1, ic2, l1
    integer mpfr_cmp_d
    external mpfr_cmp_d
    call mp_checkdp (db)
    call mp_fixlocz (za)
    l1 = za%mpc(0)
    ic1 = mpfr_cmp_d (za%mpc(1), %val(db))
    if (za%mpc(l1+3) == mpzero) then
      ic2 = 0
    else
      ic2 = 1
    endif
    if (ic1 == 0 .and. ic2 == 0) then
      mp_eqtzd = .true.
    else
      mp_eqtzd = .false.
    endif
    return
  end function

  function mp_eqtdcz (dca, zb)
    implicit none
    logical mp_eqtdcz
    complex (kind (0.d0)), intent (in):: dca
    type (mp_complex), intent (in):: zb
    integer ic1, ic2, l2
    integer mpfr_cmp_d
    external mpfr_cmp_d
    call mp_checkdp (dble (dca))
    call mp_checkdp (aimag (dca))
    call mp_fixlocz (zb)
    l2 = zb%mpc(0)
    ic1 = - mpfr_cmp_d (zb%mpc(1), %val(dble(dca)))
    ic2 = - mpfr_cmp_d (zb%mpc(l2+1), %val(aimag(dca)))
    if (ic1 == 0 .and. ic2 == 0) then
      mp_eqtdcz = .true.
    else
      mp_eqtdcz = .false.
    endif
    return
  end function

  function mp_eqtzdc (za, dcb)
    implicit none
    logical mp_eqtzdc
    type (mp_complex), intent (in):: za
    complex (kind (0.d0)), intent (in):: dcb
    integer ic1, ic2, l1
    integer mpfr_cmp_d
    external mpfr_cmp_d
    call mp_checkdp (dble (dcb))
    call mp_checkdp (aimag (dcb))
    call mp_fixlocz (za)
    l1 = za%mpc(0)
    ic1 = mpfr_cmp_d (za%mpc(1), %val(dble(dcb)))
    ic2 = mpfr_cmp_d (za%mpc(l1+1), %val(aimag(dcb)))
    if (ic1 == 0 .and. ic2 == 0) then
      mp_eqtzdc = .true.
    else
      mp_eqtzdc = .false.
    endif
    return
  end function

  function mp_eqtrz (ra, zb)
    implicit none
    logical mp_eqtrz
    type (mp_real), intent (in):: ra
    type (mp_complex), intent (in):: zb
    integer ic1, ic2, l2, mpfr_cmp
    external mpfr_cmp
    call mp_fixlocr (ra)
    call mp_fixlocz (zb)
    l2 = zb%mpc(0)
    ic1 = mpfr_cmp (ra%mpr(1), zb%mpc(1))
    if (zb%mpc(l2+3) == mpzero) then
      ic2 = 0
    else
      ic2 = 1
    endif
    if (ic1 == 0 .and. ic2 == 0) then
      mp_eqtrz = .true.
    else
      mp_eqtrz = .false.
    endif
    return
  end function

  function mp_eqtzr (za, rb)
    implicit none
    logical mp_eqtzr
    type (mp_complex), intent (in):: za
    type (mp_real), intent (in):: rb
    integer ic1, ic2, l1, mpfr_cmp
    external mpfr_cmp
    call mp_fixlocz (za)
    call mp_fixlocr (rb)
    l1 = za%mpc(0)
    ic1 = mpfr_cmp (za%mpc(1), rb%mpr(1))
    if (za%mpc(l1+3) == mpzero) then
      ic2 = 0
    else
      ic2 = 1
    endif
    if (ic1 == 0 .and. ic2 == 0) then
      mp_eqtzr = .true.
    else
      mp_eqtzr = .false.
    endif
    return
  end function

!  Non-equality test routines:

  function mp_netrr (ra, rb)
    implicit none
    logical mp_netrr
    type (mp_real), intent (in):: ra, rb
    integer ic, mpfr_cmp
    external mpfr_cmp
    call mp_fixlocr (ra)
    call mp_fixlocr (rb)
    ic = mpfr_cmp (ra%mpr(1), rb%mpr(1))
    if (ic /= 0) then
      mp_netrr = .true.
    else
      mp_netrr = .false.
    endif
    return
  end function

  function mp_netdr (da, rb)
    implicit none
    logical mp_netdr
    double precision, intent (in):: da
    type (mp_real), intent (in):: rb
    integer ic, mpfr_cmp_d
    external mpfr_cmp_d
    call mp_checkdp (da)
    call mp_fixlocr (rb)
    ic = - mpfr_cmp_d (rb%mpr(1), %val(da))
    if (ic /= 0) then
      mp_netdr = .true.
    else
      mp_netdr = .false.
    endif
    return
  end function

  function mp_netrd (ra, db)
    implicit none
    logical mp_netrd
    type (mp_real), intent (in):: ra
    double precision, intent (in):: db
    integer ic, mpfr_cmp_d
    external mpfr_cmp_d
    call mp_checkdp (db)
    call mp_fixlocr (ra)
    ic = mpfr_cmp_d (ra%mpr(1), %val(db))
    if (ic /= 0) then
      mp_netrd = .true.
    else
      mp_netrd = .false.
    endif
    return
  end function

  function mp_netir (ia, rb)
    implicit none
    logical mp_netir
    integer, intent (in):: ia
    type (mp_real), intent (in):: rb
    double precision da
    integer ic, mpfr_cmp_d
    external mpfr_cmp_d
    da = ia
    call mp_checkdp (da)
    call mp_fixlocr (rb)
    ic = - mpfr_cmp_d (rb%mpr(1), %val(da))
    if (ic /= 0) then
      mp_netir = .true.
    else
      mp_netir = .false.
    endif
    return
  end function

  function mp_netri (ra, ib)
    implicit none
    logical mp_netri
    type (mp_real), intent (in):: ra
    integer, intent (in):: ib
    double precision db
    integer ic, mpfr_cmp_d
    external mpfr_cmp_d
    db = ib
    call mp_checkdp (db)
    call mp_fixlocr (ra)
    ic = mpfr_cmp_d (ra%mpr(1), %val(db))
    if (ic /= 0) then
      mp_netri = .true.
    else
      mp_netri = .false.
    endif
    return
  end function

  function mp_netzz (za, zb)
    implicit none
    logical mp_netzz
    type (mp_complex), intent (in):: za, zb
    integer ic1, ic2, l1, l2
    integer mpfr_cmp
    external mpfr_cmp
    call mp_fixlocz (za)
    call mp_fixlocz (zb)
    l1 = za%mpc(0)
    l2 = zb%mpc(0)
    ic1 = mpfr_cmp (za%mpc(1), zb%mpc(1))
    ic2 = mpfr_cmp (za%mpc(l1+1), zb%mpc(l2+1))
    if (ic1 /= 0 .or. ic2 /= 0) then
      mp_netzz = .true.
    else
      mp_netzz = .false.
    endif
    return
  end function

  function mp_netdz (da, zb)
    implicit none
    logical mp_netdz
    double precision, intent (in):: da
    type (mp_complex), intent (in):: zb
    integer ic1, ic2, l2
    integer mpfr_cmp_d
    external mpfr_cmp_d
    call mp_checkdp (da)
    call mp_fixlocz (zb)
    l2 = zb%mpc(0)
    ic1 = - mpfr_cmp_d (zb%mpc(1), %val(da))
    if (zb%mpc(l2+3) == mpzero) then
      ic2 = 0
    else
      ic2 = 1
    endif
    if (ic1 /= 0 .or. ic2 /= 0) then
      mp_netdz = .true.
    else
      mp_netdz = .false.
    endif
    return
  end function

  function mp_netzd (za, db)
    implicit none
    logical mp_netzd
    type (mp_complex), intent (in):: za
    double precision, intent (in):: db
    integer ic1, ic2, l1
    integer mpfr_cmp_d
    external mpfr_cmp_d
    call mp_checkdp (db)
    call mp_fixlocz (za)
    l1 = za%mpc(0)
    ic1 = mpfr_cmp_d (za%mpc(1), %val(db))
    if (za%mpc(l1+3) == mpzero) then
      ic2 = 0
    else
      ic2 = 1
    endif
    if (ic1 /= 0 .or. ic2 /= 0) then
      mp_netzd = .true.
    else
      mp_netzd = .false.
    endif
    return
  end function

  function mp_netdcz (dca, zb)
    implicit none
    logical mp_netdcz
    complex (kind (0.d0)), intent (in):: dca
    type (mp_complex), intent (in):: zb
    integer ic1, ic2, l2
    integer mpfr_cmp_d
    external mpfr_cmp_d
    call mp_checkdp (dble (dca))
    call mp_checkdp (aimag (dca))
    call mp_fixlocz (zb)
    l2 = zb%mpc(0)
    ic1 = - mpfr_cmp_d (zb%mpc(1), %val(dble(dca)))
    ic2 = - mpfr_cmp_d (zb%mpc(l2+1), %val(aimag(dca)))
    if (ic1 /= 0 .or. ic2 /= 0) then
      mp_netdcz = .true.
    else
      mp_netdcz = .false.
    endif
    return
  end function

  function mp_netzdc (za, dcb)
    implicit none
    logical mp_netzdc
    type (mp_complex), intent (in):: za
    complex (kind (0.d0)), intent (in):: dcb
    integer ic1, ic2, l1
    integer mpfr_cmp_d
    external mpfr_cmp_d
    call mp_checkdp (dble (dcb))
    call mp_checkdp (aimag (dcb))
    call mp_fixlocz (za)
    l1 = za%mpc(0)
    ic1 = mpfr_cmp_d (za%mpc(1), %val(dble(dcb)))
    ic2 = mpfr_cmp_d (za%mpc(l1+1), %val(aimag(dcb)))
    if (ic1 /= 0 .or. ic2 /= 0) then
      mp_netzdc = .true.
    else
      mp_netzdc = .false.
    endif
    return
  end function

  function mp_netrz (ra, zb)
    implicit none
    logical mp_netrz
    type (mp_real), intent (in):: ra
    type (mp_complex), intent (in):: zb
    integer ic1, ic2, l2, mpfr_cmp
    external mpfr_cmp
    call mp_fixlocr (ra)
    call mp_fixlocz (zb)
    l2 = zb%mpc(0)
    ic1 = mpfr_cmp (ra%mpr(1), zb%mpc(1))
    if (zb%mpc(l2+3) == mpzero) then
      ic2 = 0
    else
      ic2 = 1
    endif
    if (ic1 /= 0 .or. ic2 /= 0) then
      mp_netrz = .true.
    else
      mp_netrz = .false.
    endif
    return
  end function

  function mp_netzr (za, rb)
    implicit none
    logical mp_netzr
    type (mp_complex), intent (in):: za
    type (mp_real), intent (in):: rb
    integer ic1, ic2, l1, mpfr_cmp
    external mpfr_cmp
    call mp_fixlocz (za)
    call mp_fixlocr (rb)
    l1 = za%mpc(0)
    ic1 = mpfr_cmp (za%mpc(1), rb%mpr(1))
    if (za%mpc(l1+3) == mpzero) then
      ic2 = 0
    else
      ic2 = 1
    endif
    if (ic1 /= 0 .or. ic2 /= 0) then
      mp_netzr = .true.
    else
      mp_netzr = .false.
    endif
    return
  end function

!  Less-than-or-equal test routines:

  function mp_letrr (ra, rb)
    implicit none
    logical mp_letrr
    type (mp_real), intent (in):: ra, rb
    integer ic, mpfr_cmp
    external mpfr_cmp
    call mp_fixlocr (ra)
    call mp_fixlocr (rb)
    ic = mpfr_cmp (ra%mpr(1), rb%mpr(1))
    if (ic <= 0) then
      mp_letrr = .true.
    else
      mp_letrr = .false.
    endif
    return
  end function

  function mp_letdr (da, rb)
    implicit none
    logical mp_letdr
    double precision, intent (in):: da
    type (mp_real), intent (in):: rb
    integer ic, mpfr_cmp_d
    external mpfr_cmp_d
    call mp_checkdp (da)
    call mp_fixlocr (rb)
    ic = - mpfr_cmp_d (rb%mpr(1), %val(da))
    if (ic <= 0) then
      mp_letdr = .true.
    else
      mp_letdr = .false.
    endif
    return
  end function

  function mp_letrd (ra, db)
    implicit none
    logical mp_letrd
    type (mp_real), intent (in):: ra
    double precision, intent (in):: db
    integer ic, mpfr_cmp_d
    external mpfr_cmp_d
    call mp_checkdp (db)
    call mp_fixlocr (ra)
    ic = mpfr_cmp_d (ra%mpr(1), %val(db))
    if (ic <= 0) then
      mp_letrd = .true.
    else
      mp_letrd = .false.
    endif
    return
  end function

  function mp_letir (ia, rb)
    implicit none
    logical mp_letir
    integer, intent (in):: ia
    type (mp_real), intent (in):: rb
    double precision da
    integer ic, mpfr_cmp_d
    external mpfr_cmp_d
    da = ia
    call mp_checkdp (da)
    call mp_fixlocr (rb)
    ic = - mpfr_cmp_d (rb%mpr(1), %val(da))
    if (ic <= 0) then
      mp_letir = .true.
    else
      mp_letir = .false.
    endif
    return
  end function

  function mp_letri (ra, ib)
    implicit none
    logical mp_letri
    type (mp_real), intent (in):: ra
    integer, intent (in):: ib
    double precision db
    integer ic, mpfr_cmp_d
    external mpfr_cmp_d
    db = ib
    call mp_checkdp (db)
    call mp_fixlocr (ra)
    ic = mpfr_cmp_d (ra%mpr(1), %val(db))
    if (ic <= 0) then
      mp_letri = .true.
    else
      mp_letri = .false.
    endif
    return
  end function

!  Greater-than-or-equal test routines:

  function mp_getrr (ra, rb)
    implicit none
    logical mp_getrr
    type (mp_real), intent (in):: ra, rb
    integer ic, mpfr_cmp
    external mpfr_cmp
    call mp_fixlocr (ra)
    call mp_fixlocr (rb)
    ic = mpfr_cmp (ra%mpr(1), rb%mpr(1))
    if (ic >= 0) then
      mp_getrr = .true.
    else
      mp_getrr = .false.
    endif
    return
  end function

  function mp_getdr (da, rb)
    implicit none
    logical mp_getdr
    double precision, intent (in):: da
    type (mp_real), intent (in):: rb
    integer ic, mpfr_cmp_d
    external mpfr_cmp_d
    call mp_checkdp (da)
    call mp_fixlocr (rb)
    ic = - mpfr_cmp_d (rb%mpr(1), %val(da))
    if (ic >= 0) then
      mp_getdr = .true.
    else
      mp_getdr = .false.
    endif
    return
  end function

  function mp_getrd (ra, db)
    implicit none
    logical mp_getrd
    type (mp_real), intent (in):: ra
    double precision, intent (in):: db
    integer ic, mpfr_cmp_d
    external mpfr_cmp_d
    call mp_checkdp (db)
    call mp_fixlocr (ra)
    ic = mpfr_cmp_d (ra%mpr(1), %val(db))
    if (ic >= 0) then
      mp_getrd = .true.
    else
      mp_getrd = .false.
    endif
    return
  end function

  function mp_getir (ia, rb)
    implicit none
    logical mp_getir
    integer, intent (in):: ia
    type (mp_real), intent (in):: rb
    double precision da
    integer ic, mpfr_cmp_d
    external mpfr_cmp_d
    da = ia
    call mp_checkdp (da)
    call mp_fixlocr (rb)
    ic = - mpfr_cmp_d (rb%mpr(1), %val(da))
    if (ic >= 0) then
      mp_getir = .true.
    else
      mp_getir = .false.
    endif
    return
  end function

  function mp_getri (ra, ib)
    implicit none
    logical mp_getri
    type (mp_real), intent (in):: ra
    integer, intent (in):: ib
    double precision db
    integer ic, mpfr_cmp_d
    external mpfr_cmp_d
    db = ib
    call mp_checkdp (db)
    call mp_fixlocr (ra)
    ic = mpfr_cmp_d (ra%mpr(1), %val(db))
    if (ic >= 0) then
      mp_getri = .true.
    else
      mp_getri = .false.
    endif
    return
  end function

!  Less-than test routines:

  function mp_lttrr (ra, rb)
    implicit none
    logical mp_lttrr
    type (mp_real), intent (in):: ra, rb
    integer ic, mpfr_cmp
    external mpfr_cmp
    call mp_fixlocr (ra)
    call mp_fixlocr (rb)
    ic = mpfr_cmp (ra%mpr(1), rb%mpr(1))
    if (ic < 0) then
      mp_lttrr = .true.
    else
      mp_lttrr = .false.
    endif
    return
  end function

  function mp_lttdr (da, rb)
    implicit none
    logical mp_lttdr
    double precision, intent (in):: da
    type (mp_real), intent (in):: rb
    integer ic, mpfr_cmp_d
    external mpfr_cmp_d
    call mp_checkdp (da)
    call mp_fixlocr (rb)
    ic = - mpfr_cmp_d (rb%mpr(1), %val(da))
    if (ic < 0) then
      mp_lttdr = .true.
    else
      mp_lttdr = .false.
    endif
    return
  end function

  function mp_lttrd (ra, db)
    implicit none
    logical mp_lttrd
    type (mp_real), intent (in):: ra
    double precision, intent (in):: db
    integer ic, mpfr_cmp_d
    external mpfr_cmp_d
    call mp_checkdp (db)
    call mp_fixlocr (ra)
    ic = mpfr_cmp_d (ra%mpr(1), %val(db))
    if (ic < 0) then
      mp_lttrd = .true.
    else
      mp_lttrd = .false.
    endif
    return
  end function

  function mp_lttir (ia, rb)
    implicit none
    logical mp_lttir
    integer, intent (in):: ia
    type (mp_real), intent (in):: rb
    double precision da
    integer ic, mpfr_cmp_d
    external mpfr_cmp_d
    da = ia
    call mp_checkdp (da)
    call mp_fixlocr (rb)
    ic = - mpfr_cmp_d (rb%mpr(1), %val(da))
    if (ic < 0) then
      mp_lttir = .true.
    else
      mp_lttir = .false.
    endif
    return
  end function

  function mp_lttri (ra, ib)
    implicit none
    logical mp_lttri
    type (mp_real), intent (in):: ra
    integer, intent (in):: ib
    double precision db
    integer ic, mpfr_cmp_d
    external mpfr_cmp_d
    db = ib
    call mp_checkdp (db)
    call mp_fixlocr (ra)
    ic = mpfr_cmp_d (ra%mpr(1), %val(db))
    if (ic < 0) then
      mp_lttri = .true.
    else
      mp_lttri = .false.
    endif
    return
  end function

!  Greater-than test routines:

  function mp_gttrr (ra, rb)
    implicit none
    logical mp_gttrr
    type (mp_real), intent (in):: ra, rb
    integer ic, mpfr_cmp
    external mpfr_cmp
    call mp_fixlocr (ra)
    call mp_fixlocr (rb)
    ic = mpfr_cmp (ra%mpr(1), rb%mpr(1))
    if (ic > 0) then
      mp_gttrr = .true.
    else
      mp_gttrr = .false.
    endif
    return
  end function

  function mp_gttdr (da, rb)
    implicit none
    logical mp_gttdr
    double precision, intent (in):: da
    type (mp_real), intent (in):: rb
    integer ic, mpfr_cmp_d
    external mpfr_cmp_d
    call mp_checkdp (da)
    call mp_fixlocr (rb)
    ic = - mpfr_cmp_d (rb%mpr(1), %val(da))
    if (ic > 0) then
      mp_gttdr = .true.
    else
      mp_gttdr = .false.
    endif
    return
  end function

  function mp_gttrd (ra, db)
    implicit none
    logical mp_gttrd
    type (mp_real), intent (in):: ra
    double precision, intent (in):: db
    integer ic, mpfr_cmp_d
    external mpfr_cmp_d
    call mp_checkdp (db)
    call mp_fixlocr (ra)
    ic = mpfr_cmp_d (ra%mpr(1), %val(db))
    if (ic > 0) then
      mp_gttrd = .true.
    else
      mp_gttrd = .false.
    endif
    return
  end function

  function mp_gttir (ia, rb)
    implicit none
    logical mp_gttir
    integer, intent (in):: ia
    type (mp_real), intent (in):: rb
    double precision da
    integer ic, mpfr_cmp_d
    external mpfr_cmp_d
    da = ia
    call mp_checkdp (da)
    call mp_fixlocr (rb)
    ic = - mpfr_cmp_d (rb%mpr(1), %val(da))
    if (ic > 0) then
      mp_gttir = .true.
    else
      mp_gttir = .false.
    endif
    return
  end function

  function mp_gttri (ra, ib)
    implicit none
    logical mp_gttri
    type (mp_real), intent (in):: ra
    integer, intent (in):: ib
    double precision db
    integer ic, mpfr_cmp_d
    external mpfr_cmp_d
    db = ib
    call mp_checkdp (db)
    call mp_fixlocr (ra)
    ic = mpfr_cmp_d (ra%mpr(1), %val(db))
    if (ic > 0) then
      mp_gttri = .true.
    else
      mp_gttri = .false.
    endif
    return
  end function

!  Algebraic and transcendental function definitions, listed alphabetically:

  function mp_absr (ra)
    implicit none
    type (mp_real):: mp_absr
    type (mp_real), intent (in):: ra
    integer mpnwbt
    call mp_fixlocr (ra)
    mpnwbt = min (ra%mpr(1), mpwdsbt)
    call mp_initvr (mp_absr, mpnwbt)
    call mpfr_abs (mp_absr%mpr(1), ra%mpr(1), %val(mprnd))
    return
  end function
  
  function mp_absz (za)
    implicit none
    type (mp_real):: mp_absz
    type (mp_complex), intent (in):: za
    integer l1, mpnwbt
    type (mp_real) r1, r2, r3
    call mp_fixlocz (za)
    l1 = za%mpc(0)
    mpnwbt = max (za%mpc(1), za%mpc(l1+1))
    mpnwbt = min (mpnwbt, mpwdsbt)
    call mp_initvr (r1, mpnwbt)
    call mp_initvr (r2, mpnwbt)
    call mp_initvr (r3, mpnwbt)
    call mp_initvr (mp_absz, mpnwbt)
    call mpfr_mul (r1%mpr(1), za%mpc(1), za%mpc(1), %val(mprnd))
    call mpfr_mul (r2%mpr(1), za%mpc(l1+1), za%mpc(l1+1), %val(mprnd))
    call mpfr_add (r3%mpr(1), r1%mpr(1), r2%mpr(1), %val(mprnd))
    call mpfr_sqrt (mp_absz%mpr(1), r3%mpr(1), %val(mprnd))
    return
  end function

  function mp_acos (ra)
    implicit none
    type (mp_real):: mp_acos
    type (mp_real), intent (in):: ra
    integer mpnwbt
    call mp_fixlocr (ra)
    mpnwbt = min (ra%mpr(1), mpwdsbt)
    call mp_initvr (mp_acos, mpnwbt)
    call mpfr_acos (mp_acos%mpr(1), ra%mpr(1), %val(mprnd))
    return
  end function
  
  function mp_acosh (ra)
    implicit none
    type (mp_real):: mp_acosh
    type (mp_real), intent (in):: ra
    integer mpnwbt
    call mp_fixlocr (ra)
    mpnwbt = min (ra%mpr(1), mpwdsbt)
    call mp_initvr (mp_acosh, mpnwbt)
    call mpfr_acosh (mp_acosh%mpr(1), ra%mpr(1), %val(mprnd))
    return
  end function
  
   function mp_agm (ra, rb)
    implicit none
    type (mp_real):: mp_agm
    type (mp_real), intent (in):: ra, rb
    integer mpnwbt
    call mp_fixlocr (ra)
    call mp_fixlocr (rb)
    mpnwbt = max (ra%mpr(1), rb%mpr(1))
    mpnwbt = min (mpnwbt, mpwdsbt)
    call mp_initvr (mp_agm, mpnwbt)
    call mpfr_agm (mp_agm%mpr(1), ra%mpr(1), rb%mpr(1), %val(mprnd))
    return
  end function

  function mp_aimag (za)
    implicit none
    type (mp_real):: mp_aimag
    type (mp_complex), intent (in):: za
    integer l1, mpnwbt
    call mp_fixlocz (za)
    l1 = za%mpc(0)
    mpnwbt = max (za%mpc(1), za%mpc(l1+1))
    mpnwbt = min (mpnwbt, mpwdsbt)
    call mp_initvr (mp_aimag, mpnwbt)
    call mpfr_set (mp_aimag%mpr(1), za%mpc(l1+1), %val(mprnd))
    return
  end function

  function mp_aint (ra)
    implicit none
    type (mp_real):: mp_aint
    type (mp_real), intent (in):: ra
    integer mpnwbt
    call mp_fixlocr (ra)
    mpnwbt = min (ra%mpr(1), mpwdsbt)
    call mp_initvr (mp_aint, mpnwbt)
    call mpfr_trunc (mp_aint%mpr(1), ra%mpr(1), %val(mprnd))
    return
  end function

  function mp_airy (ra)
    implicit none
    type (mp_real):: mp_airy
    type (mp_real), intent (in):: ra
    integer mpnwbt
    call mp_fixlocr (ra)
    mpnwbt = min (ra%mpr(1), mpwdsbt)
    call mp_initvr (mp_airy, mpnwbt)
    call mpfr_ai (mp_airy%mpr(1), ra%mpr(1), %val(mprnd))
    return
  end function
  
   function mp_anint (ra)
    implicit none
    type (mp_real):: mp_anint
    type (mp_real), intent (in):: ra
    integer mpnwbt
    call mp_fixlocr (ra)
    mpnwbt = min (ra%mpr(1), mpwdsbt)
    call mp_initvr (mp_anint, mpnwbt)
    call mpfr_round (mp_anint%mpr(1), ra%mpr(1), %val(mprnd))
    return
  end function

   function mp_asin (ra)
    implicit none
    type (mp_real):: mp_asin
    type (mp_real), intent (in):: ra
    type (mp_real) r1, r2, r3
    integer mpnwbt
    call mp_fixlocr (ra)
    mpnwbt = min (ra%mpr(1), mpwdsbt)
    call mp_initvr (mp_asin, mpnwbt)
    call mpfr_asin (mp_asin%mpr(1), ra%mpr(1), %val(mprnd))
    return
  end function

   function mp_asinh (ra)
    implicit none
    type (mp_real):: mp_asinh
    type (mp_real), intent (in):: ra
    type (mp_real) r1, r2, r3
    integer mpnwbt
    call mp_fixlocr (ra)
    mpnwbt = min (ra%mpr(1), mpwdsbt)
    call mp_initvr (mp_asinh, mpnwbt)
    call mpfr_asinh (mp_asinh%mpr(1), ra%mpr(1), %val(mprnd))
    return
  end function

   function mp_atan (ra)
    implicit none
    type (mp_real):: mp_atan
    type (mp_real), intent (in):: ra
    integer mpnwbt
    call mp_fixlocr (ra)
    mpnwbt = min (ra%mpr(1), mpwdsbt)
    call mp_initvr (mp_atan, mpnwbt)
    call mpfr_atan (mp_atan%mpr(1), ra%mpr(1), %val(mprnd))
    return
  end function

   function mp_atanh (ra)
    implicit none
    type (mp_real):: mp_atanh
    type (mp_real), intent (in):: ra
    integer mpnwbt
    call mp_fixlocr (ra)
    mpnwbt = min (ra%mpr(1), mpwdsbt)
    call mp_initvr (mp_atanh, mpnwbt)
    call mpfr_atanh (mp_atanh%mpr(1), ra%mpr(1), %val(mprnd))
    return
  end function

   function mp_atan2 (ra, rb)
    implicit none
    type (mp_real):: mp_atan2
    type (mp_real), intent (in):: ra, rb
    integer mpnwbt
    call mp_fixlocr (ra)
    call mp_fixlocr (rb)
    mpnwbt = max (ra%mpr(1), rb%mpr(1))
    mpnwbt = min (mpnwbt, mpwdsbt)
    call mp_initvr (mp_atan2, mpnwbt)
    call mpfr_atan2 (mp_atan2%mpr(1), ra%mpr(1), rb%mpr(1), %val(mprnd))
    return
  end function

  function mp_ator1 (a, ib, iprec)
    implicit none
    type (mp_real):: mp_ator1
    integer, intent (in):: ib
    character(1), intent (in):: a(ib)
    character(1) a1(ib+1)
    integer i, mpnwbt

!>  In variant #1, uncomment these six lines:
    integer, optional, intent (in):: iprec
    if (present (iprec)) then
      mpnwbt = mpnbt * mp_setwp (iprec)
    else
      mpnwbt = mpnbt * mpwds
    endif
!>  Otherwise in variant #2, uncomment these two lines:
!    integer, intent (in):: iprec
!    mpnwbt = mpnbt * mp_setwp (iprec)

    call mp_initvr (mp_ator1, mpnwbt)
    do i = 1, ib
      if (a(i) == 'D' .or. a(i) == 'd') then
        a1(i) = 'e'
      else
        a1(i) = a(i)
      endif
    enddo
    a1(ib+1) = char(0)
    call mpfrsetstr (mp_ator1%mpr(1), a1, mprnd)
    return
  end function

  function mp_atorn (aa, iprec)
    implicit none
    character(*), intent (in):: aa
    type (mp_real):: mp_atorn
    character(1) :: chr1(len(aa)+1)
    integer i, l1, mpnwbt

!>  In variant #1, uncomment these six lines:
    integer, optional, intent (in):: iprec
    if (present (iprec)) then
      mpnwbt = mpnbt * mp_setwp (iprec)
    else
      mpnwbt = mpnbt * mpwds
    endif
!>  Otherwise in variant #2, uncomment these two lines:
!    integer, intent (in):: iprec
!    mpnwbt = mpnbt * mp_setwp (iprec)

    l1 = len (aa)
    call mp_initvr (mp_atorn, mpnwbt)
    do i = 1, l1
      if (aa(i:i) == 'D' .or. aa(i:i) == 'd') then
        chr1(i) = 'e'
      else
        chr1(i) = aa(i:i)
      endif
    enddo
    chr1(l1+1) = char(0)
    call mpfrsetstr (mp_atorn%mpr(1), chr1, mprnd)
    return
  end function
  
  subroutine mp_berne (nb, rb, iprec)
    implicit none
    integer, intent (in):: nb
    type (mp_real), intent (out):: rb(nb)
    integer i, n1, mpnwbt

!>  In variant #1, uncomment these six lines:
    integer, optional, intent (in):: iprec
    if (present (iprec)) then
      mpnwbt = mpnbt * mp_setwp (iprec)
    else
      mpnwbt = mpnbt * mpwds
    endif
!>  Otherwise in variant #2, uncomment these two lines:
!    integer, intent (in):: iprec
!    mpnwbt = mpnbt * mp_setwp (iprec)

    do i = 1, nb
      call mp_initvr (rb(i), mpnwbt)
    enddo

!   This is implemented yet.
    call mp_abrt (1)
    return
  end subroutine

  function mp_besselj (ra, rb)
    implicit none
    type (mp_real):: mp_besselj
    type (mp_real), intent (in):: ra, rb

!   This is implemented yet.
    mp_besselj%mpr(0) = mpwds6
    call mp_abrt (1)
    return
  end function
  
  function mp_bessel_j0 (ra)
    implicit none
    type (mp_real):: mp_bessel_j0
    type (mp_real), intent (in):: ra
    integer mpnwbt
    call mp_fixlocr (ra)
    mpnwbt = min (ra%mpr(1), mpwdsbt)
    call mp_initvr (mp_bessel_j0, mpnwbt)
    call mpfr_j0 (mp_bessel_j0%mpr(1), ra%mpr(1), %val(mprnd))
    return
  end function

  function mp_bessel_j1 (ra)
    implicit none
    type (mp_real):: mp_bessel_j1
    type (mp_real), intent (in):: ra
    integer mpnwbt
    call mp_fixlocr (ra)
    mpnwbt = min (ra%mpr(1), mpwdsbt)
    call mp_initvr (mp_bessel_j1, mpnwbt)
    call mpfr_j1 (mp_bessel_j1%mpr(1), ra%mpr(1), %val(mprnd))
    return
  end function

  function mp_bessel_jn (ia, rb)
    implicit none
    type (mp_real):: mp_bessel_jn
    integer, intent (in):: ia
    type (mp_real), intent (in):: rb
    integer mpnwbt
    call mp_fixlocr (rb)
    mpnwbt = min (rb%mpr(1), mpwdsbt)
    call mp_initvr (mp_bessel_jn, mpnwbt)
    call mpfrbesseljn (mp_bessel_jn%mpr(1), ia, rb%mpr(1), mprnd)
    return
  end function

  function mp_bessel_y0 (ra)
    implicit none
    type (mp_real):: mp_bessel_y0
    type (mp_real), intent (in):: ra
    type (mp_real) r1
    integer mpnwbt
    call mp_fixlocr (ra)
    mpnwbt = min (ra%mpr(1), mpwdsbt)
    call mp_initvr (mp_bessel_y0, mpnwbt)
    call mpfr_y0 (mp_bessel_y0%mpr(1), ra%mpr(1), %val(mprnd))
    return
  end function

  function mp_bessel_y1 (ra)
    implicit none
    type (mp_real):: mp_bessel_y1
    type (mp_real), intent (in):: ra
    type (mp_real) r1
    integer mpnwbt
    call mp_fixlocr (ra)
    mpnwbt = min (ra%mpr(1), mpwdsbt)
    call mp_initvr (mp_bessel_y1, mpnwbt)
    call mpfr_y1 (mp_bessel_y1%mpr(1), ra%mpr(1), %val(mprnd))
    return
  end function

  function mp_bessel_yn (ia, rb)
    implicit none
    type (mp_real):: mp_bessel_yn
    integer, intent (in):: ia
    type (mp_real), intent (in):: rb
    integer mpnwbt
    call mp_fixlocr (rb)
    mpnwbt = min (rb%mpr(1), mpwdsbt)
    call mp_initvr (mp_bessel_yn, mpnwbt)
    call mpfrbesselyn (mp_bessel_yn%mpr(1), ia, rb%mpr(1), mprnd)
    return
  end function

  function mp_ccos (za)
    implicit none
    type (mp_complex):: mp_ccos
    type (mp_complex), intent (in):: za
    integer l1, l2, l3, mpnwbt
    type (mp_real) r1
    type (mp_complex) z1, z2, z3
    call mp_fixlocz (za)
    l1 = za%mpc(0)
    mpnwbt = max (za%mpc(1), za%mpc(l1+1))
    mpnwbt = min (mpnwbt, mpwdsbt)
    l2 = mpwds6
    call mp_initvz (z1, mpnwbt)
    call mp_initvz (z2, mpnwbt)
    call mp_initvz (z3, mpnwbt)
    l3 = mpwds6
    call mp_initvz (mp_ccos, mpnwbt)
    call mpfrsetd (z1%mpc(1), 1.d0, mprnd)
    call mpfrsetd (z1%mpc(l2+1), 0.d0, mprnd)
    call mpfr_set (z3%mpc(l2+1), za%mpc(1), %val(mprnd))
    call mpfr_set (z3%mpc(1), za%mpc(l1+1), %val(mprnd))
    call mpfr_neg (z3%mpc(1), z3%mpc(1), %val(mprnd))
    z2 = mp_cexp (z3)
    z3 = mp_divzz (z1, z2)
    z1 = mp_addzz (z2, z3)
    call mpfrmuld (mp_ccos%mpc(1), z1%mpc(1), 0.5d0, mprnd)
    call mpfrmuld (mp_ccos%mpc(l3+1), z1%mpc(l2+1), 0.5d0, mprnd)
    return
  end function
  
  function mp_cexp (za)
    implicit none
    type (mp_complex):: mp_cexp
    type (mp_complex), intent (in):: za
    integer l1, l2, mpnwbt
    type (mp_real) r1
    call mp_fixlocz (za)
    l1 = za%mpc(0)
    mpnwbt = max (za%mpc(1), za%mpc(l1+1))
    mpnwbt = min (mpnwbt, mpwdsbt)
    call mp_initvr (r1, mpnwbt)
    l2 = mpwds6
    call mp_initvz (mp_cexp, mpnwbt)
    call mpfr_sin_cos (mp_cexp%mpc(l2+1), mp_cexp%mpc(1), za%mpc(l1+1), %val(mprnd))
    call mpfr_exp (r1%mpr(1), za%mpc(1), %val(mprnd))
    call mpfr_mul (mp_cexp%mpc(1), mp_cexp%mpc(1), r1%mpr(1), %val(mprnd))
    call mpfr_mul (mp_cexp%mpc(l2+1), mp_cexp%mpc(l2+1), r1%mpr(1), %val(mprnd))
    return
  end function
  
  function mp_clog (za)
    implicit none
    type (mp_complex):: mp_clog
    type (mp_complex), intent (in):: za
    integer l1, l2, mpnwbt
    type (mp_real) r1, r2, r3
    call mp_fixlocz (za)
    l1 = za%mpc(0)
    mpnwbt = max (za%mpc(1), za%mpc(l1+1))
    mpnwbt = min (mpnwbt, mpwdsbt)
    call mp_initvr (r1, mpnwbt)
    call mp_initvr (r2, mpnwbt)
    call mp_initvr (r3, mpnwbt)
    l2 = mpwds6
    call mp_initvz (mp_clog, mpnwbt)
    call mpfr_mul (r1%mpr(1), za%mpc(1), za%mpc(1), %val(mprnd))
    call mpfr_mul (r2%mpr(1), za%mpc(l1+1), za%mpc(l1+1), %val(mprnd))
    call mpfr_add (r3%mpr(1), r1%mpr(1), r2%mpr(1), %val(mprnd))
    call mpfr_log (r1%mpr(1), r3%mpr(1), %val(mprnd))
    call mpfrmuld (mp_clog%mpc(1), r1%mpr(1), 0.5d0, mprnd)
    call mpfr_atan2 (mp_clog%mpc(l2+1), za%mpc(l1+1), za%mpc(1), %val(mprnd))
    return
  end function
  
  function mp_conjg (za)
    implicit none
    type (mp_complex):: mp_conjg
    type (mp_complex), intent (in):: za
    integer l1, l2, mpnwbt
    call mp_fixlocz (za)
    l1 = za%mpc(0)
    mpnwbt = max (za%mpc(1), za%mpc(l1+1))
    mpnwbt = min (mpnwbt, mpwdsbt)
    l2 = mpwds6
    call mp_initvz (mp_conjg, mpnwbt)
    call mpfr_set (mp_conjg%mpc(1), za%mpc(1), %val(mprnd))
    call mpfr_set (mp_conjg%mpc(l2+1), za%mpc(l1+1), %val(mprnd))
    call mpfr_neg (mp_conjg%mpc(l2+1), mp_conjg%mpc(l2+1), %val(mprnd))
    return
  end function
  
  function mp_cos (ra)
    implicit none
    type (mp_real):: mp_cos
    type (mp_real), intent (in):: ra
    integer mpnwbt
    call mp_fixlocr (ra)
    mpnwbt = min (ra%mpr(1), mpwdsbt)
    call mp_initvr (mp_cos, mpnwbt)
    call mpfr_cos (mp_cos%mpr(1), ra%mpr(1), %val(mprnd))
    return
  end function
  
  function mp_cosh (ra)
    implicit none
    type (mp_real):: mp_cosh
    type (mp_real), intent (in):: ra
    integer mpnwbt
    call mp_fixlocr (ra)
    mpnwbt = min (ra%mpr(1), mpwdsbt)
    call mp_initvr (mp_cosh, mpnwbt)
    call mpfr_cosh (mp_cosh%mpr(1), ra%mpr(1), %val(mprnd))
    return
  end function
   
  function mp_csin (za)
    implicit none
    type (mp_complex):: mp_csin
    type (mp_complex), intent (in):: za
    integer l1, l2, l3, mpnwbt
    type (mp_real) r1
    type (mp_complex) z1, z2, z3
    call mp_fixlocz (za)
    l1 = za%mpc(0)
    mpnwbt = max (za%mpc(1), za%mpc(l1+1))
    mpnwbt = min (mpnwbt, mpwdsbt)
    l2 = mpwds6
    call mp_initvz (z1, mpnwbt)
    call mp_initvz (z2, mpnwbt)
    call mp_initvz (z3, mpnwbt)
    l3 = mpwds6
    call mp_initvz (mp_csin, mpnwbt)
    call mpfrsetd (z1%mpc(1), 1.d0, mprnd)
    call mpfrsetd (z1%mpc(l2+1), 0.d0, mprnd)
    call mpfr_set (z3%mpc(l2+1), za%mpc(1), %val(mprnd))
    call mpfr_set (z3%mpc(1), za%mpc(l1+1), %val(mprnd))
    call mpfr_neg (z3%mpc(1), z3%mpc(1), %val(mprnd))
    z2 = mp_cexp (z3)
    z3 = mp_divzz (z1, z2)
    z1 = mp_subzz (z2, z3)
    call mpfrmuld (mp_csin%mpc(l3+1), z1%mpc(1), 0.5d0, mprnd)
    call mpfrmuld (mp_csin%mpc(1), z1%mpc(l2+1), 0.5d0, mprnd)
    call mpfr_neg (mp_csin%mpc(l3+1), mp_csin%mpc(l3+1), %val(mprnd))
    return
  end function
  
  function mp_csqrt (za)
    implicit none
    type (mp_complex):: mp_csqrt
    type (mp_complex), intent (in):: za
    integer l1, l2, mpnwbt
    type (mp_real) r1, r2, r3
    call mp_fixlocz (za)
    l1 = za%mpc(0)
    mpnwbt = max (za%mpc(1), za%mpc(l1+1))
    mpnwbt = min (mpnwbt, mpwdsbt)
    call mp_initvr (r1, mpnwbt)
    call mp_initvr (r2, mpnwbt)
    call mp_initvr (r3, mpnwbt)
    l2 = mpwds6
    call mp_initvz (mp_csqrt, mpnwbt)
    call mpfr_mul (r1%mpr(1), za%mpc(1), za%mpc(1), %val(mprnd))
    call mpfr_mul (r2%mpr(1), za%mpc(l1+1), za%mpc(l1+1), %val(mprnd))
    call mpfr_add (r3%mpr(1), r1%mpr(1), r2%mpr(1), %val(mprnd))
    call mpfr_sqrt (r1%mpr(1), r3%mpr(1), %val(mprnd))

    if (za%mpc(2) == 1) then
      call mpfr_add (r2%mpr(1), r1%mpr(1), za%mpc(1), %val(mprnd))
      call mpfr_sqrt (r3%mpr(1), r2%mpr(1), %val(mprnd))
      call mpfr_set (mp_csqrt%mpc(1), r3%mpr(1), %val(mprnd))
      call mpfr_div (mp_csqrt%mpc(l2+1), za%mpc(l1+1), r3%mpr(1), %val(mprnd))
    else
      call mpfr_sub (r2%mpr(1), r1%mpr(1), za%mpc(1), %val(mprnd))
      call mpfr_sqrt (r3%mpr(1), r2%mpr(1), %val(mprnd))
      call mpfr_abs (r2%mpr(1), za%mpc(l1+1), %val(mprnd))
      call mpfr_div (mp_csqrt%mpc(1), r2%mpr(1), r3%mpr(1), %val(mprnd))
      call mpfr_set (mp_csqrt%mpc(l2+1), r3%mpr(1), %val(mprnd))
      if (za%mpc(l1+2) /= 1) &
         call mpfr_neg (mp_csqrt%mpc(l2+1), mp_csqrt%mpc(l2+1), %val(mprnd))
    endif

    call mpfrsetd (r1%mpr(1), 2.d0, mprnd)
    call mpfr_sqrt (r2%mpr(1), r1%mpr(1), %val(mprnd))
    call mpfr_div (mp_csqrt%mpc(1), mp_csqrt%mpc(1), r2%mpr(1), %val(mprnd))
    call mpfr_div (mp_csqrt%mpc(l2+1), mp_csqrt%mpc(l2+1), r2%mpr(1), %val(mprnd))
    return
  end function
  
  subroutine mp_cssh (ra, rb, rc)
    implicit none
    type (mp_real), intent (in):: ra
    type (mp_real), intent (out):: rb, rc
    integer mpnwbt
    call mp_fixlocr (ra)
    mpnwbt = min (ra%mpr(1), mpwdsbt)
    call mp_initvr (rb, mpnwbt)
    call mp_initvr (rc, mpnwbt)
    call mpfr_sinh_cosh (rc%mpr(1), rb%mpr(1), ra%mpr(1), %val(mprnd))
    return
  end subroutine
  
  subroutine mp_cssn (ra, rb, rc)
    implicit none
    type (mp_real), intent (in):: ra
    type (mp_real), intent (out):: rb, rc
    integer mpnwbt
    call mp_fixlocr (ra)
    mpnwbt = min (ra%mpr(1), mpwdsbt)
    call mp_initvr (rb, mpnwbt)
    call mp_initvr (rc, mpnwbt)
    call mpfr_sin_cos (rc%mpr(1), rb%mpr(1), ra%mpr(1), %val(mprnd))
    return
  end subroutine
  
  function mp_dctoz (dca, iprec)
    implicit none
    type (mp_complex):: mp_dctoz
    complex (kind(0.d0)), intent (in):: dca
    integer l1, l2, mpnwbt

!>  In variant #1, uncomment these six lines:
    integer, optional, intent (in):: iprec
    if (present (iprec)) then
      mpnwbt = mpnbt * mp_setwp (iprec)
    else
      mpnwbt = mpnbt * mpwds
    endif
!>  Otherwise in variant #2, uncomment these two lines:
!    integer, intent (in):: iprec
!    mpnwbt = mpnbt * mp_setwp (iprec)

    call mp_checkdp (dble (dca))
    call mp_checkdp (aimag (dca))
    l2 = mpwds6
    call mp_initvz (mp_dctoz, mpnwbt)
    call mpfrsetd (mp_dctoz%mpc(1), dble (dca), mprnd)
    call mpfrsetd (mp_dctoz%mpc(l2+1), aimag (dca), mprnd)
    return
  end function
  
  function mp_dctoz2 (dca, iprec)
    implicit none
    type (mp_complex):: mp_dctoz2
    complex (kind(0.d0)), intent (in):: dca
    integer l1, l2, mpnwbt

!>  In variant #1, uncomment these six lines:
    integer, optional, intent (in):: iprec
    if (present (iprec)) then
      mpnwbt = mpnbt * mp_setwp (iprec)
    else
      mpnwbt = mpnbt * mpwds
    endif
!>  Otherwise in variant #2, uncomment these two lines:
!    integer, intent (in):: iprec
!    mpnwbt = mpnbt * mp_setwp (iprec)

    l2 = mpwds6
    call mp_initvz (mp_dctoz2, mpnwbt)
    call mpfrsetd (mp_dctoz2%mpc(1), dble (dca), mprnd)
    call mpfrsetd (mp_dctoz2%mpc(l2+1), aimag (dca), mprnd)
    return
  end function
  
  function mp_digamma (ra)
    implicit none
    type (mp_real):: mp_digamma
    type (mp_real), intent (in):: ra
    integer mpnwbt
    call mp_fixlocr (ra)
    mpnwbt = min (ra%mpr(1), mpwdsbt)
    call mp_initvr (mp_digamma, mpnwbt)
    call mpfr_digamma (mp_digamma%mpr(1), ra%mpr(1), %val(mprnd))
    return
  end function

  function mp_dtor (da, iprec)
    implicit none
    type (mp_real):: mp_dtor
    double precision, intent (in):: da
    integer mpnwbt

!>  In variant #1, uncomment these six lines:
    integer, optional, intent (in):: iprec
    if (present (iprec)) then
      mpnwbt = mpnbt * mp_setwp (iprec)
    else
      mpnwbt = mpnbt * mpwds
    endif
!>  Otherwise in variant #2, uncomment these two lines:
!    integer, intent (in):: iprec
!    mpnwbt = mpnbt * mp_setwp (iprec)
    
    call mp_checkdp (da)
    call mp_initvr (mp_dtor, mpnwbt)
    call mpfrsetd (mp_dtor%mpr(1), da, mprnd)    
    return
  end function
 
  function mp_dtor2 (da, iprec)
    implicit none
    type (mp_real):: mp_dtor2
    double precision, intent (in):: da
    integer mpnwbt

!>  In variant #1, uncomment these six lines:
    integer, optional, intent (in):: iprec
    if (present (iprec)) then
      mpnwbt = mpnbt * mp_setwp (iprec)
    else
      mpnwbt = mpnbt * mpwds
    endif
!>  Otherwise in variant #2, uncomment these two lines:
!    integer, intent (in):: iprec
!    mpnwbt = mpnbt * mp_setwp (iprec)

    call mp_initvr (mp_dtor2, mpnwbt)
    call mpfrsetd (mp_dtor2%mpr(1), da, mprnd)
    return
  end function
 
  subroutine mp_eform (ra, nb, nd, b)
    implicit none
    type (mp_real), intent (in):: ra
    integer, intent (in):: nb, nd
    character(1), intent (out):: b(nb)
    character(1) b1(nd+8)
    integer ix, i, j
    integer(8) iexp
    character(16) str16

    call mp_fixlocr (ra)

!  Check for overflow of field.
    if (nb < nd + 20) then
      do i = 1, nb
        b(i) = '*'
      enddo
      return
    endif

!  Call mpfrgetstr to convert number.

    call mpfrgetstr (ra%mpr(1), b1, nd, iexp, mprnd)

    if (b1(1) == '-') then
      b(1) = '-'
      b(2) = b1(2)
      b(3) = '.' 
      do i = 1, nd - 2
        b(i+3) = b1(i+2)
      enddo 
      ix = nd + 1
    else
      b(1) = b1(1)
      b(2) = '.'
      do i = 1, nd - 2
        b(i+2) = b1(i+1)
      enddo
      ix = nd
    endif
!  Insert exponent.
    b(ix+1) = 'e'
    ix = ix + 1
    write (str16, '(i16)') iexp - 1
    do i = 1, 16
      if (str16(i:i) /= ' ') goto 100
    enddo
100 continue
    do j = 1, 16 - i + 1
      b(ix+j) = str16(i+j-1:i+j-1)
    enddo
    ix = ix + 16 - i + 1
    do j = ix + 1, nb
      b(j) = ' '
    enddo

    return
  end subroutine

  function mp_egamma (iprec)
    implicit none
    type (mp_real):: mp_egamma
    integer mpnwbt

!>  In variant #1, uncomment these six lines:
    integer, optional, intent (in):: iprec
    if (present (iprec)) then
      mpnwbt = mpnbt * mp_setwp (iprec)
    else
      mpnwbt = mpnbt * mpwds
    endif
!>  Otherwise in variant #2, uncomment these two lines:
!    integer, intent (in):: iprec
!    mpnwbt = mpnbt * mp_setwp (iprec)

    call mp_initvr (mp_egamma, mpnwbt)
    call mpfr_const_euler (mp_egamma%mpr(1), %val(mprnd))
  end function

  function mp_expint (ra)
    implicit none
    type (mp_real):: mp_expint
    type (mp_real), intent (in):: ra
    integer mpnwbt
    call mp_fixlocr (ra)
    mpnwbt = min (ra%mpr(1), mpwdsbt)
    call mp_initvr (mp_expint, mpnwbt)
    call mpfr_eint (mp_expint%mpr(1), ra%mpr(1), %val(mprnd))
    return
  end function

  function mp_erf (ra)
    implicit none
    type (mp_real):: mp_erf
    type (mp_real), intent (in):: ra
    integer mpnwbt
    call mp_fixlocr (ra)
    mpnwbt = min (ra%mpr(1), mpwdsbt)
    call mp_initvr (mp_erf, mpnwbt)
    call mpfr_erf (mp_erf%mpr(1), ra%mpr(1), %val(mprnd))
    return
  end function

  function mp_erfc (ra)
    implicit none
    type (mp_real):: mp_erfc
    type (mp_real), intent (in):: ra
    integer mpnwbt
    call mp_fixlocr (ra)
    mpnwbt = min (ra%mpr(1), mpwdsbt)
    call mp_initvr (mp_erfc, mpnwbt)
    call mpfr_erfc (mp_erfc%mpr(1), ra%mpr(1), %val(mprnd))
    return
  end function

  function mp_exp (ra)
    implicit none
    type (mp_real):: mp_exp
    type (mp_real), intent (in):: ra
    integer mpnwbt
    call mp_fixlocr (ra)
    mpnwbt = min (ra%mpr(1), mpwdsbt)
    call mp_initvr (mp_exp, mpnwbt)
    call mpfr_exp (mp_exp%mpr(1), ra%mpr(1), %val(mprnd))
    return
  end function

  subroutine mp_fform (ra, nb, nd, b)
    implicit none
    type (mp_real), intent (in):: ra
    integer, intent (in):: nb, nd
    character(1), intent (out):: b(nb)
    character(1) b1(nb+8)
    integer ix, i, j
    integer(8) iexp
    character(16) str16

    call mp_fixlocr (ra)

!  Call mpfrgetstr to convert numbmer.

    call mpfrgetstr (ra%mpr(1), b1, nb, iexp, mprnd)

!  Check for overflow of field.

    if ( &
      b1(1) /= '-' .and. iexp > 0 .and. nb < iexp + nd + 1 .or. &
      b1(1) /= '-' .and. iexp <= 0 .and. nb < nd + 2 .or. &
      b1(1) == '-' .and. iexp > 0 .and. nb < iexp + nd + 2 .or. &
      b1(1) == '-' .and. iexp <= 0 .and. nb < nd + 3) then
      do i = 1, nb
        b(i) = '*'
      enddo
    endif        
    if (b1(1) == '0' .or. iexp <= - nd) then

!  Output is zero.

      do i = 1, nb - nd - 2
        b(i) = ' '
      enddo
      ix = nb - nd - 2
      b(ix+1) = '0'
      b(ix+2) = '.'
      ix = ix + 2
      do i = 1, nd
        b(ix+i) = '0'
      enddo
    elseif (b1(1) /= '-' .and. iexp > 0) then

!  Value is positive and exponent is positive.

      do i = 1, nb - iexp - nd - 1
        b(i) = ' '
      enddo
      ix = nb - iexp - nd - 1
      do i = 1, iexp
        b(ix+i) = b1(i)
      enddo
      ix = ix + iexp
      b(ix+1) = '.'
      ix = ix + 1
      do i = 1, nd
        b(ix+i) = b1(i+iexp)
      enddo
    elseif (b1(1) /= '-' .and. iexp <= 0) then

!  Value is positive and exponent is negative or zero.

      do i = 1, nb - nd - 2
        b(i) = ' '
      enddo
      ix = nb - nd - 2
      b(ix+1) = '0'
      b(ix+2) = '.'
      ix = ix + 2
      do i = 1, abs (iexp)
        b(ix+i) = '0'
      enddo
      ix = ix + abs (iexp)
      do i = 1, nd - abs (iexp)
        b(ix+i) = b1(i)
      enddo
    elseif (b1(1) == '-' .and. iexp > 0) then

!  Value is negative and exponent is positive.

      do i = 1, nb - iexp - nd - 2
        b(i) = ' '
      enddo
      ix = nb - iexp - nd - 2
      do i = 1, iexp + 1
        b(ix+i) = b1(i)
      enddo
      ix = ix + iexp + 1
      b(ix+1) = '.'
      ix = ix + 1
      do i = 1, nd
        b(ix+i) = b1(i+iexp+1)
      enddo
    elseif (b1(1) /= '-' .and. iexp <= 0) then

!  Value is negative and exponent is negative or zero.

      do i = 1, nb - nd - 3
        b(i) = ' '
      enddo
      ix = nb - nd - 3
      b(ix+1) = '-'
      b(ix+2) = '0'
      b(ix+3) = '.'
      ix = ix + 3
      do i = 1, abs (iexp)
        b(ix+i) = '0'
      enddo
      ix = ix + abs (iexp)
      do i = 1, nd - abs (iexp)
        b(ix+i) = b1(i)
      enddo
    endif
    return
  end subroutine
 
  function mp_gamma (ra)
    implicit none
    type (mp_real):: mp_gamma
    type (mp_real), intent (in):: ra
    integer mpnwbt
    call mp_fixlocr (ra)
    mpnwbt = min (ra%mpr(1), mpwdsbt)
    call mp_initvr (mp_gamma, mpnwbt)
    call mpfr_gamma (mp_gamma%mpr(1), ra%mpr(1), %val(mprnd))
    return
  end function
  
  function mp_gammainc (ra, rb)
    implicit none
    type (mp_real):: mp_gammainc
    type (mp_real), intent (in):: ra, rb
    integer mpnw
    mpnw = max (int (ra%mpr(1)), int (rb%mpr(1)))
    mpnw = min (mpnw, mpwds)
    mp_gammainc%mpr(0) = mpwds6

!  This is not implemented yet
    call mp_abrt (1)
    return
  end function
  
  function mp_hypot (ra, rb)
    implicit none
    type (mp_real):: mp_hypot
    type (mp_real), intent (in):: ra, rb
    integer mpnwbt
    call mp_fixlocr (ra)
    call mp_fixlocr (rb)
    mpnwbt = max (ra%mpr(1), rb%mpr(1))
    mpnwbt = min (mpnwbt, mpwdsbt)
    call mp_initvr (mp_hypot, mpnwbt)
    call mpfr_hypot (mp_hypot%mpr(1), ra%mpr(1), rb%mpr(1), %val(mprnd))
    return
  end function
    
  subroutine mp_init (iprec)
    implicit none
    integer mpnwbt

!>  In variant #1, uncomment these six lines:
    integer, optional, intent (in):: iprec
    if (present (iprec)) then
      mpnwbt = mpnbt * mp_setwp (iprec)
    else
      mpnwbt = mpnbt * mpwds
    endif
!>  Otherwise in variant #2, uncomment these two lines:
!    integer, intent (in):: iprec
!    mpnwbt = mpnbt * mp_setwp (iprec)

!   This is not implemented yet.
    call mp_abrt (1)
    return
  end subroutine
  
  function mp_log (ra)
    implicit none
    type (mp_real):: mp_log
    type (mp_real), intent (in):: ra
    integer mpnwbt
    call mp_fixlocr (ra)
    mpnwbt = min (ra%mpr(1), mpwdsbt)
    call mp_initvr (mp_log, mpnwbt)
    call mpfr_log (mp_log%mpr(1), ra%mpr(1), %val(mprnd))
    return
  end function

  function mp_log_gamma (ra)
    implicit none
    type (mp_real):: mp_log_gamma
    type (mp_real), intent (in):: ra
    integer mpnwbt
    call mp_fixlocr (ra)
    mpnwbt = min (ra%mpr(1), mpwdsbt)
    call mp_initvr (mp_log_gamma, mpnwbt)
    call mpfr_lngamma (mp_log_gamma%mpr(1), ra%mpr(1), %val(mprnd))
    return
  end function

  function mp_log2 (iprec)
    implicit none
    type (mp_real):: mp_log2
    type (mp_real) qpi
    integer mpnwbt

!>  In variant #1, uncomment these six lines:
    integer, optional, intent (in):: iprec
    if (present (iprec)) then
      mpnwbt = mpnbt * mp_setwp (iprec)
    else
      mpnwbt = mpnbt * mpwds
    endif
!>  Otherwise in variant #2, uncomment these two lines:
!    integer, intent (in):: iprec
!    mpnwbt = mpnbt * mp_setwp (iprec)

    call mp_initvr (mp_log2, mpnwbt)
    call mpfr_const_log2 (mp_log2%mpr(1), %val(mprnd))
  end function

  function mp_max (ra, rb, rc)
    implicit none
    type (mp_real):: mp_max
    type (mp_real), intent (in):: ra, rb
    type (mp_real), optional, intent (in):: rc
    integer mpnwbt
    call mp_fixlocr (ra)
    call mp_fixlocr (rb)
    mpnwbt = max (ra%mpr(1), rb%mpr(1))
    mpnwbt = min (mpnwbt, mpwdsbt)
    call mp_initvr (mp_max, mpnwbt)
    call mpfr_max (mp_max%mpr(1), ra%mpr(1), rb%mpr(1), %val(mprnd))
    if (present (rc)) then
    call mp_fixlocr (rc)
      call mpfr_max (mp_max%mpr(1), mp_max%mpr(1), rc%mpr(1), %val(mprnd))
    endif
    return
  end function
  
  subroutine mp_mdi (ra, db, ic)
    implicit none
    integer(8) ic8
    type (mp_real), intent (in):: ra
    double precision, intent (out):: db
    integer, intent (out):: ic
    double precision mpfr_get_d_2exp
    external mpfr_get_d_2exp
    call mp_fixlocr (ra)
    db = 2.d0 * mpfr_get_d_2exp (ic8, ra%mpr(1), %val(mprnd))
    if (db /= 0.d0) ic = ic8 - 1
    return
  end subroutine

  function mp_min (ra, rb, rc)
    implicit none
    type (mp_real):: mp_min
    type (mp_real), intent (in):: ra, rb
    type (mp_real), optional, intent (in):: rc
    integer mpnwbt
    call mp_fixlocr (ra)
    call mp_fixlocr (rb)
    mpnwbt = max (ra%mpr(1), rb%mpr(1))
    mpnwbt = min (mpnwbt, mpwdsbt)
    call mp_initvr (mp_min, mpnwbt)
    call mpfr_min (mp_min%mpr(1), ra%mpr(1), rb%mpr(1), %val(mprnd))
    if (present (rc)) then
      call mp_fixlocr (rc)
      call mpfr_min (mp_min%mpr(1), mp_min%mpr(1), rc%mpr(1), %val(mprnd))
    endif
  end function

  function mp_nrt (ra, ib)
    implicit none
    type (mp_real):: mp_nrt
    type (mp_real), intent (in):: ra
    integer, intent (in):: ib
    integer mpnwbt
    call mp_fixlocr (ra)
    mpnwbt = min (ra%mpr(1), mpwdsbt)
    call mp_initvr (mp_nrt, mpnwbt)
    call mpfrroot (mp_nrt%mpr(1), ra%mpr(1), ib, mprnd) 
    return
  end function
  
  function mp_pi (iprec)
    implicit none
    type (mp_real):: mp_pi
    integer mpnwbt

!>  In variant #1, uncomment these six lines:
    integer, optional, intent (in):: iprec
    if (present (iprec)) then
      mpnwbt = mpnbt * mp_setwp (iprec)
    else
      mpnwbt = mpnbt * mpwds
    endif
!>  Otherwise in variant #2, uncomment these two lines:
!    integer, intent (in):: iprec
!    mpnwbt = mpnbt * mp_setwp (iprec)

    call mp_initvr (mp_pi, mpnwbt)
    call mpfr_const_pi (mp_pi%mpr(1), %val(mprnd))
  end function

  function mp_polylog (ia, rb)
    implicit none
    type (mp_real):: mp_polylog
    integer ia
    type (mp_real), intent (in):: rb
    integer mpnwbt
    if (ia /= 2) then
      write (6, 1) ia
1     format ('***mp_polylog: first argument must be 2 for time being:',i6)
      call mp_abrt (89)
    endif
    call mp_fixlocr (rb)
    mpnwbt = min (rb%mpr(1), mpwdsbt)
    call mp_initvr (mp_polylog, mpnwbt)
    call mpfr_li2 (mp_polylog%mpr(1), rb%mpr(1), %val(mprnd))
    return
  end function

  function mp_prodd (ra, db)
    implicit none
    type (mp_real):: mp_prodd
    type (mp_real), intent (in):: ra
    double precision, intent (in):: db
    integer mpnwbt
    call mp_fixlocr (ra)
    mpnwbt = min (ra%mpr(1), mpwdsbt)
    call mp_initvr (mp_prodd, mpnwbt)
    call mpfrmuld (mp_prodd%mpr(1), ra%mpr(1), db, mprnd)
  end function
  
!>  Real*16 support is not available yet with MPFR.

  function mp_qtor (qa, iprec)
    implicit none
    type (mp_real):: mp_qtor
    double precision, intent (in):: qa
    integer mpnwbt

!>  In variant #1, uncomment these six lines:
    integer, optional, intent (in):: iprec
    if (present (iprec)) then
      mpnwbt = mpnbt * mp_setwp (iprec)
    else
      mpnwbt = mpnbt * mpwds
    endif
!>  Otherwise in variant #2, uncomment these two lines:
!    integer, intent (in):: iprec
!    mpnwbt = mpnbt * mp_setwp (iprec)

!   This is not yet implemented.
    mp_qtor%mpr(0) = mpwds6
    call mp_abrt (1)
    return
  end function

  function mp_quotd (ra, db)
    implicit none
    type (mp_real):: mp_quotd
    type (mp_real), intent (in):: ra
    double precision, intent (in):: db
    integer mpnwbt
    call mp_fixlocr (ra)
    mpnwbt = min (ra%mpr(1), mpwdsbt)
    call mp_initvr (mp_quotd, mpnwbt)
    call mpfrdivd (mp_quotd%mpr(1), ra%mpr(1), db, mprnd)
  end function
    
!   Five variations of read are necessary due to Fortran rules about optional arguments.

  subroutine mp_readr1 (iu, r1, iprec)
    implicit none
    integer, intent (in):: iu
    type (mp_real), intent (out):: r1
    integer mpnw

!>  In variant #1, uncomment these six lines:
    integer, optional, intent (in):: iprec
    if (present (iprec)) then
      mpnw = mp_setwp (iprec)
    else
      mpnw = mpwds
    endif
!>  Otherwise in variant #2, uncomment these two lines:
!    integer, intent (in):: iprec
!    mpnw = mp_setwp (iprec)

    call mp_inpr (iu, r1, mpnw)
    return
  end subroutine

  subroutine mp_readr2 (iu, r1, r2, iprec)
    implicit none
    integer, intent (in):: iu
    type (mp_real), intent (out):: r1, r2
    integer mpnw

!>  In variant #1, uncomment these six lines:
    integer, optional, intent (in):: iprec
    if (present (iprec)) then
      mpnw = mp_setwp (iprec)
    else
      mpnw = mpwds
    endif
!>  Otherwise in variant #2, uncomment these two lines:
!    integer, intent (in):: iprec
!    mpnw = mp_setwp (iprec)

    call mp_inpr (iu, r1, mpnw)
    call mp_inpr (iu, r2, mpnw)
    return
  end subroutine

  subroutine mp_readr3 (iu, r1, r2, r3, iprec)
    implicit none
    integer, intent (in):: iu
    type (mp_real), intent (out):: r1, r2, r3
    integer mpnw

!>  In variant #1, uncomment these six lines:
    integer, optional, intent (in):: iprec
    if (present (iprec)) then
      mpnw = mp_setwp (iprec)
    else
      mpnw = mpwds
    endif
!>  Otherwise in variant #2, uncomment these two lines:
!    integer, intent (in):: iprec
!    mpnw = mp_setwp (iprec)

    call mp_inpr (iu, r1, mpnw)
    call mp_inpr (iu, r2, mpnw)
    call mp_inpr (iu, r3, mpnw)
    return
  end subroutine

  subroutine mp_readr4 (iu, r1, r2, r3, r4, iprec)
    implicit none
    integer, intent (in):: iu
    type (mp_real), intent (out):: r1, r2, r3, r4
    integer mpnw

!>  In variant #1, uncomment these six lines:
    integer, optional, intent (in):: iprec
    if (present (iprec)) then
      mpnw = mp_setwp (iprec)
    else
      mpnw = mpwds
    endif
!>  Otherwise in variant #2, uncomment these two lines:
!    integer, intent (in):: iprec
!    mpnw = mp_setwp (iprec)

    call mp_inpr (iu, r1, mpnw)
    call mp_inpr (iu, r2, mpnw)
    call mp_inpr (iu, r3, mpnw)
    call mp_inpr (iu, r4, mpnw)
    return
  end subroutine

  subroutine mp_readr5 (iu, r1, r2, r3, r4, r5, iprec)
    implicit none
    integer, intent (in):: iu
    type (mp_real), intent (out):: r1, r2, r3, r4, r5
    integer mpnw

!>  In variant #1, uncomment these six lines:
    integer, optional, intent (in):: iprec
    if (present (iprec)) then
      mpnw = mp_setwp (iprec)
    else
      mpnw = mpwds
    endif
!>  Otherwise in variant #2, uncomment these two lines:
!    integer, intent (in):: iprec
!    mpnw = mp_setwp (iprec)

    call mp_inpr (iu, r1, mpnw)
    call mp_inpr (iu, r2, mpnw)
    call mp_inpr (iu, r3, mpnw)
    call mp_inpr (iu, r4, mpnw)
    call mp_inpr (iu, r5, mpnw)
    return
  end subroutine

  subroutine mp_readz1 (iu, z1, iprec)
    implicit none
    integer, intent (in):: iu
    type (mp_complex), intent (out):: z1
    integer mpnw

!>  In variant #1, uncomment these six lines:
    integer, optional, intent (in):: iprec
    if (present (iprec)) then
      mpnw = mp_setwp (iprec)
    else
      mpnw = mpwds
    endif
!>  Otherwise in variant #2, uncomment these two lines:
!    integer, intent (in):: iprec
!    mpnw = mp_setwp (iprec)

    call mp_inpz (iu, z1, mpnw)
    return
  end subroutine

  subroutine mp_readz2 (iu, z1, z2, iprec)
    implicit none
    integer, intent (in):: iu
    type (mp_complex), intent (out):: z1, z2
    integer mpnw

!>  In variant #1, uncomment these six lines:
    integer, optional, intent (in):: iprec
    if (present (iprec)) then
      mpnw = mp_setwp (iprec)
    else
      mpnw = mpwds
    endif
!>  Otherwise in variant #2, uncomment these two lines:
!    integer, intent (in):: iprec
!    mpnw = mp_setwp (iprec)

    call mp_inpz (iu, z1, mpnw)
    call mp_inpz (iu, z2, mpnw)
    return
  end subroutine

  subroutine mp_readz3 (iu, z1, z2, z3, iprec)
    implicit none
    integer, intent (in):: iu
    type (mp_complex), intent (out):: z1, z2, z3
    integer mpnw

!>  In variant #1, uncomment these six lines:
    integer, optional, intent (in):: iprec
    if (present (iprec)) then
      mpnw = mp_setwp (iprec)
    else
      mpnw = mpwds
    endif
!>  Otherwise in variant #2, uncomment these two lines:
!    integer, intent (in):: iprec
!    mpnw = mp_setwp (iprec)

    call mp_inpz (iu, z1, mpnw)
    call mp_inpz (iu, z2, mpnw)
    call mp_inpz (iu, z3, mpnw)
    return
  end subroutine

  subroutine mp_readz4 (iu, z1, z2, z3, z4, iprec)
    implicit none
    integer, intent (in):: iu
    type (mp_complex), intent (out):: z1, z2, z3, z4
    integer mpnw

!>  In variant #1, uncomment these six lines:
    integer, optional, intent (in):: iprec
    if (present (iprec)) then
      mpnw = mp_setwp (iprec)
    else
      mpnw = mpwds
    endif
!>  Otherwise in variant #2, uncomment these two lines:
!    integer, intent (in):: iprec
!    mpnw = mp_setwp (iprec)

    call mp_inpz (iu, z1, mpnw)
    call mp_inpz (iu, z2, mpnw)
    call mp_inpz (iu, z3, mpnw)
    call mp_inpz (iu, z4, mpnw)
    return
  end subroutine

  subroutine mp_readz5 (iu, z1, z2, z3, z4, z5, iprec)
    implicit none
    integer, intent (in):: iu
    type (mp_complex), intent (out):: z1, z2, z3, z4, z5
    integer mpnw

!>  In variant #1, uncomment these six lines:
    integer, optional, intent (in):: iprec
    if (present (iprec)) then
      mpnw = mp_setwp (iprec)
    else
      mpnw = mpwds
    endif
!>  Otherwise in variant #2, uncomment these two lines:
!    integer, intent (in):: iprec
!    mpnw = mp_setwp (iprec)

    call mp_inpz (iu, z1, mpnw)
    call mp_inpz (iu, z2, mpnw)
    call mp_inpz (iu, z3, mpnw)
    call mp_inpz (iu, z4, mpnw)
    call mp_inpz (iu, z5, mpnw)
    return
  end subroutine

  function mp_rtod (ra)
    implicit none
    double precision:: mp_rtod
    type (mp_real), intent (in):: ra
    double precision mpfr_get_d
    external mpfr_get_d
    call mp_fixlocr (ra)
    mp_rtod = mpfr_get_d (ra%mpr(1), %val(mprnd))
    return
  end function
  
!>  Real*16 support is not available yet with MPFR.

  function mp_rtoq (ra)
    implicit none
    double precision:: mp_rtoq
    type (mp_real), intent (in):: ra
    double precision mpfr_get_d
    external mpfr_get_d
    call mp_fixlocr (ra)
!    mp_rtod = mpfr_get_d (ra%mpr(1), %val(mprnd))

!   This is not yet implemented.
    mp_rtoq = 0.d0
    call mp_abrt (1)
    return
  end function

  function mp_rtor (ra, iprec)
    implicit none
    type (mp_real):: mp_rtor
    type (mp_real), intent (in):: ra
    integer mpnwbt

!>  In variant #1, uncomment these six lines:
    integer, optional, intent (in):: iprec
    if (present (iprec)) then
      mpnwbt = mpnbt * mp_setwp (iprec)
    else
      mpnwbt = mpnbt * mpwds
    endif
!>  Otherwise in variant #2, uncomment these two lines:
!    integer, intent (in):: iprec
!    mpnwbt = mpnbt * mp_setwp (iprec)

    call mp_fixlocr (ra)
    call mp_initvr (mp_rtor, mpnwbt)
    call mpfr_set (mp_rtor%mpr(1), ra%mpr(1), %val(mprnd))
    return
  end function

  function mp_rtoz (ra, rb, iprec)
    implicit none
    type (mp_complex):: mp_rtoz
    type (mp_real), intent (in):: ra, rb
    integer l1, mpnwbt

!>  In variant #1, uncomment these six lines:
    integer, optional, intent (in):: iprec
    if (present (iprec)) then
      mpnwbt = mpnbt * mp_setwp (iprec)
    else
      mpnwbt = mpnbt * mpwds
    endif
!>  Otherwise in variant #2, uncomment these two lines:
!    integer, intent (in):: iprec
!    mpnwbt = mpnbt * mp_setwp (iprec)

    call mp_fixlocr (ra)
    call mp_fixlocr (rb)
    l1 = mpwds6
    call mp_initvz (mp_rtoz, mpnwbt)
    call mpfr_set (mp_rtoz%mpc(1), ra%mpr(1), %val(mprnd))
    call mpfr_set (mp_rtoz%mpc(l1+1), rb%mpr(1), %val(mprnd))
    return
  end function

  function mp_sign (ra, rb)
    implicit none
    type (mp_real):: mp_sign
    type (mp_real), intent (in):: ra, rb
    integer mpnwbt
    call mp_fixlocr (ra)
    call mp_fixlocr (rb)
    mpnwbt = max (ra%mpr(1), rb%mpr(1))
    mpnwbt = min (mpnwbt, mpwdsbt)
    call mp_initvr (mp_sign, mpnwbt)
    if (rb%mpr(2) == 1) then
      call mpfr_set (mp_sign%mpr(1), ra%mpr(1), %val(mprnd))
    else
      call mpfr_neg (mp_sign%mpr(1), ra%mpr(1), %val(mprnd))
    endif    
    return
  end function
  
  function mp_sin (ra)
    implicit none
    type (mp_real):: mp_sin
    type (mp_real), intent (in):: ra
    integer mpnwbt
    call mp_fixlocr (ra)
    mpnwbt = min (ra%mpr(1), mpwdsbt)
    call mp_initvr (mp_sin, mpnwbt)
    call mpfr_sin (mp_sin%mpr(1), ra%mpr(1), %val(mprnd))
    return
  end function
  
  function mp_sinh (ra)
    implicit none
    type (mp_real):: mp_sinh
    type (mp_real), intent (in):: ra
    integer mpnwbt
    call mp_fixlocr (ra)
    mpnwbt = min (ra%mpr(1), mpwdsbt)
    call mp_initvr (mp_sinh, mpnwbt)
    call mpfr_sinh (mp_sinh%mpr(1), ra%mpr(1), %val(mprnd))
    return
  end function
  
  function mp_sqrt (ra)
    implicit none
    type (mp_real):: mp_sqrt
    type (mp_real), intent (in):: ra
    integer mpnwbt
    call mp_fixlocr (ra)
    mpnwbt = min (ra%mpr(1), mpwdsbt)
    call mp_initvr (mp_sqrt, mpnwbt)
    call mpfr_sqrt (mp_sqrt%mpr(1), ra%mpr(1), %val(mprnd))
    return
  end function
  
  function mp_tan (ra)
    implicit none
    type (mp_real):: mp_tan
    type (mp_real), intent (in):: ra
    integer mpnwbt
    call mp_fixlocr (ra)
    mpnwbt = min (ra%mpr(1), mpwdsbt)
    call mp_initvr (mp_tan, mpnwbt)
    call mpfr_tan (mp_tan%mpr(1), ra%mpr(1), %val(mprnd))
    return
  end function

  function mp_tanh (ra)
    implicit none
    type (mp_real):: mp_tanh
    type (mp_real), intent (in):: ra
    integer mpnwbt
    call mp_fixlocr (ra)
    mpnwbt = min (ra%mpr(1), mpwdsbt)
    call mp_initvr (mp_tanh, mpnwbt)
    call mpfr_tanh (mp_tanh%mpr(1), ra%mpr(1), %val(mprnd))
    return
  end function

!   Return working precision level (in words) of input MP value.

  function mp_wprec (ra)
    implicit none
    integer mp_wprec
    type (mp_real), intent (in):: ra
    integer mpnwbt
    mpnwbt = min (ra%mpr(1), mpwdsbt)
    mp_wprec = mpnwbt / mpnbt
    return
  end function

  function mp_wprecz (za)
    implicit none
    integer mp_wprecz
    type (mp_complex), intent (in):: za
    integer l1, mpnwbt
    l1 = za%mpc(0)
    mpnwbt = max (int (za%mpc(1)), int (za%mpc(l1+1)))
    mp_wprecz = min (mpnwbt, mpwdsbt) / mpnbt
    return
  end function

!   Output routines.

  subroutine mp_writer (iu, ln, ld, r1, r2, r3, r4, r5)
    implicit none
    integer, intent (in):: iu, ln, ld
    type (mp_real), intent (in):: r1, r2, r3, r4, r5
    optional:: r2, r3, r4, r5
     
    call mp_outr (iu, ln, ld, r1)
    if (present (r2)) then
      call mp_outr (iu, ln, ld, r2)
    endif
    if (present (r3)) then
      call mp_outr (iu, ln, ld, r3)
    endif
    if (present (r4)) then
      call mp_outr (iu, ln, ld, r4)
    endif
    if (present (r5)) then
      call mp_outr (iu, ln, ld, r5)
    endif
    
    return
  end subroutine
  
  subroutine mp_writez (iu, ln, ld, z1, z2, z3, z4, z5)
    implicit none
    integer, intent (in):: iu, ln, ld
    type (mp_complex), intent (in):: z1, z2, z3, z4, z5
    optional:: z2, z3, z4, z5

    call mp_outz (iu, ln, ld, z1)
    if (present (z2)) then
      call mp_outz (iu, ln, ld, z2)
    endif
    if (present (z3)) then
      call mp_outz (iu, ln, ld, z3)
    endif
    if (present (z4)) then
      call mp_outz (iu, ln, ld, z4)
    endif
    if (present (z5)) then
      call mp_outz (iu, ln, ld, z5)
    endif

    return
  end subroutine
  
  function mp_zeta (ra)
    implicit none
    type (mp_real):: mp_zeta
    type (mp_real), intent (in):: ra
    integer mpnwbt
    call mp_fixlocr (ra)
    mpnwbt = min (ra%mpr(1), mpwdsbt)
    call mp_initvr (mp_zeta, mpnwbt)
    call mpfr_zeta (mp_zeta%mpr(1), ra%mpr(1), %val(mprnd))
    return
  end function

  function mp_zetaem (nb, rb, rc)
    implicit none
    integer, intent (in):: nb
    type (mp_real):: mp_zetaem
    type (mp_real), intent (in):: rb(nb), rc
    integer n1, mpnw
    mpnw = max (int (rb(1)%mpr(1)), int (rc%mpr(1)))
    mpnw = min (mpnw, mpwds)
    mp_zetaem%mpr(0) = mpwds6
    n1 = mpwds

!   This is not implemented.
    call mp_abrt (1)
    return
  end function
 
  function mp_ztodc (za)
    implicit none
    complex (kind(0.d0)):: mp_ztodc
    type (mp_complex), intent (in):: za
    integer l1
    double precision d1, d2, mpfr_get_d
    external mpfr_get_d
    call mp_fixlocz (za)
    l1 = za%mpc(0)
    d1 = mpfr_get_d (za%mpc(1), %val(mprnd))
    d2 = mpfr_get_d (za%mpc(l1+1), %val(mprnd))
    mp_ztodc = dcmplx (d1, d2)
    return
  end function

  function mp_ztor (za, iprec)
    implicit none
    type (mp_real):: mp_ztor
    type (mp_complex), intent (in):: za
    integer l1, mpnwbt

!>  In variant #1, uncomment these six lines:
    integer, optional, intent (in):: iprec
    if (present (iprec)) then
      mpnwbt = mpnbt * mp_setwp (iprec)
    else
      mpnwbt = mpnbt * mpwds
    endif
!>  Otherwise in variant #2, uncomment these two lines:
!    integer, intent (in):: iprec
!    mpnwbt = mpnbt * mp_setwp (iprec)

    l1 = za%mpc(0)
    call mp_fixlocz (za)
    call mp_initvr (mp_ztor, mpnwbt)
    call mpfr_set (mp_ztor%mpr(1), za%mpc(1), %val(mprnd))
    return
  end function

  function mp_ztor2 (za)
    implicit none
    type (mp_real):: mp_ztor2
    type (mp_complex), intent (in):: za
    integer l1, mpnwbt
    mpnwbt = mpnbt * mpwds
    l1 = za%mpc(0)
    call mp_fixlocz (za)
    call mp_initvr (mp_ztor2, mpnwbt)
    call mpfr_set (mp_ztor2%mpr(1), za%mpc(1), %val(mprnd))
    return
  end function

  function mp_ztoz (za, iprec)
    implicit none
    type (mp_complex):: mp_ztoz
    type (mp_complex), intent (in):: za
    integer l1, l2, mpnwbt

!>  In variant #1, uncomment these six lines:
    integer, optional, intent (in):: iprec
    if (present (iprec)) then
      mpnwbt = mpnbt * mp_setwp (iprec)
    else
      mpnwbt = mpnbt * mpwds
    endif
!>  Otherwise in variant #2, uncomment these two lines:
!    integer, intent (in):: iprec
!    mpnwbt = mpnbt * mp_setwp (iprec)

    l1 = za%mpc(0)
    call mp_fixlocz (za)
    l2 = mpwds6
    call mp_initvz (mp_ztoz, mpnwbt)
    call mpfr_set (mp_ztoz%mpc(1), za%mpc(1), %val(mprnd))
    call mpfr_set (mp_ztoz%mpc(l2+1), za%mpc(l1+1), %val(mprnd))
    return
  end function

  subroutine mp_inpr (iu, a, mpnw)

!   This routine reads the MPR number A from logical unit IU.  The digits of A 
!   may span more than one line, provided that a "\" appears at the end of 
!   a line to be continued (any characters after the "\" on the same line
!   are ignored).  Individual input lines may not exceed 2048 characters in
!   length, although this limit can be changed in the system parameters
!   (parameter mpnstr) in module MPFUNA.  Embedded blanks are allowed anywhere.
!   An exponent with "e" or "d" may optionally follow the numeric value.

!   A scratch array below (CHR1) holds character data for input to MPFR_SET_STR.
!   It is dimensioned MPNWBT * (MPNDPW + 1) + 1000 (see below).
!   If more nonblank input characters than this are input, they are ignored.

  implicit none
  integer i, i1, iu, lnc1, lncx, ln1, mpnw
  character(mpnstr) line1
  character(18) validc
  parameter (validc = ' 0123456789+-.dDeE')
  character(1) chr1(mpnw*(mpndpw+1)+1001)
  type (mp_real) a
  integer mpnwbt

  mpnwbt = mpnw * mpnbt
  call mp_initvr (a, mpnwbt)
  lnc1 = 0
  lncx = mpnw * (mpndpw + 1) + 1000

100 continue

  read (iu, '(a)', end = 200) line1

!   Find the last nonblank character.

  do i = mpnstr, 1, -1
    if (line1(i:i) /= ' ') goto 110
  enddo

!   Input line is blank -- ignore.

  goto 100

110 continue

  ln1 = i

!   Scan input line, looking for valid characters.

  do i = 1, ln1
    if (line1(i:i) == '\') goto 100
    i1 = index (validc, line1(i:i))
    if (i1 == 0 .and. line1(i:i) /= ' ') then
      write (6, 2) line1(i:i)
2     format ('*** mp_inpr: Invalid input character = ',a)
      call mp_abrt (87)
    elseif (line1(i:i) .ne. ' ') then
      if (lnc1 < lncx) then
        lnc1 = lnc1 + 1
        if (line1(i:i) == 'D' .or. line1(i:i) == 'd') then
          chr1(lnc1) = 'e'
        else
          chr1(lnc1) = line1(i:i)
        endif
      endif
    endif
  enddo
  
  chr1(lnc1+1) = char(0)
  call mpfrsetstr (a%mpr(1), chr1, mprnd)

  goto 300

200  continue

  write (mpldb, 4)
4 format ('*** mp_inpr: End-of-file encountered.')
  call mp_abrt (72)

300 return
  end subroutine
  
  subroutine mp_inpz (iu, a, mpnw)
    implicit none
    integer, intent (in):: iu, mpnw
    type (mp_complex), intent (out):: a
    type (mp_real) r1, r2
    call mp_inpr (iu, r1, mpnw)
    call mp_inpr (iu, r2, mpnw)
    a = mp_rtoz (r1, r2, mpnw)
    return
  end subroutine

  subroutine mp_outr (iu, ln, nd, a)

!   This routine writes MPR number A to logical unit IU in E(LN,ND) format.
!   This is output on mpoutln characters per line.  The value of mpoutln is set 
!   in the system parameters at the start of module MPFUNA.

  implicit none
  integer i, iu, ln, ln1, nd
  character(1) chr1(ln)
  character(32) cform1, cform2
  type (mp_real) a

  call mp_fixlocr (a)
  call mp_eform (a, ln, nd, chr1)

  write (cform1, 1) mpoutln
  1 format ('(',i8,'a1)')
  write (cform2, 2) mpoutln
  2 format ('(',i8,'a1,"\")')

  if (ln <= mpoutln) then
    write (iu, fmt = cform1) (chr1(i), i = 1, ln)
  elseif (mod (ln, mpoutln) == 0) then
    ln1 = mpoutln * (ln / mpoutln) - mpoutln
    write (iu, fmt = cform2) (chr1(i), i = 1, ln1)
    write (iu, fmt = cform1) (chr1(i), i = ln1 + 1, ln)
  else
    ln1 = mpoutln * (ln / mpoutln)
    write (iu, fmt = cform2) (chr1(i), i = 1, ln1)
    write (iu, fmt = cform1) (chr1(i), i = ln1 + 1, ln)
  endif

  return
  end subroutine

  subroutine mp_outz (iu, ln, nd, a)
    implicit none
    integer, intent (in):: iu, ln, nd
    type (mp_complex), intent (in):: a    
    call mp_outr (iu, ln, nd, mp_ztor2 (a))
    call mp_outr (iu, ln, nd, mp_aimag (a))
    return
  end subroutine

end module mpfung

