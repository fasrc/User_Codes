!=====================================================================
!
! Standard Lanczos algorithm with re-orthogonalization
!
! Serial implementation
!
! LANCZOS ALGORITHM: https://en.wikipedia.org/wiki/Lanczos_algorithm
!
! 
! Author:  Plamen G. Krastev, Ph.D.
!
! Address: Harvard University
!          Research Computing
!          Faculty of Arts and Sciences
!          38 Oxford Street, Room 117
!          Cambridge, MA 02138, USA
!
! Phone:   +1 (617) 496-0968
! Email:   plamenkrastev@fas.harvard.edu
!
!  +++ COMPILING THE PROGRAM +++
!
!      GFORTRAN: gfortran -o lanczos.x lanczos.f90 -O2 
!      INTEL:    ifort -o lanczos.x lanczos.f90 -O2
!
!  +++ RUNNING THE PROGRAM +++
!
!      ./lanczos.x
!
!      Upon execustion, program generates the files "lanczosvec.lvec",
!      which contains all Lanczos vectors, and "eigenvalues.out",
!      with the exact eigen values. 
!
!=====================================================================
program lanczos
  implicit none
  integer(4)           :: n
  integer(4)           :: i 
  integer(4)           :: j 
  integer(4)           :: iseed 
  integer(4)           :: iter
  integer(4)           :: niter
  integer(4)           :: nkeep
  integer(4)           :: ierr
  real(8)              :: dnorm
  real(8), allocatable :: h(:,:)
  real(8), allocatable :: dh(:)
  real(8), allocatable :: eh(:)
  real(8), allocatable :: zh(:,:)
! Lanczos vectors.....................................................
  real(8), allocatable :: v(:)
  real(8), allocatable :: w(:)
  real(8), allocatable :: vold(:)
  real(8), allocatable :: lvec(:,:)
! Lanczos coefficients................................................
  real(8), allocatable :: alpha(:)
  real(8), allocatable :: beta(:)
! arrays for diagonalization of tridiagonal matrix
  real(8), allocatable :: d(:)
  real(8), allocatable :: e(:)
  real(8), allocatable :: z(:,:)
! Random number generator to compute a random test-matrix
  real(8), external    :: ran3
! file control........................................................
  integer(4)           :: eig_file = 8
  integer(4)           :: lvec_file = 11
  integer(4)           :: coef_file = 22

  real(8)              :: da
  real(8)              :: db

  logical              :: writetodisk
  logical              :: reorthog
  
  writetodisk = .true.  ! All Lanczos vectors written to disk
  reorthog    = .true.  ! Re-orthogonalization of Lanczos vectors enabled

  if ( .not. writetodisk ) then
     if ( .not. allocated(lvec) ) allocate( lvec(n,niter) )
  end if
  if ( writetodisk ) call openlanczosfile(lvec_file)

! Initialize random number generator..................................
  iseed = -99

! Get required input..................................................
  write(6,*)'Enter matrix dimension:'
  read(5,*)n
  write(6,*)'Enter number of eigen-values to compute:'
  read(5,*)nkeep
  write(6,*)'Enter MAX number of iterations ( typically 100 - 300, but <= matrix dim.)'
  read(5,*)niter

! Allocate memory.....................................................
  if ( .not. allocated (h) )     allocate ( h(n,n) )
  if ( .not. allocated (dh) )    allocate (dh(n) )
  if ( .not. allocated (eh) )    allocate ( eh(n) )
  if ( .not. allocated (zh) )    allocate ( zh(n,n) )  
  if ( .not. allocated (v) )     allocate ( v(n) )
  if ( .not. allocated (w) )     allocate ( w(n) )
  if ( .not. allocated (vold) )  allocate ( vold(n) )
  if ( .not. allocated (alpha) ) allocate ( alpha(niter) )
  if ( .not. allocated (beta) )  allocate ( beta(niter) )
  if ( .not. allocated (d) )     allocate ( d(niter) )
  if ( .not. allocated (e) )     allocate ( e(niter) )
  if ( .not. allocated (z) )     allocate ( z(n,niter) )

! Create random test matrix h.........................................
  do i = 1, n
     do j = 1, i
        h(i,j) = ran3(iseed)
        h(j,i) = h(i,j)
     end do
  end do

! Diagonalize matrix h (exact diagonalization)........................
  call tred2(n,n,h,dh,eh,zh) 
  call tqli(n,n,dh,eh,zh,ierr)
  call eigsrt(dh,zh,n,n)
  if ( ierr .ne. 0 ) write(6,*) 'h - diag ierr=',ierr
  open(unit=8,file='eigenvalues.out')
  do i = 1, n
     write(eig_file,*) i,dh(i)
  end do
  close(unit=eig_file)

! Lanczos diagonalization.............................................
! Create a starting vector............................................
  dnorm = 1.d0 / dsqrt(dfloat(n))
  do i = 1, n
     v(i) = dnorm * ran3(iseed)
  end do
! Normalize it........................................................
  call dnormvec(n,v,da)
  do j = 1, niter
     alpha(j) = 0.0d0
     beta(j) = 0.0d0
  end do
  iter = 0
! Start Lanczos iterations............................................
  do while ( iter < niter )
     iter = iter + 1
     call writelanczosvec(writetodisk,lvec_file,n,niter,iter,v,lvec)
     call applyh(n,h,v,w)
     if ( niter > 1 )then
        call dvecproj(n,w,vold,db)
     end if
     call dvecproj(n,w,v,da)
     alpha(iter) = da
     if ( iter < niter ) then
        call dnormvec(n,w,db)
        beta(iter+1) = db
! Reorthogonalize.....................................................
        if ( reorthog ) then
           do j = 1, iter - 1
              call readlanczosvec(writetodisk,lvec_file,n,niter,j,vold,lvec)
              call dvecproj(n,w,vold,da)
           end do
           call readlanczosvec(writetodisk,lvec_file,n,niter,iter,vold,lvec)
           call dnormvec(n,w,da)
        end if
!.....................................................................
     end if
     
     do j = 1, n
        vold(j) = v(j)
        v(j) = w(j)
        w(j) = 0.0d0
     end do
! Prepare to diagonalize..............................................
     do j = 1, niter
        d(j) = 0.0d0
        e(j) = 0.0d0
        do i = 1, j
           z(i,j) = 0.0d0
           z(j,i) = z(i,j)
        end do
        z(j,j) = 1.0d0
     end do
     do j = 1, iter
        d(j) = alpha(j)
        e(j) = beta(j)
     end do
! Diagonalize tridiagonal matrix......................................
     call tqli(n,iter,d,e,z,ierr)
     call eigsrt(d,z,niter,niter)
     if ( ierr .ne. 0  ) write(6,*) 'diag ierr=',ierr
     write(6,*) nkeep,' lowest eigenvalues - Lanczos, exact'
     write(6,*) 'iteration:',iter
     do i = 1, nkeep
        write(6,*) i,d(i),dh(i)
     end do
  end do
! End Lanczos iterations.............................................. 
  if ( writetodisk ) call closelanczosfile(lvec_file)

! Free memory.........................................................
  if ( allocated (h) )     deallocate ( h )
  if ( allocated (dh) )    deallocate ( dh )
  if ( allocated (eh) )    deallocate ( eh )
  if ( allocated (zh) )    deallocate ( zh )
  if ( allocated (v) )     deallocate ( v )
  if ( allocated (w) )     deallocate ( w )
  if ( allocated (vold) )  deallocate ( vold )
  if ( allocated (alpha) ) deallocate ( alpha )
  if ( allocated (beta) )  deallocate ( beta )
  if ( allocated (d) )     deallocate ( d )
  if ( allocated (e) )     deallocate ( e )
  if ( allocated (z) )     deallocate ( z )

  stop 'End of program.'

end program lanczos

!=====================================================================
! Matrix-vector multiplication
!=====================================================================
subroutine applyh(n,h,vecin,vecout)
  implicit none
  integer(4) :: i,j
  integer(4) :: n
  real(8) :: h(n,n)
  real(8) :: vecin(n),vecout(n)
  do i = 1, n
     vecout(i) = 0.0d0
     do j = 1, n
        vecout(i) = vecout(i) + h(i,j) * vecin(j)
     end do
  end do
  return                                                                    
end subroutine applyh

!=====================================================================
! Double-precision normalization
!
! n     --> dimension of vector
! dvec  --> double-precision vector
!
! dnorm --> double-precision norm of dvec
!=====================================================================
subroutine dnormvec(n,dvec,dnorm)
  implicit none
  integer(8) :: n
  real(8)    :: dvec(n)
  real(8)    :: dnorm
  real(8)    :: d  
  integer(4) :: i
  dnorm = 0.d0
  do i = 1, n
     d = dvec(i)
     dnorm = dnorm + d * d
  end do
  dnorm = dsqrt(dnorm)
  d = 1.d0 / dnorm
  do i = 1,n
     dvec(i) = dvec(i) * d
  end do
  return
end subroutine dnormvec

!=====================================================================
! Double-precision projection
!
! n: dimension of vector
! dvec1: double-precision vector
! dvec2: double-preciscion vector
!
! dsclrprod  = dvec1*dvec2
!
! dvec1 -> dvec1 - dvec2*dsclrprod
!=====================================================================
subroutine dvecproj(n,dvec1,dvec2,dsclrprod)
  implicit none
  integer(4) :: n
  real(8)    :: dvec1(n),dvec2(n)
  real(8)    :: dsclrprod
  real(8)    :: d1,d2
  integer(4) :: i

  dsclrprod = 0.d0

  do i = 1, n
     d1 = dvec1(i)
     d2 = dvec2(i)
     dsclrprod = dsclrprod + ( d1 * d2 )
  end do

  do i = 1, n
     d1 = dvec1(i)
     d2 = dvec2(i)
     dvec1(i) = d1 - ( d2 * dsclrprod )
  end do
  return
end subroutine dvecproj

!=====================================================================
! Open Lanczos file
!=====================================================================
subroutine openlanczosfile(lvec_file)
  implicit none
  character (len=25) :: filename
  integer(4)         :: ilast
  integer(4)         :: lvec_file
  filename = 'lanczosvec.lvec'
  ilast = index(filename,' ')-1
  open(unit = lvec_file,file=filename(1:ilast),status='unknown',form ='unformatted')
  return
end subroutine openlanczosfile

!=====================================================================
! Close Lanczos file
!=====================================================================
subroutine closelanczosfile(lvec_file)
  implicit none
  integer(4) :: lvec_file  
  rewind(lvec_file)
  write(lvec_file)0
  close(lvec_file)
  return
end subroutine closelanczosfile

!=====================================================================
! Write Lanczos vector
!=====================================================================
subroutine writelanczosvec(writetodisk,lvec_file,n,niter,i,v,lvec)
  implicit none
  integer(4) :: n,niter
  integer(4) :: i, j
  integer(4) :: lvec_file
  real(8)    :: v(n),lvec(n,niter)
  logical    :: writetodisk
  if ( writetodisk ) then
     write(lvec_file) ( v(j), j = 1, n )
  else
     lvec(:,i) = v(:)
  end if
  return
end subroutine writelanczosvec

!=====================================================================
! Read Lanczos vector
!=====================================================================
subroutine readlanczosvec(writetodisk,lvec_file,n,niter,i,v,lvec)
  implicit none
  integer(4) :: n,niter
  integer(4) :: i, j
  integer(4) :: lvec_file
  real(8)    :: v(n),lvec(n,niter)
  logical    :: writetodisk
  if ( writetodisk ) then
     if ( i == 1 ) rewind(lvec_file)
     read(lvec_file) ( v(j), j = 1, n )
  else
     v(:) = lvec(:,i)
  end if
  return
end subroutine readlanczosvec

!===========================================================================
!     THIS SUBROUTINE IS A TRANSLATION OF THE ALGOL PROCEDURE TRED2, 
!     NUM. MATH. 11, 181-195(1968) BY MARTIN, REINSCH, AND WILKINSON. 
!     HANDBOOK FOR AUTO. COMP., VOL.II-LINEAR ALGEBRA, 212-226(1971). 
! 
!     THIS SUBROUTINE REDUCES A REAL SYMMETRIC MATRIX TO A 
!     SYMMETRIC TRIDIAGONAL MATRIX USING AND ACCUMULATING 
!     ORTHOGONAL SIMILARITY TRANSFORMATIONS. 
! 
!     ON INPUT 
! 
!        NM MUST BE SET TO THE ROW DIMENSION OF TWO-DIMENSIONAL 
!          ARRAY PARAMETERS AS DECLARED IN THE CALLING PROGRAM 
!          DIMENSION STATEMENT. 
! 
!        N IS THE ORDER OF THE MATRIX. 
! 
!        A CONTAINS THE REAL SYMMETRIC INPUT MATRIX.  ONLY THE 
!          LOWER TRIANGLE OF THE MATRIX NEED BE SUPPLIED. 
! 
!     ON OUTPUT 
! 
!        D CONTAINS THE DIAGONAL ELEMENTS OF THE TRIDIAGONAL MATRIX. 
! 
!        E CONTAINS THE SUBDIAGONAL ELEMENTS OF THE TRIDIAGONAL 
!          MATRIX IN ITS LAST N-1 POSITIONS.  E(1) IS SET TO ZERO. 
! 
!        Z CONTAINS THE ORTHOGONAL TRANSFORMATION MATRIX 
!          PRODUCED IN THE REDUCTION. 
! 
!        A AND Z MAY COINCIDE.  IF DISTINCT, A IS UNALTERED. 
! 
!     QUESTIONS AND COMMENTS SHOULD BE DIRECTED TO BURTON S. GARBOW, 
!     MATHEMATICS AND COMPUTER SCIENCE DIV, ARGONNE NATIONAL LABORATORY 
! 
!     THIS VERSION DATED AUGUST 1983. 
! 
!===========================================================================
SUBROUTINE TRED2(NM,N,A,D,E,Z)
  implicit none
  INTEGER(4) :: I,J,K,L,N,II,NM,JP1 
  REAL(8)    :: A(NM,N),D(N),E(N),Z(NM,N) 
  REAL(8)    :: F,G,H,HH,SCALE  
!........................................................................... 
      DO 100 I = 1, N 
 
         DO 80 J = I, N 
   80    Z(J,I) = A(J,I) 
 
         D(I) = A(N,I) 
  100 CONTINUE 
 
      IF (N .EQ. 1) GO TO 510 
!     .......... FOR I=N STEP -1 UNTIL 2 DO -- ............................. 
      DO 300 II = 2, N 
         I = N + 2 - II 
         L = I - 1 
         H = 0.0d0 
         SCALE = 0.0d0 
         IF (L .LT. 2) GO TO 130 
!     .......... SCALE ROW (ALGOL TOL THEN NOT NEEDED) .......... ..........
         DO 120 K = 1, L 
  120    SCALE = SCALE + DABS(D(K)) 
 
         IF (SCALE .NE. 0.0d0) GO TO 140 
  130    E(I) = D(L) 
 
         DO 135 J = 1, L 
            D(J) = Z(L,J) 
            Z(I,J) = 0.0d0 
            Z(J,I) = 0.0d0 
  135    CONTINUE 
 
         GO TO 290 
 
  140    DO 150 K = 1, L 
            D(K) = D(K) / SCALE 
            H = H + D(K) * D(K) 
  150    CONTINUE 
 
         F = D(L) 
         G = -DSIGN(DSQRT(H),F) 
         E(I) = SCALE * G 
         H = H - F * G 
         D(L) = F - G 
!     .......... FORM A*U ..................................................
         DO 170 J = 1, L 
  170    E(J) = 0.0d0 
 
         DO 240 J = 1, L 
            F = D(J) 
            Z(J,I) = F 
            G = E(J) + Z(J,J) * F 
            JP1 = J + 1 
            IF (L .LT. JP1) GO TO 220 
 
            DO 200 K = JP1, L 
               G = G + Z(K,J) * D(K) 
               E(K) = E(K) + Z(K,J) * F 
  200       CONTINUE 
 
  220       E(J) = G 
  240    CONTINUE 
!     .......... FORM P ....................................................
         F = 0.0E0 
 
         DO 245 J = 1, L 
            E(J) = E(J) / H 
            F = F + E(J) * D(J) 
  245    CONTINUE 
 
         HH = F / (H + H) 
!     .......... FORM Q ....................................................
         DO 250 J = 1, L 
  250    E(J) = E(J) - HH * D(J) 
!     .......... FORM REDUCED A ............................................
         DO 280 J = 1, L 
            F = D(J) 
            G = E(J) 
 
            DO 260 K = J, L 
  260       Z(K,J) = Z(K,J) - F * E(K) - G * D(K) 
 
            D(J) = Z(L,J) 
            Z(I,J) = 0.0d0 
  280    CONTINUE 
 
  290    D(I) = H 
  300 CONTINUE 
!     .......... ACCUMULATION OF TRANSFORMATION MATRICES .......... ........
      DO 500 I = 2, N 
         L = I - 1 
         Z(N,L) = Z(L,L) 
         Z(L,L) = 1.0d0 
         H = D(I) 
         IF (H .EQ. 0.0d0) GO TO 380 
 
         DO 330 K = 1, L 
  330    D(K) = Z(K,I) / H 
 
         DO 360 J = 1, L 
            G = 0.0d0 
 
            DO 340 K = 1, L 
  340       G = G + Z(K,I) * Z(K,J) 
 
            DO 360 K = 1, L 
               Z(K,J) = Z(K,J) - G * D(K) 
  360    CONTINUE 
 
  380    DO 400 K = 1, L 
  400    Z(K,I) = 0.0d0 
 
  500 CONTINUE 
 
  510 DO 520 I = 1, N 
         D(I) = Z(N,I) 
         Z(N,I) = 0.0d0 
  520 CONTINUE 
 
      Z(N,N) = 1.0d0 
      E(1) = 0.0d0 
      RETURN 
end subroutine tred2

!===========================================================================
!     THIS SUBROUTINE IS A TRANSLATION OF THE ALGOL PROCEDURE TQL2, 
!     NUM. MATH. 11, 293-306(1968) BY BOWDLER, MARTIN, REINSCH, AND 
!     WILKINSON. 
!     HANDBOOK FOR AUTO. COMP., VOL.II-LINEAR ALGEBRA, 227-240(1971). 
! 
!     THIS SUBROUTINE FINDS THE EIGENVALUES AND EIGENVECTORS 
!     OF A SYMMETRIC TRIDIAGONAL MATRIX BY THE QL METHOD. 
!     THE EIGENVECTORS OF A FULL SYMMETRIC MATRIX CAN ALSO 
!     BE FOUND IF  TRED2  HAS BEEN USED TO REDUCE THIS 
!     FULL MATRIX TO TRIDIAGONAL FORM. 
! 
!     ON INPUT 
! 
!        NM MUST BE SET TO THE ROW DIMENSION OF TWO-DIMENSIONAL 
!          ARRAY PARAMETERS AS DECLARED IN THE CALLING PROGRAM 
!          DIMENSION STATEMENT. 
! 
!        N IS THE ORDER OF THE MATRIX. 
! 
!        D CONTAINS THE DIAGONAL ELEMENTS OF THE INPUT MATRIX. 
! 
!        E CONTAINS THE SUBDIAGONAL ELEMENTS OF THE INPUT MATRIX 
!          IN ITS LAST N-1 POSITIONS.  E(1) IS ARBITRARY. 
! 
!        Z CONTAINS THE TRANSFORMATION MATRIX PRODUCED IN THE 
!          REDUCTION BY  TRED2, IF PERFORMED.  IF THE EIGENVECTORS 
!          OF THE TRIDIAGONAL MATRIX ARE DESIRED, Z MUST CONTAIN 
!          THE IDENTITY MATRIX. 
! 
!      ON OUTPUT 
! 
!        D CONTAINS THE EIGENVALUES IN ASCENDING ORDER.  IF AN 
!          ERROR EXIT IS MADE, THE EIGENVALUES ARE CORRECT BUT 
!          UNORDERED FOR INDICES 1,2,...,IERR-1. 
! 
!        E HAS BEEN DESTROYED. 
! 
!        Z CONTAINS ORTHONORMAL EIGENVECTORS OF THE SYMMETRIC 
!          TRIDIAGONAL (OR FULL) MATRIX.  IF AN ERROR EXIT IS MADE, 
!          Z CONTAINS THE EIGENVECTORS ASSOCIATED WITH THE STORED 
!          EIGENVALUES. 
! 
!        IERR IS SET TO 
!          ZERO       FOR NORMAL RETURN, 
!          J          IF THE J-TH EIGENVALUE HAS NOT BEEN 
!                     DETERMINED AFTER 30 ITERATIONS. 
! 
!     CALLS PYTHAG FOR  SQRT(A*A + B*B) . 
! 
!     QUESTIONS AND COMMENTS SHOULD BE DIRECTED TO BURTON S. GARBOW, 
!     MATHEMATICS AND COMPUTER SCIENCE DIV, ARGONNE NATIONAL LABORATORY 
! 
!     THIS VERSION DATED AUGUST 1983. 
! 
!===========================================================================
SUBROUTINE TQLI(NM,N,D,E,Z,IERR) 
  implicit none 
  INTEGER(4) :: I,J,K,L,M,N,II,L1,L2,NM,MML,IERR 
  REAL(8)    :: D(N),E(N),Z(NM,N) 
  REAL(8)    :: C,C2,C3,DL1,EL1,F,G,H,P,R,S,S2,TST1,TST2,PYTHAG 
!...........................................................................
 
      IERR = 0 
      IF (N .EQ. 1) GO TO 1001 
 
      DO 100 I = 2, N 
  100 E(I-1) = E(I) 
 
      F = 0.0d0 
      TST1 = 0.0d0 
      E(N) = 0.0d0 
 
      DO 240 L = 1, N 
         J = 0 
         H = DABS(D(L)) + DABS(E(L)) 
         IF (TST1 .LT. H) TST1 = H 
!     .......... LOOK FOR SMALL SUB-DIAGONAL ELEMENT ....................... 
         DO 110 M = L, N 
            TST2 = TST1 + DABS(E(M)) 
            IF (TST2 .EQ. TST1) GO TO 120 
!     .......... E(N) IS ALWAYS ZERO, SO THERE IS NO EXIT 
!                THROUGH THE BOTTOM OF THE LOOP ............................
  110    CONTINUE 
 
  120    IF (M .EQ. L) GO TO 220 
  130    IF (J .EQ. 30) GO TO 1000 
         J = J + 1 
!     .......... FORM SHIFT ................................................
         L1 = L + 1 
         L2 = L1 + 1 
         G = D(L) 
         P = (D(L1) - G) / (2.0d0 * E(L)) 
         R = PYTHAG(P,1.0d0) 
         D(L) = E(L) / (P + DSIGN(R,P)) 
         D(L1) = E(L) * (P + DSIGN(R,P)) 
         DL1 = D(L1) 
         H = G - D(L) 
         IF (L2 .GT. N) GO TO 145 
 
         DO 140 I = L2, N 
  140    D(I) = D(I) - H 
 
  145    F = F + H 
!     .......... QL TRANSFORMATION ......................................... 
         P = D(M) 
         C = 1.0d0 
         C2 = C 
         EL1 = E(L1) 
         S = 0.0d0 
         MML = M - L 
!     .......... FOR I=M-1 STEP -1 UNTIL L DO -- ........................... 
         DO 200 II = 1, MML 
            C3 = C2 
            C2 = C 
            S2 = S 
            I = M - II 
            G = C * E(I) 
            H = C * P 
            R = PYTHAG(P,E(I)) 
            E(I+1) = S * R 
            S = E(I) / R 
            C = P / R 
            P = C * D(I) - S * G 
            D(I+1) = H + S * (C * G + S * D(I)) 
!     .......... FORM VECTOR ............................................... 
            DO 180 K = 1, N 
               H = Z(K,I+1) 
               Z(K,I+1) = S * Z(K,I) + C * H 
               Z(K,I) = C * Z(K,I) - S * H 
  180       CONTINUE 
 
  200    CONTINUE 
 
         P = -S * S2 * C3 * EL1 * E(L) / DL1 
         E(L) = S * P 
         D(L) = C * P 
         TST2 = TST1 + DABS(E(L)) 
         IF (TST2 .GT. TST1) GO TO 130 
  220    D(L) = D(L) + F 
  240 CONTINUE 
!     .......... ORDER EIGENVALUES AND EIGENVECTORS ........................ 
      DO 300 II = 2, N 
         I = II - 1 
         K = I 
         P = D(I) 
 
         DO 260 J = II, N 
            IF (D(J) .GE. P) GO TO 260 
            K = J 
            P = D(J) 
  260    CONTINUE 
! 
         IF (K .EQ. I) GO TO 300 
         D(K) = D(I) 
         D(I) = P 
 
         DO 280 J = 1, N 
            P = Z(J,I) 
            Z(J,I) = Z(J,K) 
            Z(J,K) = P 
  280    CONTINUE 
 
  300 CONTINUE 
 
      GO TO 1001 
!     .......... SET ERROR -- NO CONVERGENCE TO AN 
!                EIGENVALUE AFTER 30 ITERATIONS ............................ 
 1000 IERR = L 
 1001 RETURN 
end subroutine tqli
 
!===========================================================================
!     FINDS SQRT(A**2+B**2) WITHOUT OVERFLOW OR DESTRUCTIVE UNDERFLOW 
!===========================================================================
REAL(8) FUNCTION PYTHAG(A,B)
  implicit none
  REAL(8) :: A,B 
  REAL(8) :: P,R,S,T,U 
  P = MAX(DABS(A),DABS(B)) 
  IF (P .EQ. 0.0d0) GO TO 20 
  R = (MIN(DABS(A),DABS(B))/P)**2 
10 CONTINUE 
  T = 4.0d0 + R 
  IF (T .EQ. 4.0d0) GO TO 20 
  S = R/T 
  U = 1.0d0 + 2.0d0*S 
  P = U*P 
  R = (S/U)**2 * R 
  GO TO 10 
20 PYTHAG = P 
  RETURN 
END FUNCTION PYTHAG

!=====================================================================
! Sorts out the eigen-values and eigen-vectors computed by TQLI
!=====================================================================
subroutine eigsrt(d,v,n,np)
  implicit none
  integer(4) ::  n,np
  real(8)    :: d(np),v(np,np)
  integer(4) :: i,j,k
  real(8)    :: p
  do i=1,n-1
     k=i
     p=d(i)
     do j=i+1,n
        if(d(j).ge.p)then
           k=j
           p=d(j)
        endif
     end do
     if(k.ne.i)then
        d(k)=d(i)
        d(i)=p
        do j=1,n
           p=v(j,i)
           v(j,i)=v(j,k)
           v(j,k)=p
        end do
     endif
  end do
  return
end subroutine eigsrt

!=====================================================================
!     The function
!        ran3
!     returns a uniform random number deviate between 0.0 and 1.0. Set
!     the idum to any negative value to initialize or reinitialize the
!     sequence. Any large MBIG, and any small (but still large) MSEED
!     can be substituted for the present values. 
!=====================================================================
REAL(8) FUNCTION ran3(idum)
  IMPLICIT NONE
  INTEGER :: idum
  INTEGER :: mbig,mseed,mz
  REAL(8) ::  fac
  PARAMETER (mbig=1000000000,mseed=161803398,mz=0,fac=1./mbig)
  INTEGER :: i,iff,ii,inext,inextp,k
  INTEGER :: mj,mk,ma(55)
  SAVE iff,inext,inextp,ma
  DATA iff /0/
  
  IF ( (idum < 0) .or. (iff == 0) ) THEN
     iff=1
     mj=mseed-IABS(idum)
     mj=MOD(mj,mbig)
     ma(55)=mj
     mk=1
     DO i=1,54
        ii=MOD(21*i,55)
        ma(ii)=mk
        mk=mj-mk
        IF(mk < mz)mk=mk+mbig
        mj=ma(ii)
     ENDDO
     DO k=1,4
        DO i=1,55
           ma(i)=ma(i)-ma(1+MOD(i+30,55))
           IF (ma(i) < mz)ma(i)=ma(i)+mbig
        ENDDO
     ENDDO
     inext=0
     inextp=31
     idum=1
  ENDIF
  inext=inext+1
  IF (inext == 56) inext=1
  inextp=inextp+1
  IF (inextp == 56) inextp=1
  mj=ma(inext)-ma(inextp)
  IF (mj < mz) mj=mj+mbig
  ma(inext)=mj
  ran3=mj*fac
  return
END FUNCTION ran3
