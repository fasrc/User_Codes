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

!=====================================================================
!  Shell sort routine from Numerical Recipes
!  The best of the O(n^2) sorting algorithms
!  INPUT:
!  n    --> number of lines
!  dim1 --> number of rows
!  ind  --> index of the row by which to make the sort
!  arr  --> array to be sorted, arr(dim1,n) 
!
!  OUTPUT:
!  arr  --> sorted array
!  
! Note: Data type -- Integer
!=====================================================================
subroutine shellsort(n,ind,dim1,arr)
  implicit none
  integer(4) :: n
  integer(4) :: ind
  integer(4) :: dim1
  integer(4) :: arr(dim1,n)
  integer(4) :: i
  integer(4) :: j
  integer(4) :: inc
  integer(4) ::  v
  integer(4) ::  varr(dim1)
  
  inc = 1
1 inc = 3 * inc + 1
  if ( inc .le. n ) go to 1
2 continue
  inc = inc / 3
  do i = inc + 1, n
     v = arr(ind,i)
     varr = arr(:,i)
     j = i
3    if ( arr(ind,j-inc) .gt. v ) then
        arr(:,j) = arr(:,j-inc)
        j = j - inc
        if ( j .le. inc ) go to 4
        go to 3
     end if
4    arr(:,j) = varr
  end do
  if ( inc .gt. 1 ) go to 2
  return
end subroutine shellsort

!=====================================================================
! Sorts out the array arr(n)
! piksrt --> adapted from Numerical recepies
!=====================================================================
SUBROUTINE piksrt(n,arr)
  INTEGER(4) :: n
  INTEGER(4) :: i,j
  REAL(8)    :: arr(n)
  REAL(8)    :: a
  do j = 2, n
     a = arr(j)
     do i = j-1, 1, -1 
        if ( arr(i) .le. a ) goto 10
        arr(i+1)=arr(i)
     end do
     i = 0
10   arr(i+1) = a 
  end do
  return
END SUBROUTINE piksrt
