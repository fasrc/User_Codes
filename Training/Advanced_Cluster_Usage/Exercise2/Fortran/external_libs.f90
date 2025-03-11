SUBROUTINE TRED2(NM, N, A, D, E, Z)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: NM, N
  REAL(8), INTENT(INOUT) :: A(NM, N), Z(NM, N)
  REAL(8), INTENT(OUT) :: D(N), E(N)
  INTEGER :: I, J, K, L, II, JP1
  REAL(8) :: F, G, H, HH, SCALE

  ! Copy lower triangle of A to Z and initialize D
  DO I = 1, N
    DO J = I, N
      Z(J, I) = A(J, I)
    END DO
    D(I) = A(N, I)
  END DO

  ! Handle the special case of N = 1
  IF (N == 1) THEN
    E(1) = 0.0D0
    Z(1, 1) = 1.0D0
    RETURN
  END IF

  ! Tridiagonalization loop
  DO II = 2, N
    I = N + 2 - II
    L = I - 1
    H = 0.0D0
    SCALE = 0.0D0

    ! Compute scale factor
    IF (L >= 2) THEN
      DO K = 1, L
        SCALE = SCALE + ABS(D(K))
      END DO
    END IF

    IF (SCALE == 0.0D0) THEN
      E(I) = D(L)
      DO J = 1, L
        D(J) = Z(L, J)
        Z(I, J) = 0.0D0
        Z(J, I) = 0.0D0
      END DO
    ELSE
      ! Scale the vector and compute H
      DO K = 1, L
        D(K) = D(K) / SCALE
        H = H + D(K) * D(K)
      END DO
      F = D(L)
      G = -SIGN(SQRT(H), F)
      E(I) = SCALE * G
      H = H - F * G
      D(L) = F - G

      ! Form A*U
      DO J = 1, L
        E(J) = 0.0D0
      END DO
      DO J = 1, L
        F = D(J)
        Z(J, I) = F
        G = E(J) + Z(J, J) * F
        JP1 = J + 1
        IF (JP1 <= L) THEN
          DO K = JP1, L
            G = G + Z(K, J) * D(K)
            E(K) = E(K) + Z(K, J) * F
          END DO
        END IF
        E(J) = G
      END DO

      ! Form P and Q
      F = 0.0D0
      DO J = 1, L
        E(J) = E(J) / H
        F = F + E(J) * D(J)
      END DO
      HH = F / (H + H)
      DO J = 1, L
        E(J) = E(J) - HH * D(J)
      END DO

      ! Reduce A
      DO J = 1, L
        F = D(J)
        G = E(J)
        DO K = J, L
          Z(K, J) = Z(K, J) - F * E(K) - G * D(K)
        END DO
        D(J) = Z(L, J)
        Z(I, J) = 0.0D0
      END DO
    END IF
    D(I) = H
  END DO

  ! Accumulate transformation matrices
  DO I = 2, N
    L = I - 1
    Z(N, L) = Z(L, L)
    Z(L, L) = 1.0D0
    H = D(I)
    IF (H /= 0.0D0) THEN
      DO K = 1, L
        D(K) = Z(K, I) / H
      END DO
      DO J = 1, L
        G = 0.0D0
        DO K = 1, L
          G = G + Z(K, I) * Z(K, J)
        END DO
        DO K = 1, L
          Z(K, J) = Z(K, J) - G * D(K)
        END DO
      END DO
    END IF
    DO K = 1, L
      Z(K, I) = 0.0D0
    END DO
  END DO

  ! Finalize output
  DO I = 1, N
    D(I) = Z(N, I)
    Z(N, I) = 0.0D0
  END DO
  Z(N, N) = 1.0D0
  E(1) = 0.0D0
END SUBROUTINE TRED2

SUBROUTINE TQLI(NM, N, D, E, Z, IERR)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: NM, N
  REAL(8), INTENT(INOUT) :: D(N), E(N), Z(NM, N)
  INTEGER, INTENT(OUT) :: IERR
  INTEGER :: I, J, K, L, M, II, L1, L2, MML
  REAL(8) :: C, C2, C3, DL1, EL1, F, G, H, P, R, S, S2, TST1, TST2
  REAL(8), EXTERNAL :: PYTHAG

  IERR = 0
  IF (N == 1) RETURN

  ! Shift subdiagonal elements
  DO I = 2, N
    E(I-1) = E(I)
  END DO
  E(N) = 0.0D0
  F = 0.0D0
  TST1 = 0.0D0

  ! Main QL iteration
  DO L = 1, N
    J = 0
    H = ABS(D(L)) + ABS(E(L))
    IF (TST1 < H) TST1 = H
    M = L
    DO WHILE (M <= N)
      TST2 = TST1 + ABS(E(M))
      IF (TST2 == TST1) EXIT
      M = M + 1
    END DO
    IF (M == L) THEN
      D(L) = D(L) + F
      CYCLE
    END IF
    DO
      IF (J == 30) THEN
        IERR = L
        RETURN
      END IF
      J = J + 1
      L1 = L + 1
      L2 = L1 + 1
      G = D(L)
      P = (D(L1) - G) / (2.0D0 * E(L))
      R = PYTHAG(P, 1.0D0)
      D(L) = E(L) / (P + SIGN(R, P))
      D(L1) = E(L) * (P + SIGN(R, P))
      DL1 = D(L1)
      H = G - D(L)
      IF (L2 <= N) THEN
        DO I = L2, N
          D(I) = D(I) - H
        END DO
      END IF
      F = F + H

      ! QL transformation
      P = D(M)
      C = 1.0D0
      C2 = C
      EL1 = E(L1)
      S = 0.0D0
      MML = M - L
      DO II = 1, MML
        C3 = C2
        C2 = C
        S2 = S
        I = M - II
        G = C * E(I)
        H = C * P
        R = PYTHAG(P, E(I))
        E(I+1) = S * R
        S = E(I) / R
        C = P / R
        P = C * D(I) - S * G
        D(I+1) = H + S * (C * G + S * D(I))
        DO K = 1, N
          H = Z(K, I+1)
          Z(K, I+1) = S * Z(K, I) + C * H
          Z(K, I) = C * Z(K, I) - S * H
        END DO
      END DO
      P = -S * S2 * C3 * EL1 * E(L) / DL1
      E(L) = S * P
      D(L) = C * P
      TST2 = TST1 + ABS(E(L))
      IF (TST2 <= TST1) EXIT
    END DO
    D(L) = D(L) + F
  END DO

  ! Order eigenvalues and eigenvectors
  DO II = 2, N
    I = II - 1
    K = I
    P = D(I)
    DO J = II, N
      IF (D(J) < P) THEN
        K = J
        P = D(J)
      END IF
    END DO
    IF (K /= I) THEN
      D(K) = D(I)
      D(I) = P
      DO J = 1, N
        P = Z(J, I)
        Z(J, I) = Z(J, K)
        Z(J, K) = P
      END DO
    END IF
  END DO
END SUBROUTINE TQLI

REAL(8) FUNCTION PYTHAG(A, B)
  IMPLICIT NONE
  REAL(8), INTENT(IN) :: A, B
  REAL(8) :: P, R, S, T, U
  P = MAX(ABS(A), ABS(B))
  IF (P == 0.0D0) THEN
    PYTHAG = 0.0D0
    RETURN
  END IF
  R = (MIN(ABS(A), ABS(B)) / P)**2
  DO
    T = 4.0D0 + R
    IF (T == 4.0D0) EXIT
    S = R / T
    U = 1.0D0 + 2.0D0 * S
    P = U * P
    R = (S / U)**2 * R
  END DO
  PYTHAG = P
END FUNCTION PYTHAG

SUBROUTINE EIGSRT(D, V, N, NP)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: N, NP
  REAL(8), INTENT(INOUT) :: D(NP), V(NP, NP)
  INTEGER :: I, J, K
  REAL(8) :: P
  DO I = 1, N-1
    K = I
    P = D(I)
    DO J = I+1, N
      IF (D(J) >= P) THEN
        K = J
        P = D(J)
      END IF
    END DO
    IF (K /= I) THEN
      D(K) = D(I)
      D(I) = P
      DO J = 1, N
        P = V(J, I)
        V(J, I) = V(J, K)
        V(J, K) = P
      END DO
    END IF
  END DO
END SUBROUTINE EIGSRT

REAL(8) FUNCTION RAN3(IDUM)
  IMPLICIT NONE
  INTEGER, INTENT(INOUT) :: IDUM
  INTEGER, PARAMETER :: MBIG = 1000000000, MSEED = 161803398, MZ = 0
  REAL(8), PARAMETER :: FAC = 1.0D0 / MBIG
  INTEGER :: I, IFF, II, INEXT, INEXTP, K, MJ, MK
  INTEGER :: MA(55)
  SAVE IFF, INEXT, INEXTP, MA
  DATA IFF /0/

  IF (IDUM < 0 .OR. IFF == 0) THEN
    IFF = 1
    MJ = MSEED - ABS(IDUM)
    MJ = MOD(MJ, MBIG)
    MA(55) = MJ
    MK = 1
    DO I = 1, 54
      II = MOD(21 * I, 55)
      MA(II) = MK
      MK = MJ - MK
      IF (MK < MZ) MK = MK + MBIG
      MJ = MA(II)
    END DO
    DO K = 1, 4
      DO I = 1, 55
        MA(I) = MA(I) - MA(1 + MOD(I + 30, 55))
        IF (MA(I) < MZ) MA(I) = MA(I) + MBIG
      END DO
    END DO
    INEXT = 0
    INEXTP = 31
    IDUM = 1
  END IF
  INEXT = INEXT + 1
  IF (INEXT == 56) INEXT = 1
  INEXTP = INEXTP + 1
  IF (INEXTP == 56) INEXTP = 1
  MJ = MA(INEXT) - MA(INEXTP)
  IF (MJ < MZ) MJ = MJ + MBIG
  MA(INEXT) = MJ
  RAN3 = MJ * FAC
END FUNCTION RAN3

SUBROUTINE SHELLSORT(N, IND, DIM1, ARR)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: N, IND, DIM1
  INTEGER, INTENT(INOUT) :: ARR(DIM1, N)
  INTEGER :: I, J, INC, V
  INTEGER :: VARR(DIM1)

  INC = 1
  DO
    INC = 3 * INC + 1
    IF (INC > N) EXIT
  END DO
  DO
    INC = INC / 3
    DO I = INC + 1, N
      V = ARR(IND, I)
      VARR = ARR(:, I)
      J = I
      DO WHILE (J > INC .AND. ARR(IND, J - INC) > V)
        ARR(:, J) = ARR(:, J - INC)
        J = J - INC
      END DO
      ARR(:, J) = VARR
    END DO
    IF (INC <= 1) EXIT
  END DO
END SUBROUTINE SHELLSORT

SUBROUTINE PIKSRT(N, ARR)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: N
  REAL(8), INTENT(INOUT) :: ARR(N)
  INTEGER :: I, J
  REAL(8) :: A
  DO J = 2, N
    A = ARR(J)
    I = J - 1
    DO WHILE (I >= 1 .AND. ARR(I) > A)
      ARR(I + 1) = ARR(I)
      I = I - 1
    END DO
    ARR(I + 1) = A
  END DO
END SUBROUTINE PIKSRT

