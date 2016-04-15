!=====================================================================
!
! Program: openblas_test.f90
!          Program illustrates use of OpenBLAS
!
! Test prob: [ 1 2 3 ][x]   [ 6] ,  w/ true sol'n  [1]
!            [ 4 5 6 ][y] = [15]                   [1]
!            [ 7 8 0 ][z]   [15]                   [1]
!
!=====================================================================
program test
  character(1) :: trans
  dimension A(4,5), b(3), ipiv(3) ! deliberately chosen larger dimensions
                                  ! to illustrate role of 'lda'

  data A / 1., 4., 7., 0., 2., 5., 8., 0., 3., 6., 0., 0., 8*0. /
  data b / 6., 15., 15. /
  data trans, m, n, lda, ldb, nrhs / 'N', 3, 3, 4, 3, 1 /

  call sgetrf( m, n, A, lda, ipiv, info )                      ! LU-factor

  write(6,*) ' info, ipiv = ', info, ipiv

  call sgetrs( trans, n, nrhs, A, lda, ipiv, b, ldb, info )    ! back solve

  write(6,*) ' info, b = ', info, b

  stop 'End of program.'
end program test
!=====================================================================
! Appended sample output:
!
!  info, ipiv =  0 3 3 3
!  info, b =  0  1.00000012  0.999999881  1.
!
!=====================================================================
