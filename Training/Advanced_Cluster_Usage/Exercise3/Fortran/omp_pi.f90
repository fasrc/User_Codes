!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
! Program: omp_pi.f90
!          OpenMP implementation of Monte-Carlo algorithm for calculating PI
!          Translated from omp_pi.c, adapted for gfortran
!
! Compile: gfortran -o omp_pi.x omp_pi.f90 -O2 -fopenmp
! 
! Run:     ./omp_pi.x <number_of_samples> <number_of_threads>
!          OMP_NUM_THREADS=<number_of_threads> ./omp_pi.x (optional override)
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
PROGRAM omp_pi
  USE OMP_LIB   ! OpenMP library
  IMPLICIT NONE
  
  INTEGER(4) :: i, count, samples, nthreads, tid
  INTEGER(4) :: seed_size
  INTEGER(4), ALLOCATABLE :: seed(:)
  REAL(8)    :: x, y, z
  REAL(8)    :: t0, t1, tf, PI
  REAL(8), PARAMETER :: PI_EXACT = 3.14159265358979323846D0
  CHARACTER(LEN=20) :: arg1, arg2
  
  ! Get command-line arguments
  CALL GET_COMMAND_ARGUMENT(1, arg1)
  CALL GET_COMMAND_ARGUMENT(2, arg2)
  READ(arg1, *) samples   ! Number of samples
  READ(arg2, *) nthreads  ! Number of threads
  
  CALL OMP_SET_NUM_THREADS(nthreads)
  
  WRITE(*,'(A,I2)') "Number of threads: ", nthreads
  
  t0 = OMP_GET_WTIME()
  count = 0
  
  !$OMP PARALLEL PRIVATE(i, x, y, z, tid, seed) SHARED(samples)
    tid = OMP_GET_THREAD_NUM()
    
    ! Thread-specific seed for intrinsic RNG
    CALL RANDOM_SEED(SIZE=seed_size)
    ALLOCATE(seed(seed_size))
    seed = 1202107158 + tid * 1999  ! Match C seeding strategy
    CALL RANDOM_SEED(PUT=seed)
    DEALLOCATE(seed)
    
    !$OMP DO REDUCTION(+:count)
    DO i = 0, samples - 1           ! Match C loop range (0 to samples-1)
      CALL RANDOM_NUMBER(x)
      CALL RANDOM_NUMBER(y)
      z = x*x + y*y
      IF (z <= 1.0D0) count = count + 1
    END DO
    !$OMP END DO
  !$OMP END PARALLEL
  
  t1 = OMP_GET_WTIME()
  tf = t1 - t0
  
  ! Estimate PI
  PI = 4.0D0 * REAL(count, KIND=8) / REAL(samples, KIND=8)
  
  WRITE(*,'(A,F7.5)') "Exact value of PI: ", PI_EXACT
  WRITE(*,'(A,F7.5)') "Estimate of PI:    ", PI
  WRITE(*,'(A,F7.2,A)') "Time: ", tf, " sec."
  
END PROGRAM omp_pi
