!=====================================================================
! Module: rng_module
!         Defines the RNGState type for thread-safe random number generation
!=====================================================================
module rng_module
  implicit none
  type :: RNGState
     integer(4) :: ma(55)
     integer(4) :: inext, inextp, iff
  end type RNGState
end module rng_module

!=====================================================================
! Program: omp_mem_test.f90
!          Program generates a symmetric random matrix of dimension 60K
!=====================================================================
program mem_test
  use omp_lib       ! Import OpenMP library for thread functions
  use rng_module    ! Import RNGState type
  implicit none
  integer(4) :: n = 60000 ! Matrix dimension
  integer(4) :: i, j
  integer(4) :: iseed
  type(RNGState) :: state
  real(8), allocatable :: h(:,:)

  ! Random number generator function
  real(8), external :: ran3

  ! Allocate memory
  if (.not. allocated(h)) allocate(h(n,n))

  ! Create random test matrix with OpenMP
  !$omp parallel private(i, j, iseed, state)
    iseed = -(99 + omp_get_thread_num()) ! Unique seed per thread
    state%ma = 0
    state%inext = 0
    state%inextp = 0
    state%iff = 0

    !$omp do schedule(dynamic)
    do i = 1, n
       do j = 1, i
          h(i,j) = ran3(iseed, state)
          h(j,i) = h(i,j)
       end do
    end do
    !$omp end do
  !$omp end parallel

  write(6,*) 'Hamiltonian matrix created successfully with ', &
             omp_get_max_threads(), ' threads (n=', n, ')!'

  ! Free memory
  if (allocated(h)) deallocate(h)

end program mem_test

!=====================================================================
!     The function
!        ran3
!     returns a uniform random number deviate between 0.0 and 1.0. Set
!     the idum to any negative value to initialize or reinitialize the
!     sequence. Thread-safe version with state passed as argument.
!=====================================================================
real(8) function ran3(idum, state)
  use rng_module    ! Import RNGState type
  implicit none
  integer(4), intent(inout) :: idum
  type(RNGState), intent(inout) :: state

  integer(4), parameter :: mbig = 1000000000, mseed = 161803398, mz = 0
  real(8), parameter :: fac = 1.0d0 / mbig
  integer(4) :: i, ii, k
  integer(4) :: mj, mk

  if ((idum < 0) .or. (state%iff == 0)) then
     state%iff = 1
     mj = mseed - iabs(idum)
     mj = mod(mj, mbig)
     state%ma(55) = mj
     mk = 1
     do i = 1, 54
        ii = mod(21 * i, 55)
        state%ma(ii) = mk
        mk = mj - mk
        if (mk < mz) mk = mk + mbig
        mj = state%ma(ii)
     end do
     do k = 1, 4
        do i = 1, 55
           state%ma(i) = state%ma(i) - state%ma(1 + mod(i + 30, 55))
           if (state%ma(i) < mz) state%ma(i) = state%ma(i) + mbig
        end do
     end do
     state%inext = 0
     state%inextp = 31
     idum = 1
  end if

  state%inext = state%inext + 1
  if (state%inext == 56) state%inext = 1
  state%inextp = state%inextp + 1
  if (state%inextp == 56) state%inextp = 1
  mj = state%ma(state%inext) - state%ma(state%inextp)
  if (mj < mz) mj = mj + mbig
  state%ma(state%inext) = mj
  ran3 = mj * fac
end function ran3
