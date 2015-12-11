!=====================================================================
! Program: fftw_test.f90
!          Perform 1D  transform in Fortran
!=====================================================================
program test
  implicit none
  include "fftw3.f"
  integer(4)            :: i
  integer(4), parameter :: N = 4
  integer(8)            :: plan
  complex(8)            :: in(N)
  complex(8)            :: out(N)

  write(6,*) 'Input array:'

  do i = 1, N, 1
     in(i) = dcmplx(float(i),float(i+1))
     write(6,*) '    in(',i,') = ',in(i)
  end do

  call dfftw_plan_dft_1d ( plan, N, in, out,FFTW_FORWARD, FFTW_ESTIMATE )

  call dfftw_execute ( plan )

  write(6,*) 'Output array:'
  do i = 1,N,1
     write(6,*) '    out(',i,') = ',out(i)
  end do

  call dfftw_destroy_plan ( plan )

  call dfftw_plan_dft_1d ( plan, N, out, in, FFTW_FORWARD, FFTW_ESTIMATE )

  call dfftw_execute ( plan )

  write(6,*) 'Output array after inverse FFT:'
  do i = 1,N,1
     write(6,*) '    ',N,' * in(',i,') = ',in(i)
  end do

  call dfftw_destroy_plan ( plan )

  stop 'End of program'
  
end program test
