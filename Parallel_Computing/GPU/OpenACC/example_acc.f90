module mod_saxpy

contains

   subroutine saxpy(n, a, x, y)

      implicit none

      real :: x(:), y(:), a
      integer :: n, i

!$ACC PARALLEL LOOP
      do i = 1, n
         y(i) = a*x(i) + y(i)
      end do
!$ACC END PARALLEL LOOP

   end subroutine saxpy

end module mod_saxpy

program main

   use mod_saxpy

   implicit none

   integer, parameter :: n = 100000
   real :: x(n), y(n), a = 2.3
   integer :: i

   print *, "Initializing X and Y..."

!$ACC PARALLEL LOOP
   do i = 1, n
      x(i) = sqrt(real(i))
      y(i) = sqrt(1.0/real(i))
   end do
!$ACC END PARALLEL LOOP

   print *, "Computing the SAXPY operation..."

!$ACC PARALLEL LOOP
   do i = 1, n
      y(i) = a*x(i) + y(i)
   end do
!$ACC END PARALLEL LOOP

   call saxpy(n, a, x, y)

end program main

