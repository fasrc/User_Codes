# Using OpenACC on the FASRC Cluster

<img src="openacc-logo.jpeg" alt="openacc-logo" width="200"/>

[OpenACC](https://www.openacc.org/) is an open standard for parallel computing that enables developers to accelerate their applications across a range of hardware platforms, including GPUs and multicore CPUs. OpenACC provides a high-level, directive-based programming model that allows programmers to specify which parts of their code should be parallelized and how they should be parallelized, without having to write low-level, platform-specific code. This makes it much easier for developers to take advantage of the parallel processing capabilities of modern hardware architectures, without having to become experts in low-level parallel programming.

One of the key benefits of OpenACC is that it allows programmers to write code that can run on a range of different hardware platforms, without having to make significant modifications to the code itself. This is particularly useful in scientific computing, where researchers often need to run their simulations on a variety of different supercomputers and clusters. With OpenACC, developers can write code once and then easily port it to different platforms, making it much easier to take advantage of the latest hardware innovations and achieve the best possible performance.

## **Example:** SAXPY in Fortran and OpenACC

Here we use OpenACC to perform a SAXPY operation in Fortran.

```fortran
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

   integer, parameter :: n = huge(n)
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
```

## Compile the code

The code can be compiled with the NVIDIA Fortran compiler, which is included in the NVIDIA HPC SDK. You can use the below commands to load the NVIDIA HPC SDK software module and compile the code:

```bash
module load nvhpc/23.7-fasrc01
nvfortran -o example_acc.x example_acc.f90 -acc
```

## Example batch-job submission script

```bash
#!/bin/bash
#SBATCH -p gpu_test
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH --mem=12000
#SBATCH -J openacc_test
#SBATCH -o openacc_test.out
#SBATCH -e openacc_test.err
#SBATCH -t 30

# Load required modules
module load nvhpc/23.7-fasrc01

# Run the executable
./example_acc.x 
```
Assuming the batch-job submission script is named <code>run.sbatch</code>, the jobs is sent to the queue, as usual, with:

```bash
sbatch run.sbatch
```
## Example Output

```bash
cat openacc_test.out 
 Initializing X and Y...
 Computing the SAXPY operation...
```

## References

* [OpenACC tutorial from Western Virginia University](https://wvuhpc.github.io/Modern-Fortran/30-OpenACC/index.html)
* [Official OpenACC Website](https://www.openacc.org)
* [NVIDIA OpenACC Website](https://developer.nvidia.com/openacc)