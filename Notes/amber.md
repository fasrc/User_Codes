# Amber22 with AmberTools23

## Serial version

Download Amber22 and AmberTools23 from Amber website https://ambermd.org/index.php

```bash
[jharvard@boslogin02 ~]$ salloc --partition test,sapphire --time 04:00:00 -c 8 --mem-per-cpu 4G
[jharvard@holy8a24501 software]$ tar xvfj AmberTools23.tar.bz2
[jharvard@holy8a24501 software]$ tar xvfj Amber22.tar.bz2
[jharvard@holy8a24501 software]$ cd amber22_src/
[jharvard@holy8a24501 amber22_src]$ cd build/
[jharvard@holy8a24501 build]$ module load cmake/3.27.5-fasrc01
[jharvard@holy8a24501 build]$ module load gcc/12.2.0-fasrc01 openmpi/4.1.5-fasrc02
[jharvard@holy8a24501 build]$ ./run_cmake
[jharvard@holy8a24501 build]$ make install
[jharvard@holy8a24501 build]$ cd ..
[jharvard@holy8a24501 amber22_src]$ cd ..
[jharvard@holy8a24501 software]$ cd amber22
[jharvard@holy8a24501 amber22]$ source amber.sh
[jharvard@holy8a24501 amber22]$ echo $AMBERHOME
/n/holylfs05/LABS/rc_admin/Lab/jharvard/software/amber22

# this takes a long time, ~1h
[jharvard@holy8a24501 amber22]$ make test.serial

... omitted output ...

Finished serial test suite for Amber 22 at Thu Mar  7 16:15:06 EST 2024.

make[2]: Leaving directory '/n/holylfs05/LABS/rc_admin/Lab/jharvard/software/amber22/test'
170 file comparisons passed
7 file comparisons failed (2 of which can be ignored)
0 tests experienced errors
Test log file saved as /n/holylfs05/LABS/rc_admin/Lab/jharvard/software/amber22/logs/test_amber_serial/2024-03-07_16-10-14.log
Test diffs file saved as /n/holylfs05/LABS/rc_admin/Lab/jharvard/software/amber22/logs/test_amber_serial/2024-03-07_16-10-14.diff
make[1]: *** [Makefile:15: test] Error 1
make[1]: Leaving directory '/n/holylfs05/LABS/rc_admin/Lab/jharvard/software/amber22/test'

Summary of AmberTools serial tests:

2583 file comparisons passed
46 file comparisons failed (4 of which can be ignored)
12 tests experienced errors
Test log file saved as /n/holylfs05/LABS/rc_admin/Lab/jharvard/software/amber22///logs/test_at_serial/2024-03-07_15-26-38.log
Test diffs file saved as /n/holylfs05/LABS/rc_admin/Lab/jharvard/software/amber22///logs/test_at_serial/2024-03-07_15-26-38.diff
```

## MPI and Cuda version

```
# it's very important that you request enough resources, otherwise the install will take a long time
[jharvard@boslogin02 amber_cuda_mpi]$ salloc --partition gpu --time 04:00:00 -c 8 --mem-per-cpu 4G -gres:=gpu:1
[jharvard@holygpu8a22101 amber_cuda_mpi]$ tar xvfj AmberTools23.tar.bz2
[jharvard@holygpu8a22101 amber_cuda_mpi]$ tar xvfj Amber22.tar.bz2
[jharvard@holygpu8a22101 amber_cuda_mpi]$ cd amber22_src/
[jharvard@holygpu8a22101 amber22_src]$ cd build/
[jharvard@holygpu8a22101 build]$ module load cmake/3.27.5-fasrc01 gcc/10.2.0-fasrc01 openmpi/4.1.3-fasrc01



[jharvard@holygpu8a22101 build]$ module list

Currently Loaded Modules:
  1) cmake/3.27.5-fasrc01   3) mpfr/4.1.0-fasrc01   5) gcc/10.2.0-fasrc01    7) openmpi/4.1.3-fasrc01
  2) gmp/6.2.1-fasrc01      4) mpc/1.2.1-fasrc01    6) cuda/11.8.0-fasrc01


# Open the file run_cmake and edit on line 42:
-DMPI=TRUE -DCUDA=TRUE


[jharvard@holygpu8a22101 build]$ ./run_cmake
[jharvard@holygpu8a22101 build]$ make install
[jharvard@holygpu8a22102 build]$ cd ..
[jharvard@holygpu8a22102 amber22_src]$ cd ..
[jharvard@holygpu8a22102 amber_cuda_mpi]$ cd amber22
[jharvard@holygpu8a22102 amber22]$ source amber.sh
[jharvard@holygpu8a22102 amber22]$ echo $AMBERHOME
/n/holylfs05/LABS/rc_admin/Lab/jharvard/amber_cuda_mpi/amber22
[jharvard@holygpu8a22102 amber22]$ time make test.cuda.serial
```

> [!WARNING]
> Amber need cuda version \>= 7.5 and \<= 12.1. Given these
constraints, module `openmpi/4.1.5-fasrc02` does not work because it was built
with cuda 12.2.0. When attempting to build with `openmpi/4.1.5-fasrc02`, amber
gives this error:

```
-- CUDA version 12.2 detected
CMake Error at cmake/CudaConfig.cmake:84 (message):
  Error: Untested CUDA version.  AMBER currently requires CUDA version >= 7.5
  and <= 12.1.
Call Stack (most recent call first):
  CMakeLists.txt:119 (include)
```
