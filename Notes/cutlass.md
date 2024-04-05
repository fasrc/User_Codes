# Cutlass

Local install following instructions from https://github.com/NVIDIA/cutlass#building-cutlass

Request a gpu node as you will need a gpu to run cutlass tests

```bash
salloc 
```

Go to a directory where you would like to install cutlass. We recommend that software is installed lab shares rather than home directoris for better perfomance. Then clone cutlass repository

```bash
git clone https://github.com/NVIDIA/cutlass.git cutlass_gpu
```

Load modules according to dependencies listed in https://github.com/NVIDIA/cutlass/blob/main/media/docs/quickstart.md#prerequisites

```
[paulasan@holygpu8a22105 cutlass_gpu]$ module load cuda/12.2.0-fasrc01
[paulasan@holygpu8a22105 cutlass_gpu]$ module load cmake/3.28.3-fasrc01
[paulasan@holygpu8a22105 cutlass_gpu]$ module load gcc/12.2.0-fasrc01
[paulasan@holygpu8a22105 cutlass_gpu]$ module list

Currently Loaded Modules:
  1) cuda/12.2.0-fasrc01   2) cmake/3.28.3-fasrc01   3) gmp/6.2.1-fasrc01   4) mpfr/4.2.0-fasrc01   5) mpc/1.3.1-fasrc01   6) gcc/12.2.0-fasrc01
```

Set environmental variables and build

```bash
# set environmental variables
[paulasan@holygpu8a22105 cutlass_gpu]$ export CUDACXX=${CUDA_HOME}bin/nvcc
[paulasan@holygpu8a22105 cutlass_gpu]$ echo $CUDACXX
/n/sw/helmod-rocky8/apps/Core/cuda/12.2.0-fasrc01/cuda/bin/nvcc

# build
[paulasan@holygpu8a22105 cutlass_gpu]$ mkdir build
[paulasan@holygpu8a22105 cutlass_gpu]$ cd build/
[paulasan@holygpu8a22105 build]$ cmake .. -DCUTLASS_NVCC_ARCHS=80 -DCMAKE_INSTALL_PREFIX="/path_to_my_software/cutlass_gpu/build"

# make trial -- this can take 1-2 hours
[paulasan@holygpu8a22105 build]$ make test_unit -j 4
```

## Example output

Build

```bash
[paulasan@holygpu8a22105 build]$ cmake .. -DCUTLASS_NVCC_ARCHS=80 -DCMAKE_INSTALL_PREFIX="/n/holylfs06/LABS/rc_admin/Lab/paulasan/cutlass_gpu/build"
-- CMake Version: 3.28.3
-- CUTLASS 3.5.0
-- The CXX compiler identification is GNU 12.2.0
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /n/sw/helmod-rocky8/apps/Core/gcc/12.2.0-fasrc01/bin/g++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- The CUDA compiler identification is NVIDIA 12.2.91
-- Detecting CUDA compiler ABI info
-- Detecting CUDA compiler ABI info - done
-- Check for working CUDA compiler: /n/sw/helmod-rocky8/apps/Core/cuda/12.2.0-fasrc01/cuda/bin/nvcc - skipped
-- Detecting CUDA compile features
-- Detecting CUDA compile features - done
-- CUDART: /n/sw/helmod-rocky8/apps/Core/cuda/12.2.0-fasrc01/cuda/lib64/libcudart.so
-- CUDA Driver: /n/sw/helmod-rocky8/apps/Core/cuda/12.2.0-fasrc01/cuda/lib64/stubs/libcuda.so
-- NVRTC: /n/sw/helmod-rocky8/apps/Core/cuda/12.2.0-fasrc01/cuda/lib64/libnvrtc.so
-- Default Install Location: /n/holylfs06/LABS/rc_admin/Lab/paulasan/cutlass_gpu/build
-- Found Python3: /usr/bin/python3.6 (found suitable version "3.6.8", minimum required is "3.5") found components: Interpreter
-- CUDA Compilation Architectures: 80
-- Enable caching of reference results in conv unit tests
-- Enable rigorous conv problem sizes in conv unit tests
-- Using NVCC flags: --expt-relaxed-constexpr;-DCUTLASS_TEST_LEVEL=0;-DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1;-DCUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED=1;-DCUTLASS_DEBUG_TRACE_LEVEL=0;-Xcompiler=-Wconversion;-Xcompiler=-fno-strict-aliasing
-- CUTLASS Revision: 19f3cc33
-- The C compiler identification is GNU 12.2.0
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /n/sw/helmod-rocky8/apps/Core/gcc/12.2.0-fasrc01/bin/gcc - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Found Python3: /usr/bin/python3.6 (found version "3.6.8") found components: Interpreter
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Failed
-- Looking for pthread_create in pthreads
-- Looking for pthread_create in pthreads - not found
-- Looking for pthread_create in pthread
-- Looking for pthread_create in pthread - found
-- Found Threads: TRUE
-- Configuring cublas ...
-- cuBLAS Disabled.
-- Configuring cuBLAS ... done.
-- Completed generation of library instances. See /n/holylfs06/LABS/rc_admin/Lab/paulasan/cutlass_gpu/build/tools/library/library_instance_generation.log for more information.
-- Found Python3: /usr/bin/python3.6 (found suitable version "3.6.8", minimum required is "3.5") found components: Interpreter
-- Enable device reference verification in conv unit tests
-- Configuring done (9.6s)
-- Generating done (9.6s)
-- Build files have been written to: /n/holylfs06/LABS/rc_admin/Lab/paulasan/cutlass_gpu/build
```

Make: the make output is very long; this is just a small fraction of the ouput

```bash
... omitted output ...

[ 57%] Building CXX object test/unit/cute/core/CMakeFiles/cutlass_test_unit_cute_core.dir/bitfield.cpp.o
[ 57%] Built target cutlass_test_unit_cute_msvc_compilation
[ 57%] Built target cutlass_test_unit_nvrtc_thread
[ 57%] Building CUDA object test/unit/cute/volta/CMakeFiles/cutlass_test_unit_cute_volta.dir/vectorization_auto.cu.o
[ 57%] Building CUDA object test/unit/core/CMakeFiles/cutlass_test_unit_core.dir/numeric_conversion.cu.o
[ 57%] Building CUDA object test/unit/cute/ampere/CMakeFiles/cutlass_test_unit_cute_ampere.dir/cp_async.cu.o
[ 57%] Building CUDA object test/unit/cute/ampere/CMakeFiles/cutlass_test_unit_cute_ampere.dir/ldsm.cu.o
[ 57%] Building CUDA object test/unit/cute/volta/CMakeFiles/cutlass_test_unit_cute_volta.dir/cooperative_copy.cu.o
[ 57%] Linking CXX executable cutlass_test_unit_cute_ampere
[ 57%] Built target cutlass_test_unit_cute_ampere
[ 57%] Building CUDA object test/unit/cute/hopper/CMakeFiles/cutlass_test_unit_cute_hopper_bulk_store.dir/bulk_store.cu.o
[ 57%] Linking CXX executable cutlass_test_unit_core
[ 60%] Built target cutlass_test_unit_core

... omitted output ...

[100%] Built target test_unit_cute_layout
Note: Google Test filter = -SM70*:SM89*:SM90*
[==========] Running 415 tests from 194 test suites.
[----------] Global test environment set-up.
[----------] 4 tests from SM80_Device_Syrk_f64n_f64t_l_tensor_op_f64
[ RUN      ] SM80_Device_Syrk_f64n_f64t_l_tensor_op_f64.32x32x16_16x16x16
Note: Google Test filter = -SM70*:SM89*:SM90*
[==========] Running 123 tests from 55 test suites.
[----------] Global test environment set-up.
[----------] 7 tests from SM80_Device_Gemm_f32n_f32t_f32t_simt_f32
[ RUN      ] SM80_Device_Gemm_f32n_f32t_f32t_simt_f32.32x64x8_32x64x1
[       OK ] complex.host_double (422 ms)
[----------] 7 tests from complex (933 ms total)

... omitted output ...

[----------] 3 tests from SM80_Device_Conv2d_Group_Fprop_Optimized_ImplicitGemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f32
[ RUN      ] SM80_Device_Conv2d_Group_Fprop_Optimized_ImplicitGemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f32.SingleGroupPerCTA_128x128_64x3_64x64x64
[       OK ] SM80_Device_Conv2d_Group_Fprop_Optimized_ImplicitGemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f32.SingleGroupPerCTA_128x128_64x3_64x64x64 (1364 ms)
[ RUN      ] SM80_Device_Conv2d_Group_Fprop_Optimized_ImplicitGemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f32.SingleGroupPerCTA_64x64_64x3_32x32x64
[       OK ] SM80_Device_Conv2d_Group_Fprop_Optimized_ImplicitGemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f32.SingleGroupPerCTA_64x64_64x3_32x32x64 (1041 ms)
[ RUN      ] SM80_Device_Conv2d_Group_Fprop_Optimized_ImplicitGemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f32.SingleGroupPerCTA_64x64_64x2_32x32x64
[       OK ] SM80_Device_Conv2d_Group_Fprop_Optimized_ImplicitGemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f32.SingleGroupPerCTA_64x64_64x2_32x32x64 (1010 ms)
[----------] 3 tests from SM80_Device_Conv2d_Group_Fprop_Optimized_ImplicitGemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f32 (3416 ms total)

[----------] Global test environment tear-down
[==========] 37 tests from 23 test suites ran. (642888 ms total)
[  PASSED  ] 37 tests.
[100%] Built target test_unit_conv_device_tensorop_f32_sm80
[       OK ] SM75_Device_Conv3d_Wgrad_Analytic_ImplicitGemm_f16ndhwc_f16ndhwc_f32ndhwc_tensor_op_f32.128x128_32x2_64x64x32 (33386 ms)
[----------] 1 test from SM75_Device_Conv3d_Wgrad_Analytic_ImplicitGemm_f16ndhwc_f16ndhwc_f32ndhwc_tensor_op_f32 (33386 ms total)

[----------] Global test environment tear-down
[==========] 17 tests from 17 test suites ran. (892467 ms total)
[  PASSED  ] 17 tests.
[100%] Built target test_unit_conv_device_tensorop_f32_sm75
[100%] Built target test_unit_conv_device
[100%] Built target test_unit_conv
[100%] Built target test_unit
```

## Troubleshooting

During `make`, if you get a `killed` error, you are running out of memory. You can kill the process and either (1) run the make command with less cores, i.e.`make test_unit -j 4` or (2) you can kill the interactive job, request another interactive job with more memory and run `make test_unit -j 4` again. 
