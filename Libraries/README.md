# Scientific Libraries - Examples and Usage Guide

**Generated on: October 28, 2025 at 22:00 UTC**

## Overview

This directory contains practical examples demonstrating how to use various scientific computing and I/O libraries on HPC clusters. Each subdirectory includes working code examples, compilation instructions, and SLURM job submission scripts.

## Library Examples

### Linear Algebra Libraries

#### [BLAS/](BLAS/) - Basic Linear Algebra Subprograms
Fundamental matrix and vector operations.
- [`blas_test.f90`](BLAS/blas_test.f90) - Fortran example using DGEMM (matrix multiplication)
- [`Makefile`](BLAS/Makefile) - Compilation with BLAS linking
- [`blas_test.sbatch`](BLAS/blas_test.sbatch) - SLURM job submission
- [`README.md`](BLAS/README.md) - BLAS usage instructions
- **Operations demonstrated**: Matrix multiplication, vector operations
- **Module required**: BLAS library (various implementations)

#### [LAPACK/](LAPACK/) - Linear Algebra PACKage
Advanced linear algebra routines built on BLAS.
- [`lapack_test.f90`](LAPACK/lapack_test.f90) - Eigenvalue computation example
- [`Makefile`](LAPACK/Makefile) - Build configuration with LAPACK
- [`run.sbatch`](LAPACK/run.sbatch) - Job submission script
- [`README.md`](LAPACK/README.md) - LAPACK setup and usage
- **Operations demonstrated**: Eigenvalue/eigenvector computation
- **Dependencies**: BLAS and LAPACK libraries

#### [OpenBLAS/](OpenBLAS/) - Optimized BLAS Library
Open-source optimized BLAS implementation.
- [`openblas_test.f90`](OpenBLAS/openblas_test.f90) - Performance comparison example
- [`Makefile`](OpenBLAS/Makefile) - OpenBLAS-specific compilation
- [`run.sbatch`](OpenBLAS/run.sbatch) - Submission script
- [`README.md`](OpenBLAS/README.md) - OpenBLAS configuration
- **Features**: Multi-threaded BLAS operations
- **Performance**: Optimized for various CPU architectures

#### [MKL/](MKL/) - Intel Math Kernel Library
Intel's optimized math library suite.
- [`mkltest.f90`](MKL/mkltest.f90) - MKL-specific features demonstration
- [`Makefile`](MKL/Makefile) - Intel compiler and MKL linking
- [`mkltest.sbatch`](MKL/mkltest.sbatch) - SLURM job script
- [`README.md`](MKL/README.md) - MKL module loading and usage
- **Features**: Highly optimized for Intel processors
- **Includes**: BLAS, LAPACK, FFT, and more

#### [Armadillo/](Armadillo/) - C++ Linear Algebra Library
High-level C++ library for linear algebra.
- [`armadillo_test.cpp`](Armadillo/armadillo_test.cpp) - C++ matrix operations example
- [`Makefile`](Armadillo/Makefile) - C++ compilation with Armadillo
- [`run.sbatch`](Armadillo/run.sbatch) - Job submission
- [`README.md`](Armadillo/README.md) - Armadillo installation and usage
- **Language**: C++ template library
- **Features**: MATLAB-like syntax, automatic expression optimization

### Numerical Computing Libraries

#### [GSL/](GSL/) - GNU Scientific Library
Comprehensive numerical library for C/C++.
- [`gsl_int_test.c`](GSL/gsl_int_test.c) - Numerical integration example
- [`Makefile`](GSL/Makefile) - GSL compilation flags
- [`gsl_int_test.sbatch`](GSL/gsl_int_test.sbatch) - SLURM submission
- [`README.md`](GSL/README.md) - GSL module and usage guide
- **Features**: Integration, differentiation, special functions, statistics
- **Language**: C with C++ compatibility

#### [FFTW/](FFTW/) - Fastest Fourier Transform in the West
Highly optimized FFT library.
- [`fftw_test.f90`](FFTW/fftw_test.f90) - FFT computation example
- [`Makefile`](FFTW/Makefile) - FFTW linking configuration
- [`fftw_test.sbatch`](FFTW/fftw_test.sbatch) - Job submission script
- [`README.md`](FFTW/README.md) - FFTW setup instructions
- **Operations**: 1D, 2D, 3D FFTs
- **Features**: Real and complex transforms, parallel versions

### Data I/O Libraries

#### [HDF5/](HDF5/) - Hierarchical Data Format
High-performance data management and storage.
- [`hdf5_test.f90`](HDF5/hdf5_test.f90) - HDF5 file I/O example
- [`Makefile`](HDF5/Makefile) - HDF5 compilation flags
- [`README.md`](HDF5/README.md) - HDF5 module loading and usage
- **Features**: Portable file format, parallel I/O support
- **Use cases**: Large scientific datasets, complex data structures

#### [NETCDF/](NETCDF/) - Network Common Data Form
Self-describing, portable data format.
- [`netcdf_test.f90`](NETCDF/netcdf_test.f90) - NetCDF read/write example
- [`Makefile`](NETCDF/Makefile) - NetCDF library linking
- [`netcdf_test.sbatch`](NETCDF/netcdf_test.sbatch) - SLURM job script
- [`README.md`](NETCDF/README.md) - NetCDF configuration guide
- **Features**: Array-oriented scientific data
- **Common uses**: Climate data, atmospheric science, oceanography

## Quick Reference by Programming Language

### Fortran Examples
- [BLAS/](BLAS/) - Matrix multiplication with DGEMM
- [LAPACK/](LAPACK/) - Eigenvalue computation
- [OpenBLAS/](OpenBLAS/) - Optimized BLAS operations
- [MKL/](MKL/) - Intel MKL features
- [FFTW/](FFTW/) - Fast Fourier transforms
- [HDF5/](HDF5/) - HDF5 file I/O
- [NETCDF/](NETCDF/) - NetCDF data handling

### C/C++ Examples
- [Armadillo/](Armadillo/) - C++ linear algebra (C++)
- [GSL/](GSL/) - Numerical integration (C)

## Common Compilation Patterns

### Basic BLAS/LAPACK Linking
```makefile
FC = gfortran
LIBS = -lblas -llapack
```

### Intel MKL Linking
```makefile
FC = ifort
FFLAGS = -mkl
```

### HDF5 Compilation
```makefile
FC = h5fc
FFLAGS = -O3
```

### GSL Linking
```makefile
CC = gcc
LIBS = -lgsl -lgslcblas -lm
```

## Module Loading Examples

### For BLAS/LAPACK
```bash
module load gcc/13.2.0-fasrc01
module load openblas/0.3.24-fasrc01
```

### For Intel MKL
```bash
module load intel/24.0.1-fasrc01
# MKL is included with Intel compiler
```

### For HDF5/NetCDF
```bash
module load gcc/13.2.0-fasrc01
module load hdf5/1.14.3-fasrc01
module load netcdf/4.9.2-fasrc01
```

## Usage Notes

1. **Check available modules**: Use `module spider [library]` to find available versions
2. **Linking order matters**: Some libraries depend on others (e.g., LAPACK needs BLAS)
3. **Performance considerations**: 
   - MKL is optimized for Intel processors
   - OpenBLAS provides good performance across different architectures
   - Consider thread settings for multi-threaded libraries
4. **Compilation helpers**: Some libraries provide wrapper scripts (e.g., `h5fc` for HDF5)

## Getting Started

1. Choose the library example that matches your needs
2. Navigate to the specific directory
3. Load required modules (check the README in each directory)
4. Compile using the provided Makefile
5. Submit the job using the SLURM script
6. Check output files for results

## Typical Workflow

```bash
cd Libraries/BLAS/
module load gcc/13.2.0-fasrc01 openblas/0.3.24-fasrc01
make
sbatch blas_test.sbatch
```

## Performance Tips

- **Thread control**: Set `OMP_NUM_THREADS` for multi-threaded libraries
- **Library selection**: Choose based on your hardware (Intel MKL for Intel CPUs)
- **Linking optimization**: Static linking can improve performance but increases binary size
- **Memory alignment**: Some libraries benefit from aligned memory allocation

## Related Resources

- [Main User_Codes Repository](../)
- [BLAS Documentation](http://www.netlib.org/blas/)
- [LAPACK Users' Guide](http://www.netlib.org/lapack/lug/)
- [Intel MKL Documentation](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html)
- [HDF5 Support](https://www.hdfgroup.org/solutions/hdf5/)
- [NetCDF Documentation](https://www.unidata.ucar.edu/software/netcdf/)

---

*This directory provides working examples of scientific computing libraries commonly used in HPC environments. Each example is self-contained and can be adapted for specific computational needs.*