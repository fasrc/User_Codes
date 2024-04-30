# Fortran

![Fortran Logo](Images/fortran-logo.png)

[Fortran](https://en.wikipedia.org/wiki/Fortran), short for Formula Translation, is one of the oldest high-level programming languages, first developed by IBM in the 1950s. It was primarily designed for numerical and scientific computing tasks, making it highly suitable for complex mathematical calculations. Known for its efficiency and speed, Fortran has undergone several revisions over the years, with the latest version being Fortran 2018. Despite its age, Fortran remains relevant in fields such as engineering, physics, and research where performance and numerical accuracy are paramount. Its robustness in handling large-scale scientific and engineering applications, array-oriented programming, and its ability to optimize code for parallel computing have contributed to its longevity in the realm of technical computing.

## Fortran Compilers

Currently, the supported Fortran compilers on the FASRC Cannon cluster are [GNU gfortran](https://gcc.gnu.org/fortran/), [Intel ifx](https://www.intel.com/content/www/us/en/developer/tools/oneapi/fortran-compiler.html), and [NVIDIA nvfortran](https://developer.nvidia.com/hpc-sdk).

### GNU gfortran

GNU `gfortran`, part of the GNU Compiler Collection (GCC), is an open-source compiler known for its adherence to Fortran standards and portability across different platforms. To compile a Fortran program named `example.f90` with GNU `gfortran`, you can use the following commands:

```bash
# Load a GCC software module, e.g.,
module load gcc/13.2.0-fasrc01

# Compile the program, e.g.,
gfortran -O2 -o example_gfortran.x example.f90
```

This command compiles `example.f90` into an executable named `example_gfortran.x`. In the above example we also apply level 2 optimization with the `-O2` compiler flag, improving the performance of the compiled code.

### Intel ifx

Intel Fortran Compiler (`ifx`, formerly `ifort` ) is renowned for its robust optimization capabilities and superior performance, particularly on Intel architectures. Developed by Intel, `ifx` leverages advanced optimization techniques to generate highly efficient machine code tailored to Intel processors. When compiling Fortran code with `ifx`, developers can take advantage of optimizations such as auto-vectorization, inter-procedural optimization, and CPU-specific tuning to achieve significant performance improvements. To compile the same Fortran program `example.f90` with Intel `ifx`, you can use the following command:

```bash
# Load an Intel software module, e.g.,
module load intel/24.0.1-fasrc01

# Compile the program, e.g.,
ifx -O2 -o example_ifx.x example.f90
```
Similar to `gfortran`, this command compiles `example.f90` into an executable named `example_ifx.x`. Intel `ifx` typically performs optimization by default, but you can explicitly specify optimization level using the `-O2` flag.

### NVIDIA nvfortran (formerly PGI Fortran)

[NVIDIA HPC SDK](https://developer.nvidia.com/hpc-sdk) is a suite of compilers and tools designed for high-performance computing (HPC) applications, including Fortran. NVIDIA Fortran compiler provides extensive optimization capabilities tailored to both CPU and GPU architectures, making it a preferred choice for developers working in fields such as scientific computing, weather modeling, and computational fluid dynamics. When compiling Fortran code with NVIDIA `nvfortran`, developers can harness advanced optimizations like GPU offloading, where computationally intensive portions of the code are executed on NVIDIA GPUs for accelerated performance. Additionally, it offers support for directives such as [OpenACC](https://www.openacc.org/), allowing developers to easily parallelize and optimize their code for heterogeneous computing environments. While `nvfortran` excels in optimizing code for NVIDIA GPUs, it also delivers competitive performance on CPU architectures, making it a versatile choice for HPC development.

For compiling Fortran programs with NVIDIA `nvfortran`, the process may involve targeting both CPU and GPU architectures. Here's an example command to compile `example.f90` with `nvfortran` for a CPU target:

```bash
# Load a NVIDIA HPC SDK software module, e.g.,
module load nvhpc/23.7-fasrc01

# Compile the program, e.g.,
nvfortran -o example_nvfortran.x example.f90
```

This command produces an executable named `example_nvfortran.x`

## Examples

* [Example1](Example1/): Standard serial Lanczos algorithm with re-orthogonalization
* [Example2](Example2/): Computes integer sum from 1 to 100


## References

* [Fortran Programming Language](https://fortran-lang.org/)
* [Fortran Tutorial (from tutorialspoint.com)](http://www.tutorialspoint.com/fortran)
