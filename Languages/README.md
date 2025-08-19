# Languages Examples

Run, build, and adapt language-specific code and job scripts for common HPC tasks from this collection.

For general guidance on using languages on the cluster, see the [RC User Documentation](https://docs.rc.fas.harvard.edu/).

## Systems & Build Automation

### Shell & Job Submission
- **[BASH](./BASH/)**: Utility scripts and SLURM-ready job wrappers demonstrating sbatch, srun, and environment setup for reproducible runs.

### Build Tools and C/C++ Compilation
- **[C](./C/)**: Small, portable programs with compilation flags and makefile snippets; see the summation program at [sum.c](./C/Example/sum.c) for a minimal build-and-run workflow.
- **[Cpp](./Cpp/)**: C++ examples covering modern build conventions, template usage, and parallel compile optimizations for scientific codebases.

## Compiled Scientific Languages

### Fortran Numerical Codes
- **[Fortran](./Fortran/)**: Numerical kernels and solver templates optimized for HPC compilers; run the Lanczos algorithm at [lanczos.f90](./Fortran/Example1/lanczos.f90) and a simple reducer at [sum.f90](./Fortran/Example2/sum.f90).

### Legacy & Domain-specific Languages
- **[IDL](./IDL/)**: Array-processing routines and visualization scripts tailored to remote-sensing and astronomy workflows.
- **[Perl](./Perl/)**: Text-processing pipelines and small data munging utilities used in file preparation and batch preprocessing.

## Interactive & High-level Scientific Languages

### Python Data Science and Testing
- **[Python](./Python/)**: Analysis, testing, and Monte Carlo examples with common libraries and test patterns.
  - **[myscript.py](./Python/myscript.py)**: Simple script illustrating interpreter usage and environment shebangs.
  - **[numpy_pandas_ex.py](./Python/Example2/numpy_pandas_ex.py)**: Create and manipulate a pandas DataFrame for data ingestion and cleaning.
  - **[mc_pi.py](./Python/Example1/mc_pi.py)** and **[test_mc_pi.py](./Python/Example1/test_mc_pi.py)**: Monte Carlo estimator with a unit test demonstrating seed control for reproducibility.

### High-level Scientific Environments
- **[Julia](./Julia/)**: Performance-oriented examples for numerical computing and plotting; see an illustrative figure script at [figure.py](./Julia/Example2/figure.py).
- **[MATLAB](./MATLAB/)**: Algorithm prototypes and parallel toolbox snippets for matrix-heavy workloads.
- **[Mathematica](./Mathematica/)**: Symbolic manipulation notebooks and short scripts for analytic checks.

## Quick Reference

| Language / Group | Best For | When to pick | Representative files |
|------------------|---------:|--------------|---------------------|
| **BASH** | Job orchestration, environment setup | Launching sbatch/srun and chaining tools | Scripts in **[BASH](./BASH/)** |
| **C / Cpp** | Low-level performance, system tools | Tight loops, custom memory management | [sum.c](./C/Example/sum.c), **[Cpp](./Cpp/)** examples |
| **Fortran** | Numerical kernels, legacy scientific code | Dense linear algebra and FEM codes | [lanczos.f90](./Fortran/Example1/lanczos.f90), [sum.f90](./Fortran/Example2/sum.f90) |
| **Python** | Data analysis, prototyping, testing | Rapid development, pandas/numpy workflows | [numpy_pandas_ex.py](./Python/Example2/numpy_pandas_ex.py), [test_mc_pi.py](./Python/Example1/test_mc_pi.py) |
| **Julia / MATLAB** | High-level numerical performance | Prototyping with near-C speed or toolbox integration | [figure.py](./Julia/Example2/figure.py), **[MATLAB](./MATLAB/)** |
| **IDL / Perl / Mathematica** | Domain tools & scripting glue | Specialized data formats, quick symbolic checks | **[IDL](./IDL/)**, **[Perl](./Perl/)**, **[Mathematica](./Mathematica/)** |

Backup files: [README.md.backup](./.backup/README.md.backup) | Alternate draft: [README_best.md](./.md/README_best.md)
