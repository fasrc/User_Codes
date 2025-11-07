# Example Recipes - Directory Index

**Generated on: October 28, 2025 at 21:45 UTC**

## Overview

This directory is a collection of computational workflow examples, templates, and use cases that have been contributed over time. Each subdirectory contains working examples of specific computational tasks, primarily focused on HPC cluster usage with SLURM job submission.

## Directory Contents

### Basic Optimization Examples

#### [Recipe1/](Recipe1/) - Particle Swarm Optimization (R/pso)
Rosenbrock function optimization using R's `pso` package.
- [`pso_rosenbrock.R`](Recipe1/pso_rosenbrock.R) - Main R script
- [`run.sbatch`](Recipe1/run.sbatch) - SLURM job submission script
- [`README.md`](Recipe1/README.md) - Implementation details

#### [Recipe2/](Recipe2/) - Particle Swarm Optimization (R/metaheuristicOpt)
Alternative PSO implementation for comparison.
- [`pso_rosenbrock_v2.R`](Recipe2/pso_rosenbrock_v2.R) - R script using different package
- [`run.sbatch`](Recipe2/run.sbatch) - SLURM submission script

#### [Recipe3/](Recipe3/) - Python-Fortran Integration
Example of calling compiled Fortran from Python.
- [`sum.f90`](Recipe3/sum.f90) - Fortran source code
- [`drive_sum.py`](Recipe3/drive_sum.py) - Python driver script
- [`run.sbatch`](Recipe3/run.sbatch) - Job submission script
- [`README.md`](Recipe3/README.md) - Compilation and usage instructions

### High-Precision Computing

#### [high_precision/](high_precision/) - MPFUN Multi-Precision Library
500-digit precision numerical integration using MPFUN-MPFR.
- [`mpint_test_v1.f90`](high_precision/mpint_test_v1.f90) - Test program version 1
- [`mpint_test_v2.f90`](high_precision/mpint_test_v2.f90) - Test program version 2
- [`mpfuna.f90`](high_precision/mpfuna.f90) - MPFUN arithmetic module
- [`mpfunf.f90`](high_precision/mpfunf.f90) - MPFUN functions
- [`mpfung1.f90`](high_precision/mpfung1.f90) - MPFUN generic module
- [`mpmodule.f90`](high_precision/mpmodule.f90) - Main MPFUN module
- [`mpint_lib.f90`](high_precision/mpint_lib.f90) - Integration library
- [`mpinterface.c`](high_precision/mpinterface.c) - C interface
- [`secondu.f90`](high_precision/secondu.f90) - Timing utilities
- [`Makefile`](high_precision/Makefile) - Build configuration
- [`run.sbatch`](high_precision/run.sbatch) - SLURM script
- [`README.md`](high_precision/README.md) - Detailed setup and usage

### Linear Algebra

#### [matrix_diag/](matrix_diag/) - Large Matrix Diagonalization
Eigenvalue computation using Intel MKL.
- [`matrix_diag.f90`](matrix_diag/matrix_diag.f90) - Fortran program for 3000x3000 matrix
- [`Makefile`](matrix_diag/Makefile) - Build with Intel MKL
- [`run.sbatch`](matrix_diag/run.sbatch) - SLURM submission
- [`README.md`](matrix_diag/README.md) - Usage instructions

### Job Arrays and Parameter Studies

#### [Job_Arrays/](Job_Arrays/) - Job Array Examples

##### [job_arrays_1/](Job_Arrays/job_arrays_1/) - Python-Managed Arrays
Master script approach for complex job array workflows.
- [`test.py`](Job_Arrays/job_arrays_1/test.py) - Master orchestration script
- [`pro.c`](Job_Arrays/job_arrays_1/pro.c) - C program for array tasks
- [`README.md`](Job_Arrays/job_arrays_1/README.md) - Workflow description

##### [job_arrays_2/](Job_Arrays/job_arrays_2/) - Simple C++ Arrays
Basic job array demonstration.
- [`main_program.cpp`](Job_Arrays/job_arrays_2/main_program.cpp) - C++ array program
- [`run.sbatch`](Job_Arrays/job_arrays_2/run.sbatch) - Array job script
- [`Makefile`](Job_Arrays/job_arrays_2/Makefile) - Build configuration
- [`README.md`](Job_Arrays/job_arrays_2/README.md) - Basic usage

#### [parallel_job_arrays/](parallel_job_arrays/) - MATLAB Parallel Arrays
MATLAB Parallel Computing Toolbox with job arrays.
- [`parallel_sum.m`](parallel_job_arrays/parallel_sum.m) - MATLAB parallel script
- [`run.sbatch`](parallel_job_arrays/run.sbatch) - SLURM array submission
- [`README.md`](parallel_job_arrays/README.md) - MATLAB parallel setup

#### [parameters_1/](parameters_1/) - Parameter File Iteration
File-based parameter sweeps.
- [`test.py`](parameters_1/test.py) - Python parameter processor
- [`test.sh`](parameters_1/test.sh) - Shell script wrapper
- [`ran_array.py`](parameters_1/ran_array.py) - Random array generator
- [`run.sbatch`](parameters_1/run.sbatch) - SLURM submission
- [`README.md`](parameters_1/README.md) - Parameter sweep setup
- Data files: [`data1.dat`](parameters_1/data1.dat), [`data2.dat`](parameters_1/data2.dat), [`data3.dat`](parameters_1/data3.dat), [`data4.dat`](parameters_1/data4.dat), [`data5.dat`](parameters_1/data5.dat)

### Image and Video Processing

#### [image_processing/](image_processing/) - MATLAB Image Processing
Batch image enhancement using job arrays.
- [`image_process.m`](image_processing/image_process.m) - MATLAB image processing script
- [`run.sbatch`](image_processing/run.sbatch) - Array job for parallel processing
- [`README.md`](image_processing/README.md) - Setup and usage
- Input/output directories: [`images_in/`](image_processing/images_in/), [`images_out/`](image_processing/images_out/)

#### [image_processing_matlab/](image_processing_matlab/) - Video Analysis
MATLAB video file processing and frame analysis.
- [`video_test.m`](image_processing_matlab/video_test.m) - Video processing script
- [`run.sbatch`](image_processing_matlab/run.sbatch) - SLURM submission
- [`README.md`](image_processing_matlab/README.md) - Video processing setup
- Sample videos: [`test_vid.mp4`](image_processing_matlab/test_vid.mp4), [`test_vid2.mov`](image_processing_matlab/test_vid2.mov)

#### [movie_processing/](movie_processing/) - Automated Video Workflows
CSV-driven batch video processing with FFmpeg.
- [`process_movies.sh`](movie_processing/process_movies.sh) - Bash automation script
- [`movie_info.csv`](movie_processing/movie_info.csv) - Metadata input file
- [`README.md`](movie_processing/README.md) - Workflow documentation

### Language Integration

#### [matlab_cpp/](matlab_cpp/) - MATLAB MEX Integration
Calling C++ libraries from MATLAB using MEX files.
- [`bessel_test.cpp`](matlab_cpp/bessel_test.cpp) - C++ MEX function using GSL
- [`mex_test.m`](matlab_cpp/mex_test.m) - MATLAB test script
- [`mex_test.sbatch`](matlab_cpp/mex_test.sbatch) - SLURM job script
- [`Makefile`](matlab_cpp/Makefile) - MEX compilation
- [`README.md`](matlab_cpp/README.md) - MEX setup and GSL integration

### Machine Learning

#### [pytorch/](pytorch/) - PyTorch GPU Setup
Deep learning framework installation and configuration.
- [`README.md`](pytorch/README.md) - Installation instructions and GPU setup

### Additional Resources

#### [Images/](Images/) - Documentation Assets
- [`recipes-logo.png`](Images/recipes-logo.png) - Logo image

## Quick Reference by Language

### R Examples
- [Recipe1/](Recipe1/) - PSO with `pso` package
- [Recipe2/](Recipe2/) - PSO with `metaheuristicOpt` package

### Python Examples
- [Recipe3/](Recipe3/) - Python-Fortran integration
- [Job_Arrays/job_arrays_1/](Job_Arrays/job_arrays_1/) - Array orchestration
- [parameters_1/](parameters_1/) - Parameter sweeps

### Fortran Examples
- [Recipe3/](Recipe3/) - Called from Python
- [high_precision/](high_precision/) - Multi-precision arithmetic
- [matrix_diag/](matrix_diag/) - Linear algebra with MKL

### MATLAB Examples
- [parallel_job_arrays/](parallel_job_arrays/) - Parallel computing
- [image_processing/](image_processing/) - Image enhancement
- [image_processing_matlab/](image_processing_matlab/) - Video analysis
- [matlab_cpp/](matlab_cpp/) - MEX file integration

### C/C++ Examples
- [Job_Arrays/job_arrays_1/](Job_Arrays/job_arrays_1/) - Simple C program
- [Job_Arrays/job_arrays_2/](Job_Arrays/job_arrays_2/) - C++ job arrays
- [matlab_cpp/](matlab_cpp/) - MEX with GSL libraries

### Shell/Bash Examples
- [parameters_1/](parameters_1/) - Parameter iteration
- [movie_processing/](movie_processing/) - Video processing automation

## Usage Notes

- Most examples include SLURM submission scripts (`.sbatch` files)
- Many directories contain individual README files with specific setup instructions
- Examples demonstrate various resource allocation patterns
- Build files (Makefiles) are included where compilation is required
- Sample data files are provided for testing workflows

## Getting Started

1. Choose an example that matches your computational needs
2. Read the specific README in that directory
3. Review the SLURM submission script for resource requirements
4. Modify parameters as needed for your use case
5. Submit jobs following the documented patterns

---

*This directory serves as a repository of working computational examples. Each subdirectory represents a self-contained workflow that can be adapted for similar use cases.*