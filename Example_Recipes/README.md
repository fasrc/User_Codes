# Example Recipes - HPC Workflow and Computing Pattern Examples

**Generated on: October 28, 2025 at 21:30 UTC**

## Overview

This directory contains practical workflow examples demonstrating various High-Performance Computing (HPC) patterns, job submission strategies, and scientific computing applications. These recipes serve as templates for common computational tasks encountered in research environments, showcasing best practices for cluster computing, job arrays, parallel processing, and language integration.

## Directory Purpose

The **Example_Recipes** directory provides:
- Real-world HPC workflow templates ready for adaptation
- Practical job submission and resource management examples
- Cross-language integration patterns (Python, R, MATLAB, Fortran, C++)
- Specialized scientific computing workflows
- Job array and parameter sweep demonstrations
- Performance optimization techniques

## Content Organization

### Directory Structure

```
Example_Recipes/
├── Optimization & Mathematical Computing/
│   ├── Recipe1/              # R-based Particle Swarm Optimization
│   ├── Recipe2/              # Alternative PSO implementation
│   ├── Recipe3/              # Python-Fortran integration
│   ├── high_precision/       # 500-digit precision calculations
│   └── matrix_diag/          # Large matrix diagonalization
├── Job Arrays & Parameter Sweeps/
│   ├── Job_Arrays/           # Python and C++ job array examples
│   ├── parallel_job_arrays/  # MATLAB parallel job arrays
│   └── parameters_1/         # Shell script parameter iteration
├── Image & Video Processing/
│   ├── image_processing/     # Python/MATLAB image pipelines
│   ├── image_processing_matlab/ # Video frame analysis
│   └── movie_processing/     # CSV-driven video workflows
├── Language Integration/
│   └── matlab_cpp/           # MATLAB-C++ MEX integration
└── Machine Learning/
    └── pytorch/              # PyTorch GPU computing
```

## Detailed Recipe Index

### Optimization & Mathematical Computing

#### **Recipe1** ([Recipe1/](Recipe1/))
Particle Swarm Optimization using R's `pso` package.
- **Purpose**: Serial optimization algorithms on HPC clusters
- **Language**: R
- **Algorithm**: PSO applied to Rosenbrock function
- **Key Files**: [`pso_rosenbrock.R`](Recipe1/pso_rosenbrock.R), [`run.sbatch`](Recipe1/run.sbatch)
- **Resources**: 1 CPU core, 4GB memory, 30 minutes
- **Use Cases**: Function optimization, parameter tuning, research algorithms

#### **Recipe2** ([Recipe2/](Recipe2/))
Alternative PSO implementation using R's `metaheuristicOpt` package.
- **Purpose**: Comparative PSO implementations
- **Language**: R
- **Algorithm**: Alternative PSO library for same optimization problem
- **Key Files**: [`pso_rosenbrock_v2.R`](Recipe2/pso_rosenbrock_v2.R), [`run.sbatch`](Recipe2/run.sbatch)
- **Comparison**: Performance and accuracy analysis vs Recipe1

#### **Recipe3** ([Recipe3/](Recipe3/))
Python-Fortran integration for computational workflows.
- **Purpose**: Cross-language computational pipelines
- **Languages**: Python, Fortran
- **Workflow**: Compile Fortran → Python subprocess management
- **Key Files**: [`sum.f90`](Recipe3/sum.f90), [`drive_sum.py`](Recipe3/drive_sum.py)
- **Pattern**: Calling compiled executables from scripting languages

#### **High Precision Computing** ([high_precision/](high_precision/))
Ultra-high precision numerical integration using MPFUN-MPFR libraries.
- **Purpose**: 500-digit precision mathematical calculations
- **Language**: Fortran
- **Algorithm**: Quadrature integration with arbitrary precision arithmetic
- **Dependencies**: Intel compiler, MPFR library, MPFUN
- **Key Files**: [`README.md`](high_precision/README.md), multiple Fortran modules, [`Makefile`](high_precision/Makefile)
- **Applications**: Scientific computing requiring extreme precision

#### **Matrix Diagonalization** ([matrix_diag/](matrix_diag/))
Large-scale linear algebra using Intel MKL libraries.
- **Purpose**: Eigenvalue computation for large matrices
- **Language**: Fortran
- **Problem**: 3000×3000 random matrix diagonalization
- **Dependencies**: Intel MKL, Intel compiler
- **Key Files**: [`matrix_diag.f90`](matrix_diag/matrix_diag.f90), [`Makefile`](matrix_diag/Makefile), [`run.sbatch`](matrix_diag/run.sbatch)
- **Performance**: Optimized linear algebra operations

### Job Arrays & Parameter Sweeps

#### **Job Arrays Collection** ([Job_Arrays/](Job_Arrays/))
Comprehensive job array examples with different approaches.

##### **Python-Driven Arrays** ([Job_Arrays/job_arrays_1/](Job_Arrays/job_arrays_1/))
Master script approach for automated job array management.
- **Purpose**: Complex job array orchestration
- **Languages**: Python, C
- **Workflow**: Compile → Create directories → Deploy executables → Submit arrays
- **Key Files**: [`test.py`](Job_Arrays/job_arrays_1/test.py) (master script), [`pro.c`](Job_Arrays/job_arrays_1/pro.c)
- **Features**: Dynamic directory creation, executable deployment
- **Pattern**: Template for large-scale parameter studies

##### **Simple C++ Arrays** ([Job_Arrays/job_arrays_2/](Job_Arrays/job_arrays_2/))
Basic job array demonstration with C++.
- **Purpose**: Fundamental job array concepts
- **Language**: C++
- **Algorithm**: Integer sum calculation across array instances
- **Key Files**: [`main_program.cpp`](Job_Arrays/job_arrays_2/main_program.cpp), [`run.sbatch`](Job_Arrays/job_arrays_2/run.sbatch)
- **Learning**: Job array basics and SLURM integration

#### **MATLAB Parallel Arrays** ([parallel_job_arrays/](parallel_job_arrays/))
Parallel MATLAB processing with job arrays.
- **Purpose**: MATLAB Parallel Computing Toolbox integration
- **Language**: MATLAB
- **Features**: Worker pools, parallel computation
- **Key Files**: [`parallel_sum.m`](parallel_job_arrays/parallel_sum.m), [`README.md`](parallel_job_arrays/README.md)
- **Applications**: MATLAB-based parallel scientific computing

#### **Parameter File Iteration** ([parameters_1/](parameters_1/))
Shell-based parameter sweeps using file iteration.
- **Purpose**: File-driven parameter studies
- **Languages**: Python, Bash
- **Pattern**: Data file iteration with shell automation
- **Key Files**: [`test.py`](parameters_1/test.py), [`test.sh`](parameters_1/test.sh), multiple data files
- **Use Cases**: Systematic parameter exploration

### Image & Video Processing

#### **Image Processing Pipeline** ([image_processing/](image_processing/))
Parallel image enhancement workflows.
- **Purpose**: Batch image processing using job arrays
- **Language**: MATLAB
- **Algorithm**: Contrast improvement and enhancement
- **Structure**: [`images_in/`](image_processing/images_in/) → Processing → [`images_out/`](image_processing/images_out/)
- **Key Files**: [`image_process.m`](image_processing/image_process.m), [`README.md`](image_processing/README.md)
- **Scaling**: Job arrays for parallel image processing

#### **Video Frame Analysis** ([image_processing_matlab/](image_processing_matlab/))
Video file processing and frame extraction.
- **Purpose**: Video analysis and frame counting
- **Language**: MATLAB
- **Features**: Video codec handling, frame statistics
- **Key Files**: [`video_test.m`](image_processing_matlab/video_test.m), [`README.md`](image_processing_matlab/README.md)
- **Applications**: Video content analysis, preprocessing

#### **Automated Video Workflows** ([movie_processing/](movie_processing/))
CSV-driven batch video processing.
- **Purpose**: Metadata-driven video workflow automation
- **Languages**: Bash, FFmpeg
- **Features**: CSV parsing, automated FFmpeg operations, timecode handling
- **Key Files**: [`movie_info.csv`](movie_processing/movie_info.csv), [`process_movies.sh`](movie_processing/process_movies.sh)
- **Workflow**: CSV metadata → Automated video processing

### Language Integration

#### **MATLAB-C++ Integration** ([matlab_cpp/](matlab_cpp/))
MEX file integration for calling GSL libraries from MATLAB.
- **Purpose**: High-performance numerical libraries in MATLAB
- **Languages**: MATLAB, C++
- **Libraries**: GNU Scientific Library (GSL)
- **Algorithm**: Bessel function calculations
- **Key Files**: [`bessel_test.cpp`](matlab_cpp/bessel_test.cpp), [`mex_test.m`](matlab_cpp/mex_test.m), [`Makefile`](matlab_cpp/Makefile)
- **Pattern**: Extending MATLAB with compiled libraries

### Machine Learning & GPU Computing

#### **PyTorch GPU Setup** ([pytorch/](pytorch/))
Deep learning framework deployment on HPC clusters.
- **Purpose**: GPU-accelerated deep learning workflows
- **Language**: Python
- **Framework**: PyTorch with CUDA support
- **Dependencies**: CUDA, cuDNN, Anaconda
- **Key Files**: [`README.md`](pytorch/README.md), installation scripts
- **Resources**: GPU nodes, specific CUDA versions
- **Applications**: Neural network training, research workflows

## Common HPC Patterns Demonstrated

### Resource Allocation Strategies

#### Standard Job Submission
```bash
#SBATCH -n 1                    # Single CPU core
#SBATCH -t 30                   # 30 minutes runtime
#SBATCH --mem=4000             # 4GB memory
#SBATCH -p shared              # Shared partition
```

#### GPU Resource Allocation
```bash
#SBATCH -p gpu                 # GPU partition
#SBATCH --gres=gpu:1           # Single GPU
#SBATCH --mem=8000             # Higher memory for GPU jobs
```

#### Job Array Syntax
```bash
#SBATCH --array=1-10           # Array indices 1-10
#SBATCH --array=1-100%5        # Max 5 concurrent jobs
```

### Module Loading Patterns

#### Scientific Computing Stack
```bash
module load gcc/13.2.0-fasrc01
module load intel/24.0.1-fasrc01
module load R/4.3.1-fasrc01
```

#### GPU Computing Environment
```bash
module load python/3.10.12-fasrc01
module load cuda/12.2.0-fasrc01
module load cudnn/8.8.0.121_cuda12-fasrc01
```

### Compilation and Build Patterns

#### Fortran with Intel MKL
```makefile
FC = ifort
FFLAGS = -O3 -mkl
LDFLAGS = -mkl
```

#### C++ with GSL
```makefile
CXX = g++
CXXFLAGS = -fPIC -O3
LIBS = -lgsl -lgslcblas
```

#### MEX File Compilation
```bash
mex -setup C++
mex bessel_test.cpp -lgsl -lgslcblas
```

## Software Dependencies

### Programming Languages
- **R**: Optimization packages (`pso`, `metaheuristicOpt`)
- **Python**: Scientific stack (NumPy, SciPy), PyTorch
- **MATLAB**: Parallel Computing Toolbox, Image Processing Toolbox
- **Fortran**: High-precision libraries (MPFUN, MPFR)
- **C/C++**: GNU Scientific Library (GSL)

### Compilers and Tools
- **GCC**: GNU Compiler Collection
- **Intel compilers**: ifort, icc with MKL optimization
- **FFmpeg**: Video processing and codec handling

### HPC Software
- **SLURM**: Job scheduling and resource management
- **Module system**: Environment management
- **Conda/Anaconda**: Python environment management

## Performance Considerations

### Memory Requirements
- **Light workflows**: 4GB (optimization, simple arrays)
- **Image processing**: 8-16GB (depending on image sizes)
- **Matrix operations**: 16GB+ (large matrices with MKL)
- **Video processing**: Variable (depends on video resolution/length)

### Compute Requirements
- **Serial examples**: Single CPU core sufficient
- **Parallel arrays**: Multiple cores via job arrays
- **GPU workloads**: Dedicated GPU nodes with CUDA support
- **High-precision**: CPU-intensive, may benefit from faster processors

### Scaling Strategies
- **Embarrassingly parallel**: Job arrays for independent tasks
- **Parameter sweeps**: Automated job submission with varying inputs
- **Pipeline processing**: Sequential workflow automation
- **Hybrid approaches**: Combining serial and parallel components

## Usage Guidelines

### Getting Started
1. **Choose appropriate example** based on your computational needs
2. **Review dependencies** and load required modules
3. **Modify parameters** in configuration files
4. **Test with small datasets** before scaling up
5. **Monitor resource usage** and adjust allocations

### Customization Tips
- **Resource tuning**: Adjust memory and time limits based on data size
- **Parameter modification**: Edit input files and configuration scripts
- **Path updates**: Modify file paths for your directory structure
- **Library versions**: Ensure compatibility with available modules

### Best Practices
- **Version control**: Track modifications to example scripts
- **Documentation**: Comment changes and customizations
- **Testing**: Validate output correctness before production runs
- **Monitoring**: Use SLURM tools to track job performance

## Contributing Guidelines

When adapting or extending these examples:

1. **Maintain structure**: Keep the established directory organization
2. **Document changes**: Add comments explaining modifications
3. **Version tracking**: Note software versions and test dates
4. **Resource documentation**: Document memory and time requirements
5. **Error handling**: Include appropriate error checking and logging

## Related Resources

- [Main User_Codes Repository](../)
- [Notes Directory - Software Installation Guides](../Notes/)
- [FASRC User Documentation](https://docs.rc.fas.harvard.edu/)
- [SLURM Job Arrays Guide](https://slurm.schedmd.com/job_array.html)
- [MATLAB Parallel Computing](https://www.mathworks.com/products/parallel-computing.html)

## Support

For questions about these workflow examples:
1. Check the specific example's README file
2. Review FASRC documentation for SLURM and cluster usage
3. Consult software-specific documentation for dependencies
4. Submit support tickets for cluster-specific issues

---

*This directory provides practical templates for common HPC workflows. Each example represents tested procedures that can be adapted for research and computational projects.*