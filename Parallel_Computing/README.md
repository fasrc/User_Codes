# Parallel Computing Examples

Examples of how to submit parallel jobs for different languages and applications.

## Core Parallel Computing Paradigms

### Distributed Memory Computing
Parallel computing across multiple nodes using message passing.

- **[MPI - Message Passing Interface](MPI/)**: Multi-language MPI examples (Fortran, C, C++) with comprehensive compilation guides
- **[MPI_IO](MPI_IO/)**: Parallel file I/O operations using MPI

### Shared Memory Computing  
Parallel computing within a single node using shared memory.

- **[OpenMP](OpenMP/)**: Thread-based parallelization examples with performance analysis
- **[EP - Embarrassingly Parallel](EP/)**: Job arrays and independent parallel tasks

### High-Performance I/O Libraries
Specialized libraries for parallel data access and scientific computing.

- **[Parallel_HDF5](Parallel_HDF5/)**: Hierarchical Data Format parallel I/O (1D, 2D, 3D examples)  
- **[Parallel_NetCDF](Parallel_NetCDF/)**: Network Common Data Form parallel operations
- **[PnetCDF](PnetCDF/)**: High-performance parallel NetCDF implementation
- **[ScaLAPACK](ScaLAPACK/)**: Scalable Linear Algebra Package for distributed computing

## Language Ecosystems

### Scientific Computing Languages

#### **[Python](Python/)**
- **[mpi4py](Python/mpi4py/)**: Python MPI wrapper for distributed computing
- **[Multiprocessing Tutorial](Python/Python-Multiprocessing-Tutorial/)**: Shared memory parallelization

#### **[R](R/)**  
- **[future](R/future/)**: Modern R parallel programming framework
- **[Rmpi](R/Rmpi/)**: R interface to MPI
- **[pbdMPI](R/pbdMPI/)**: Programming with Big Data using MPI
- **[Large Data Processing](R/Large_Data_Processing_R/)**: Comprehensive data analysis workflows

### Technical Computing Platforms

- **[MATLAB](MATLAB/)**: Parallel toolbox examples with Monte Carlo simulations
- **[Mathematica](Mathematica/)**: Parallel kernel computations  
- **[STATA](STATA/)**: Statistical analysis parallelization

## Quick Reference

| Paradigm | Best For | Languages | Examples |
|----------|----------|-----------|----------|
| **MPI** | Multi-node, large scale | Fortran, C, C++, Python | Monte Carlo, Integration, Matrix ops |
| **OpenMP** | Single-node, shared memory | C, Fortran | Thread scaling, Matrix diagonalization |
| **Job Arrays** | Independent tasks | Any | Parameter sweeps, Multiple simulations |
| **HDF5/NetCDF** | Large datasets | Fortran, C | Climate data, Scientific datasets |

## Getting Started

1. **Choose your parallel approach** based on problem characteristics
2. **Load appropriate modules** for your compiler and libraries  
3. **Follow directory-specific README** for detailed instructions
4. **Use provided SLURM scripts** as job submission templates

Each directory contains complete working examples with compilation instructions, batch scripts, and sample outputs.
