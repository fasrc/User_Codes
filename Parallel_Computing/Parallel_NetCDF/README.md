# Parallel NetCDF Examples

This collection provides practical examples for utilizing Parallel NetCDF for efficient parallel I/O operations in scientific computing.

For general information about parallel computing on Harvard's cluster, see the [RC User Documentation on Parallel Computing](https://docs.rc.fas.harvard.edu/kb/parallel-computing/).

## Parallel NetCDF Use Cases

### Basic Parallel I/O Operations
Fundamental examples demonstrating standard parallel file operations.

- **[Example1](Example1/)**: Basic usage of Parallel NetCDF with SBATCH job submission for data writing and reading.  
- **[Example2](Example2/)**: Advanced example with optimized I/O patterns and performance metrics.  
- **[Example3](Example3/)**: Comprehensive demonstration including error handling and data integrity checks.

### Advanced Parallel I/O Techniques
In-depth examples focusing on complex data handling scenarios.

- **[Example2](Example2/)**: Features a performance analysis of parallel writes and reads, showcasing scalability.  
- **[Example3](Example3/)**: Illustrates multi-dimensional data management with Parallel NetCDF, emphasizing efficiency.

## Code Files Overview

### Example Scripts
- **[README.md](./README.md)**: Overview and instructions for using Parallel NetCDF examples.
- **[simple_xy_par_wr.f90](./Example2/simple_xy_par_wr.f90)**: Fortran program for parallel writing of 2D arrays, adapted from the official NetCDF package.
- **[simple_xy_par_rd.f90](./Example3/simple_xy_par_rd.f90)**: Fortran program for reading 2D arrays in parallel, ensuring data consistency.
- **[Example1 README.md](./Example1/README.md)**: Documentation for the first example, detailing usage and expected outputs.
- **[Example2 README.md](./Example2/README.md)**: Documentation for the second example, including performance benchmarks.
- **[Example3 README.md](./Example3/README.md)**: Documentation for the third example, focusing on advanced features.

## Quick Reference

| Example | Description | Language | Key Features |
|---------|-------------|----------|--------------|
| **[Example1](Example1/)** | Basic parallel I/O operations | Fortran | Simple read/write, SBATCH integration |
| **[Example2](Example2/)**: | Optimized I/O with performance metrics | Fortran | Scalability analysis, advanced patterns |
| **[Example3](Example3/)** | Multi-dimensional data handling | Fortran | Error handling, data integrity checks |