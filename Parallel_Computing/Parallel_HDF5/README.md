# Parallel_HDF5 Examples

Explore practical implementations for high-performance parallel I/O using HDF5 across multiple dimensions.

For general information about parallel computing on Harvard's cluster, see the [RC User Documentation on Parallel Computing](https://docs.rc.fas.harvard.edu/kb/parallel-computing/).

## Parallel I/O Examples

### 1D Parallel HDF5
Examples demonstrating parallel data access for one-dimensional datasets.

- **[Example1](Example1/)**: 
  - Program: [parallel_hdf5.f90](./Example1/parallel_hdf5.f90)  
    Generates a random vector, distributes it to multiple processes for parallel writing.
  - Makefile: [Makefile](./Example1/Makefile)  
    Build instructions for compiling the 1D example.

### 2D Parallel HDF5
Showcasing parallel I/O operations for two-dimensional datasets.

- **[Example2](Example2/)**: 
  - Program: [parallel_hdf5_2d.f90](./Example2/parallel_hdf5_2d.f90)  
    Constructs a random 2D array split across processes for efficient storage.
  - Makefile: [Makefile](./Example2/Makefile)  
    Compilation instructions for the 2D example.

### 3D Parallel HDF5
Implementations for handling three-dimensional datasets in parallel.

- **[Example3](Example3/)**: 
  - Program: [parallel_hdf5_3d.f90](./Example3/parallel_hdf5_3d.f90)  
    Generates and writes a random 3D array, optimizing I/O performance across nodes.
  - Makefile: [Makefile](./Example3/Makefile)  
    Instructions to compile the 3D example.

## Quick Reference

| Dimension | Description | Files | Use Cases |
|-----------|-------------|-------|-----------|
| **1D**    | Random vector distribution and storage | [parallel_hdf5.f90](./Example1/parallel_hdf5.f90) | Time series data, simple datasets |
| **2D**    | Random 2D array splitting across processes | [parallel_hdf5_2d.f90](./Example2/parallel_hdf5_2d.f90) | Image processing, matrix operations |
| **3D**    | Random 3D array generation and writing | [parallel_hdf5_3d.f90](./Example3/parallel_hdf5_3d.f90) | Volume data, scientific simulations |