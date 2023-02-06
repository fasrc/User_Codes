Parallel HDF5 (PHDF5) is the parallel version of the HDF5 library. It utilizes MPI to perform parallel HDF5 operations. For example, when an HDF5 file is opened with an MPI communicator, all the processes within the communicator can perform various operations on the file. PHDF5 supports file operations such as file create, open and close, as well as dataset operations such as object creation, modification and querying, all in parallel using MPI-IO. The below examples are intended to illustrate the use of PHDF5 on the Cannon cluster. The specific examples are implemented in Fortran, but the could be easily translated to C or C++, for instance:

* [Example 1](Example1/): Parallel HDF5 - write 1D array of dimension 100
* [Example 2](Example2/): Parallel HDF5 - write 2D array of dimension 20x30 
* [Example 3](Example3/): Parallel HDF5 - write 3D array of dimension 10x30x8
