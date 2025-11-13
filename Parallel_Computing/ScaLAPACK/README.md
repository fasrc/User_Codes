# ScaLAPACK Examples

Explore efficient parallel linear algebra operations using ScaLAPACK with provided examples, compilation instructions, and batch scripts.

For general information about parallel computing on Harvard's cluster, see the [RC User Documentation on Parallel Computing](https://docs.rc.fas.harvard.edu/kb/parallel-computing/).

## ScaLAPACK Overview

### Scalable Linear Algebra Package
Utilize ScaLAPACK for distributed memory parallel computations on large matrices.

- **[Example1](Example1/)**: Demonstrates the use of ScaLAPACK for solving linear systems with detailed batch scripts and output analysis.

## Code Examples

### Example1
This directory contains a complete working example showcasing the capabilities of ScaLAPACK.

- **[README.md](./Example1/README.md)**: Overview of the example, including job submission instructions.
- **[psgesv.f90](./Example1/psgesv.f90)**: Fortran implementation for solving linear systems using ScaLAPACK.
- **[Makefile](./Example1/Makefile)**: Compilation instructions for building the Fortran code.
- **[scalapack_test.sh](./Example1/scalapack_test.sh)**: Batch script for submitting the job to SLURM.

## Quick Reference

| Feature | Description | Use Case | Example File |
|---------|-------------|----------|--------------|
| **ScaLAPACK** | Distributed linear algebra computations | Large matrix operations | [psgesv.f90](./Example1/psgesv.f90) |
| **Batch Script** | Automates job submission | Efficient job management | [scalapack_test.sh](./Example1/scalapack_test.sh) |
| **Makefile** | Simplifies compilation | Easy code building | [Makefile](./Example1/Makefile) |
| **Documentation** | Guides and instructions | Understanding usage | [README.md](./Example1/README.md) |