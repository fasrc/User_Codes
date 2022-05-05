#!/bin/bash

export PATH=$OMPI_DIR/bin:$PATH
export LD_LIBRARY_PATH=$OMPI_DIR/lib:$LD_LIBRARY_PATH

# compile fortran program
mpif90 -o mpitest.x mpitest.f90 -O2

# compile c-program
mpicc -o mpitest.exe mpitest.c

