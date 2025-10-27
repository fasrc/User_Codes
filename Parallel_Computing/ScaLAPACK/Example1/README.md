### Purpose:

Example Program solving **A x = b** via ScaLAPACK routine PSGESV.

### Contents:

* <code>psgesv.f90</code>: Fortran source code
* <code>Makefile</code>: Makefile to compile this example
* <code>run.sbatch</code>: Batch-job submission script

### Example Batch-Job Submission Script:

```bash
#!/bin/bash
#SBATCH -J scalapack_test
#SBATCH -o scalapack_test.out
#SBATCH -e scalapack_test.err
#SBATCH -p shared
#SBATCH -t 0-00:30
#SBATCH -n 6
#SBATCH --mem-per-cpu=4000

# Load required modules
module load intel/17.0.4-fasrc01
module load openmpi/2.1.0-fasrc02 

# Run program
srun -n 6 --mpi=pmi2 ./psgesv.x
```

### Makefile:

```bash
#==========================================================================
# Makefile
#==========================================================================
F90COMPILER = mpifort
F90CFLAGS   = -c  -i8 -I${MKL_HOME}/include
PRO         = psgesv

MKLROOT = $(MKL_HOME)

LIBS = ${MKLROOT}/lib/intel64/libmkl_scalapack_ilp64.a -Wl,--start-group \
	${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a \
	${MKLROOT}/lib/intel64/libmkl_sequential.a \
	${MKLROOT}/lib/intel64/libmkl_core.a \
	${MKLROOT}/lib/intel64/libmkl_blacs_openmpi_ilp64.a -Wl,--end-group \
	 -lpthread -lm -ldl 

OBJECTS =  $(PRO).o

${PRO}.x : $(OBJECTS)
	$(F90COMPILER) -o ${PRO}.x $(OBJECTS) $(LIBS)

%.o : %.f90
	$(F90COMPILER) $(F90CFLAGS) $(<F)

%.o : %.F90
	$(F90COMPILER) $(F90CFLAGS) $(<F)


clean : 
	rm -rf *.o *.x
```

### Example Usage:

```bash
module load intel/17.0.4-fasrc01
module load openmpi/2.1.0-fasrc02
make
sbatch run.sbatch
```

### Example Output:

```
cat scalapack_test.out

ScaLAPACK Example Program (PSGESV)

MYROW =   0, MYCOL =   2, X =  0.0000E+00  0.0000E+00  0.0000E+00  0.0000E+00  0.0000E+00




Solving Ax=b where A is a   9 by   9 matrix with a block size of   2
Running on   6 processes, where the process grid is   2 by   3

INFO code returned by PSGESV =   0

According to the normalized residual the solution is correct.

||A*x - b|| / ( ||x||*||A||*eps*N ) =   0.00000000E+00

MYROW =   0, MYCOL =   0, X =  7.8427E-10 -1.6667E-01  0.0000E+00 -9.3132E-10  0.0000E+00
MYROW =   0, MYCOL =   1, X =  0.0000E+00  0.0000E+00  0.0000E+00  0.0000E+00  0.0000E+00
MYROW =   1, MYCOL =   2, X =  0.0000E+00  0.0000E+00  0.0000E+00  0.0000E+00  0.0000E+00
MYROW =   1, MYCOL =   1, X =  0.0000E+00  0.0000E+00  0.0000E+00  0.0000E+00  0.0000E+00

MYROW =   1, MYCOL =   0, X =  5.0000E-01  0.0000E+00 -5.0000E-01  1.6667E-01  0.0000E+00
```
