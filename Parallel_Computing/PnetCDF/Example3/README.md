### Purpose:

This program tests PnetCDF parallel I/O functions from Fortran. This example makes a number of nonblocking API calls, each to write a block of columns into a 2D integer variable in a file. In other words, data partitioning pattern is a block-cyclic along X dimension. The pattern is described by the rank IDs if run with 4 processes.

### Contents:

* <code>block\_cyclic.f90</code>: Fortran source code
* <code>utils.F90</code>: Fortran source code
* <code>Makefile</code>: Makefile to compile this example
* <code>run.sbatch</code>: Batch-job submission script

### Example Batch-Job Submission Script:

```bash
#!/bin/bash
#SBATCH -J parallel_netcdf
#SBATCH -o parallel_netcdf.out
#SBATCH -e parallel_netcdf.err
#SBATCH -p general
#SBATCH -t 0-00:30
#SBATCH -n 4
#SBATCH --mem-per-cpu=4000

# Load required modules
source new-modules.sh
module load intel/17.0.2-fasrc01
module load impi/2017.2.174-fasrc01
module load parallel-netcdf/1.8.1-fasrc01

# Run program
srun -n 4 --mpi=pmi2 ./block_cyclic.x
```

### Makefile:

```bash
#==========================================================================
# Makefile
#==========================================================================
F90COMPILER = mpiifort
F90CFLAGS   = -c -O2
LIBS        = -lpnetcdf
PRO         = block_cyclic

OBJECTS =  utils.o     \
           block_cyclic.o

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
source new-modules.sh
module load intel/17.0.2-fasrc01
module load impi/2017.2.174-fasrc01
module load parallel-netcdf/1.8.1-fasrc01
make
sbatch run.sbatch
```

### Example Output:

```
ncdump testfile.nc 
netcdf testfile {
dimensions:
	Y = 10 ;
	X = 16 ;
variables:
	int var(Y, X) ;
data:

 var =
  10, 10, 11, 11, 12, 12, 13, 13, 10, 10, 11, 11, 12, 12, 13, 13,
  10, 10, 11, 11, 12, 12, 13, 13, 10, 10, 11, 11, 12, 12, 13, 13,
  10, 10, 11, 11, 12, 12, 13, 13, 10, 10, 11, 11, 12, 12, 13, 13,
  10, 10, 11, 11, 12, 12, 13, 13, 10, 10, 11, 11, 12, 12, 13, 13,
  10, 10, 11, 11, 12, 12, 13, 13, 10, 10, 11, 11, 12, 12, 13, 13,
  10, 10, 11, 11, 12, 12, 13, 13, 10, 10, 11, 11, 12, 12, 13, 13,
  10, 10, 11, 11, 12, 12, 13, 13, 10, 10, 11, 11, 12, 12, 13, 13,
  10, 10, 11, 11, 12, 12, 13, 13, 10, 10, 11, 11, 12, 12, 13, 13,
  10, 10, 11, 11, 12, 12, 13, 13, 10, 10, 11, 11, 12, 12, 13, 13,
  10, 10, 11, 11, 12, 12, 13, 13, 10, 10, 11, 11, 12, 12, 13, 13 ;
}
```
