### Purpose:

This program tests PnetCDF parallel I/O functions from Fortran. The specific example makes a number of nonblocking API calls, each writes a single row of a 2D integer array. Each process writes NY rows and any two consecutive rows are of nprocs-row distance apart from each other. In this case, the fileview of each process interleaves with all other processes.

### Contents:

* <code>column\_wise.f90</code>: Fortran source code
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
srun -n 4 --mpi=pmi2 ./column_wise.x
```

### Makefile:

```bash
#==========================================================================
# Makefile
#==========================================================================
F90COMPILER = mpiifort
F90CFLAGS   = -c -O2
LIBS        = -lpnetcdf
PRO         = column_wise

OBJECTS =  utils.o     \
           column_wise.o

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
  0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
  0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
  0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
  0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
  0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
  0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
  0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
  0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
  0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
  0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3 ;
}
```
