### Purpose:

This program tests netCDF-4 parallel I/O functions from Fortran 90. The specific example uses 10 MPI tasks to create 10x10 matrix. Each process writes a row of the matrix to a common NetCDF file.  

### Contents:

* <code>simple\_xy\_par_rd.f90</code>: Fortran source code
* <code>Makefile</code>: Makefile to compile this example
* <code>simple\_xy\_par.nc</code>: Example NetCDF file (generated in **Example2**)
* <code>run.sbatch</code>: Batch-job submission script

### Example NetCDF file:

```
ncdump simple_xy_par.nc
netcdf simple_xy_par {
dimensions:
	x = 10 ;
	y = 10 ;
variables:
	int data(x, y) ;
data:

 data =
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
  3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
  4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
  5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
  6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
  7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
  8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
  9, 9, 9, 9, 9, 9, 9, 9, 9, 9 ;
}
```

### Example Batch-Job Submission Script:

```bash
#!/bin/bash
#SBATCH -J parallel_netcdf
#SBATCH -o parallel_netcdf.out
#SBATCH -e parallel_netcdf.err
#SBATCH -p general
#SBATCH -t 0-00:30
#SBATCH -n 10
#SBATCH --mem-per-cpu=4000

# Load required modules
source new-modules.sh
module load intel/17.0.2-fasrc01
module load impi/2017.2.174-fasrc01
module load netcdf-fortran/4.4.4-fasrc03

# Run program
srun -n 10 --mpi=pmi2 ./simple_xy_par_rd.x
```

### Example Usage:

```bash
source new-modules.sh
module load intel/17.0.2-fasrc01
module load impi/2017.2.174-fasrc01
module load netcdf-fortran/4.4.4-fasrc03
make
sbatch run.sbatch
```

### Example Output:

```
 *** SUCCESS reading example file simple_xy_par.nc! 
```
