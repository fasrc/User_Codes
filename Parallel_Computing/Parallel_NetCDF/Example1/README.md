### Purpose:

This program tests netCDF-4 parallel I/O functions from Fortran.  We are writing 2D data, a 6 x 12 grid, on 4 processors. Each processor will write it's rank to it's quarter of the array.  

### Contents:

* <code>f90tst_parallel.f90</code>: Fortran source code
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
module load netcdf/4.1.3-fasrc10

# Run program
srun -n 8 --mpi=pmi2 ./f90tst_parallel.x
```

### Example Usage:

```bash
source new-modules.sh
module load intel/17.0.2-fasrc01
module load impi/2017.2.174-fasrc01
module load netcdf/4.1.3-fasrc10
make
sbatch run.sbatch
```

### Example Output:

```
ncdump f90tst_parallel.nc
netcdf f90tst_parallel {
dimensions:
        x = 16 ;
        y = 16 ;
variables:
        int data(x, y) ;
data:

 data =
  0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
  0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
  0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
  0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
  0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
  0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
  0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
  0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
  2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
  2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
  2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
  2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
  2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
  2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
  2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
  2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3 ;
}

```
