### Purpose:

Example of using OpenBLAS libraries on the cluster.

### Contents:

(1) openblas_test.f90_test.f90: Fortran source code

(2) Makefile: Makefile to compile the source code

(3) run.sbatch: Btach-job submission script to send the job to the queue

### Example Usage:

    source new-modules.sh
    module load intel/15.0.0-fasrc01
    module load OpenBLAS/0.2.14-fasrc01
    make 
    sbatch run.sbatch
    
### Example Output:

```
  info, ipiv =            0           3           3           3
  info, b =            0  0.9999995       1.000000      0.9999999    
End of program.
```
