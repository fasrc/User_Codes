### Purpose:

Example of using OpenBLAS libraries on the cluster.

### Contents:

(1) openblas\_test.f90\_test.f90: Fortran source code

(2) Makefile: Makefile to compile the source code

(3) run.sbatch: Btach-job submission script to send the job to the queue

### Example Usage:

    module load gcc/9.2.0-fasrc01
    module load OpenBLAS/0.3.7-fasrc02
    make 
    sbatch run.sbatch
    
### Example Output:

```
  info, ipiv =            0           3           3           3
  info, b =            0  0.9999995       1.000000      0.9999999    
End of program.
```
