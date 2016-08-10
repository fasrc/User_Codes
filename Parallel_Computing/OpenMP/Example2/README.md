### Purpose:

Program generates a random matrix and calculates its eigen values. Uses OpenMP to
generate the matrix and threaded version of MKL diagonalize it.

### Contents:

(1) omp_diag.f90: Fortran source code

(2) Makefile: Makefile to compile the source code

(3) run.sbatch: Btach-job submission script to send the job to the queue.

### Example Usage:

	source new-modules.sh
	module load intel/15.0.0-fasrc01
	module load intel-mkl/11.0.0.079-fasrc02
	make
	sbatch run.sbatch
    
### Example Output:

```
 Number of threads:          4
 Eigen values of h:
           1  -18.1903779141984
           2  -17.9346399306460
           3  -17.7773008896094
           4  -17.6638127619466
           5  -17.6579617445697
           6  -17.5805514443568
           7  -17.4332661135333
           8  -17.3203395624164
           9  -17.2903465087959
          10  -17.2437918174720
```
