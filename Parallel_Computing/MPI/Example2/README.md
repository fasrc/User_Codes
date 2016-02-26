### Purpose:

Prallel implementation of the trapezoidal rule for integration. Uses "cyclic" distribution of loop iterations.
Currently set up to compute integral \int_0^4 x**2 with 80 integration points.

### Contents:

(1) ptrap.f90: Fortran source code

(2) Makefile: Makefile to compile the source code

(3) run.sbatch: Btach-job submission script to send the job to the queue.

### Example Usage:

	source new-modules.sh
	module load intel/15.0.0-fasrc01
	module load openmpi/1.8.3-fasrc02
	make
	sbatch run.sbatch
    
### Example Output:

```
Integral from  0.0  to  4.0  is  21.3350
```
