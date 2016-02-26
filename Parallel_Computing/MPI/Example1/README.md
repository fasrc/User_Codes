### Purpose:

Program calculates PI via parallel Monte-Carlo algorithm.

### Contents:

(1) pi_monte_carlo.f90: Fortran source code

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
 Exact PI:    3.14159265
 Computed PI: 3.14156124
 Error:       0.00100%
 Total time: 14.72 sec
```
