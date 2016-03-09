### Purpose:

Program computes PI with parallel Monte-Carlo mathod.

### Contents:

(1) parallel_pi.m: C++ source code

(2) run.sbatch: Btach-job submission script to send the job to the queue.

### Example Usage:

	source new-modules.sh
	module load mathematica/10.3.0-fasrc01
	sbatch run.sbatch
    
### Example Output:

```
"Parallel calculation of PI via Monte-Carlo method."
" Number of kernels: "8
" Total number of hits: "100000000
" Number of hits per core: "12500000
" Computed PI = "3.14142016
" Time in parallel calculation: "3.018016`6.931266531352143
" Total time: "9.909157`7.447581702893375
```
