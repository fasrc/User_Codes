### Purpose:

Program creates 2 randon vectors and computes their dot product.

### Contents:

(1) omp_dot.cpp: C++ source code

(2) Makefile: Makefile to compile the source code

(3) run.sbatch: Btach-job submission script to send the job to the queue.

### Example Usage:

	source new-modules.sh
	make
	sbatch run.sbatch
    
### Example Output:

```
Running on 4 threads.
Scallar product of A and B: 24999527
Time in FOR loop: 0.081118567 seconds.
```
