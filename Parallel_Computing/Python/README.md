### Purpose:

Program illustrates use of MPI for Python (mpi4py) on the Odyssey cluster.

### Contents:

(1) mpi4py_test.py: Python MPI source code.

(2) run.sbatch: Btach-job submission script to send the job to the queue.

### Example Usage:

	source new-modules.sh
	module load python/2.7.6-fasrc01	
	sbatch run.sbatch
 
### Example Output:

```
[pkrastev@sa01 parallel_python]$ cat mpi4py_test.out
This code is a test for mpi4py.
Rank 0 out of 16
Rank 1 out of 16
Rank 2 out of 16
Rank 3 out of 16
Rank 4 out of 16
Rank 5 out of 16
Rank 6 out of 16
Rank 7 out of 16
Rank 8 out of 16
Rank 9 out of 16
Rank 10 out of 16
Rank 11 out of 16
Rank 12 out of 16
Rank 13 out of 16
Rank 14 out of 16
Rank 15 out of 16
```