### Purpose:

Example of using Armadillo C++ libraries on the cluster.

### Contents:

(1) armadillo_test.cpp: C++ source code

(2) Makefile: Makefile to compile the source code

(3) run.sbatch: Btach-job submission script to send the job to the queue

### Example Usage:

	source new-modules.sh
	module load gcc/4.8.2-fasrc01
	module load armadillo/5.100.2-fasrc01
	make
	sbatch run.sbatch
    
### Example Output:

```
   0.9713   1.3566   0.7946   1.6896
   1.2593   1.1457   0.9011   1.6260
   1.1954   0.8484   1.0444   1.6753
   1.6225   1.5009   1.2935   2.2019
```
