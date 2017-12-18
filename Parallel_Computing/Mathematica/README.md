### Purpose:

Program computes PI with parallel Monte-Carlo method.

### Contents:

* parallel_pi.m: Mathematica source code
* run.sbatch: Btach-job submission script to send the job to the queue.

#### Example Batch-Job Submission Script:

```bash
#!/bin/bash
#SBATCH -J parallel_pi
#SBATCH -o parallel_pi.out
#SBATCH -e parallel_pi.err
#SBATCH -p shared
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -t 0-00:30
#SBATCH --mem=4000

# Load required software modules
source new-modules.sh
module load mathematica/11.1.1-fasrc01

# Run program
srun -n 1 -c 8 math -script parallel_pi.m
```

### Example Usage:

```bash
source new-modules.sh
module load mathematica/11.1.1-fasrc01
sbatch run.sbatch
```
 
### Example Output:

```
Parallel calculation of PI via Monte-Carlo method.
Updating from Wolfram Research server ...
 Number of kernels: 8
 Total number of hits: 100000000
 Number of hits per core: 12500000
 Computed PI = 3.1410542
 Time in parallel calculation: 2.467308`6.8437683603959725
 Total time: 9.950124`7.449373486521115
```
