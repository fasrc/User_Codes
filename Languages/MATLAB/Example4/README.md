#### Purpose:

MATLAB example code illustrating using job arrays in SLURM. The specific example computes the sum of integers from 1 through N, where N is the SLURM job-array index.

#### Contents:

* <code>serial\_sum.m</code>: MATLAB source code
* <code>run.sbatch</code>: Batch-job submission script

#### Example Batch-Job Submission Script:

```bash
#!/bin/bash
#SBATCH -J array_test
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 0-00:30
#SBATCH -o array_test_%a.out
#SBATCH -e array_test_%a.err
#SBATCH -p serial_requeue
#SBATCH --mem=4000
#SBATCH --array=17-32

# Load required modules
source new-modules.sh
module load matlab/R2016b-fasrc01

# Run program
srun -n 1 -c 1 matlab -nosplash -nodesktop -nodisplay -r "serial_sum($SLURM_ARRAY_TASK_ID);exit"
```

#### Example Usage:

```bash
source new-modules.sh
module load matlab/R2016b-fasrc01
sbatch run.sbatch
```

#### Example Output:

```
cat array_test_32.out 

                            < M A T L A B (R) >
                  Copyright 1984-2016 The MathWorks, Inc.
                   R2016b (9.1.0.441655) 64-bit (glnxa64)
                             September 7, 2016

 
To get started, type one of these: helpwin, helpdesk, or demo.
For product information, visit www.mathworks.com.
 

Sum of integers from 1 to 32 is 528.
```
