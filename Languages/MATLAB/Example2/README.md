#### Purpose:

MATLAB example code computing sum of integers from 1 to N. In this case N=100.

#### Contents:

* <code>serial\_sum.m</code>: MATLAB source code
* <code>run.sbatch</code>: Batch-job submission script

#### Example Batch-Job Submission Script:

```bash
#!/bin/bash
#SBATCH -J serial_sum
#SBATCH -o serial_sum.out
#SBATCH -e serial_sum.err
#SBATCH -p serial_requeue
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 0-00:30
#SBATCH --mem=4000

# Load required software modules
source new-modules.sh
module load matlab/R2016b-fasrc01
srun -n 1 -c 1 matlab -nodesktop -nodisplay -nosplash -r "serial_sum(100); exit"
```

#### Example Usage:

```bash
source new-modules.sh
module load matlab/R2016b-fasrc01
sbatch run.sbatch
```

#### Example Output:

```
cat serial_sum.out 

                            < M A T L A B (R) >
                  Copyright 1984-2016 The MathWorks, Inc.
                   R2016b (9.1.0.441655) 64-bit (glnxa64)
                             September 7, 2016

 
To get started, type one of these: helpwin, helpdesk, or demo.
For product information, visit www.mathworks.com.
 
Sum of integers from 1 to 100 is 5050.
```
