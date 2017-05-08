#### Purpose:

MATLAB example code computing PI via Monte-Carlo method.

#### Contents:

* <code>pi\_monte\_carlo.m</code>: MATLAB source code
* <code>run.sbatch</code>: Batch-job submission script

#### Example Batch-Job Submission Script:

```bash
#!/bin/bash
#SBATCH -J pi_monte_carlo
#SBATCH -o pi_monte_carlo.out
#SBATCH -e pi_monte_carlo.err
#SBATCH -p serial_requeue
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 0-00:30
#SBATCH --mem=4000

# Load required software modules
source new-modules.sh
module load matlab/R2016b-fasrc01
srun -n 1 -c 1 matlab -nosplash -nodesktop -nodisplay -r "pi_monte_carlo"
```

#### Example Usage:

```bash
source new-modules.sh
module load matlab/R2016b-fasrc01
sbatch run.sbatch
```

#### Example Output:

```
cat pi_monte_carlo.out 

                            < M A T L A B (R) >
                  Copyright 1984-2016 The MathWorks, Inc.
                   R2016b (9.1.0.441655) 64-bit (glnxa64)
                             September 7, 2016

 
To get started, type one of these: helpwin, helpdesk, or demo.
For product information, visit www.mathworks.com.
 
The computed value of pi is 3.1399480.
```