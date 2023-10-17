#### Purpose:

MATLAB example code computing sum of integers from 1 to N. In this case N=100.

#### Contents:

* <code>serial\_sum.m</code>: MATLAB source code
* <code>run.sbatch</code>: Batch-job submission script

#### Example Code:

```matlab
%===========================================================================
% Program: serial_sum( N )
%          Calculates integer sum from 1 to N
%
% Run:     matlab -nodesktop -nodisplay -nosplash -r "serial_sum(100); exit"
%===========================================================================
function s = serial_sum(N) 
  s = 0; 
  for i = 1:N 
    s = s + i; 
  end 
  fprintf('Sum of integers from 1 to %d is %d.\n', N, s); 
end
```


#### Example Batch-Job Submission Script:

```bash
#!/bin/bash
#SBATCH -J serial_sum       # job name
#SBATCH -o serial_sum.out   # standard output file
#SBATCH -e serial_sum.err   # standard error file
#SBATCH -p test             # partition
#SBATCH -c 1                # number of cores
#SBATCH -t 0-00:30          # time in D-HH:MM
#SBATCH --mem=4000          # memory in MB

# Load required software modules
module load matlab
srun -c $SLURM_CPUS_PER_TASK matlab -nodesktop -nodisplay -nosplash -r "serial_sum(100); exit"
```

#### Example Usage:

```bash
sbatch run.sbatch
```

#### Example Output:

```
cat serial_sum.out 

                            < M A T L A B (R) >
                  Copyright 1984-2021 The MathWorks, Inc.
                  R2021a (9.10.0.1602886) 64-bit (glnxa64)
                             February 17, 2021

 
To get started, type doc.
For product information, visit www.mathworks.com.
 
Sum of integers from 1 to 100 is 5050.
```
