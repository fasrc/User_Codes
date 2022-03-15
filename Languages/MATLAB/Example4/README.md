#### Purpose:

MATLAB example code illustrating using job arrays in SLURM. The specific example computes the sum of integers from 1 through N, where N is the SLURM job-array index.

#### Contents:

* <code>serial\_sum.m</code>: MATLAB source code
* <code>run.sbatch</code>: Batch-job submission script

#### Example Code:

```matlab
%===========================================================================
% Program: serial_sum( N )
%          Calculates integer sum from 1 to N
%
% Run:     matlab -nodesktop -nodisplay -nosplash -r "serial_sum(N); exit"
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
#SBATCH -J array_test
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 0-00:30
#SBATCH -o array_test_%a.out
#SBATCH -e array_test_%a.err
#SBATCH -p test
#SBATCH --mem=4000
#SBATCH --array=17-32

# Load required modules
module load matlab/R2021a-fasrc01

# Run program
srun -n 1 -c 1 matlab -nosplash -nodesktop -nodisplay -r "serial_sum($SLURM_ARRAY_TASK_ID);exit"
```

#### Example Usage:

```bash
sbatch run.sbatch
```

#### Example Output:

```
cat array_test_32.out

                            < M A T L A B (R) >
                  Copyright 1984-2021 The MathWorks, Inc.
                  R2021a (9.10.0.1602886) 64-bit (glnxa64)
                             February 17, 2021

 
To get started, type doc.
For product information, visit www.mathworks.com.
 
Sum of integers from 1 to 32 is 528.
```
