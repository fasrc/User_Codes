### Purpose

MATLAB example code illustrating using job arrays in SLURM. The specific example computes the sum of integers from 1 through N, where N is the SLURM job-array index.

### Contents

* `serial_sum.m`: MATLAB source code
* `run.sbatch`: Batch-job submission script

### Example MATLAB Source Code

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

### Example Batch-Job Submission Script

```bash
#!/bin/bash
#SBATCH -J array_test           # job name
#SBATCH -o array_test_%a.out    # standard output file
#SBATCH -e array_test_%a.err    # standard error file
#SBATCH -p serial_requeue       # partition
#SBATCH -c 1                    # number of cores
#SBATCH -t 0-00:30              # time in D-HH:MM
#SBATCH --mem=4000              # memory in MB
#SBATCH --array=17-32           # array indices

# Load required modules
module load matlab

# Run program
srun -c $SLURM_CPUS_PER_TASK matlab -nosplash -nodesktop -nodisplay -r "serial_sum($SLURM_ARRAY_TASK_ID);exit"
```

### Example Usage

```bash
sbatch run.sbatch
```

### Example Output

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
