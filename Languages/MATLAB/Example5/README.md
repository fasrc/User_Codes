#### Purpose:

MATLAB example code illustrating using job arrays and random-vector generation. The specific example creates N random vectors, where N is the job-array dimension, with uniformly distributed elements in a predefined interval. The program takes as arguments the vector dimension (n), the interval boundaries [a, b], and the seed for the random number generator (iseed).

#### Contents:

* <code>rnd\_test.m</code>: MATLAB source code
* <code>run.sbatch</code>: Batch-job submission script

#### MATLAB Source Code:

```matlab
%===============================================================
% Program: rnd_test(n, a ,b, iseed)
%          Create a random vector and sum up its elements
%===============================================================
function [] = rnd_test(n, a, b, iseed)
  % Create random vector vec
  vec = rand_vec(n, a, b, iseed);

  % Sum up elements of vec
  s = vec_sum(n, vec);

  % Print out results
  r = sprintf('%f ', vec);
  fprintf('Random vector: [ %s ]\n', r);
  fprintf('Sum of elements: %f\n', s);
end

%===============================================================
% Function: rand_vec(n, a ,b)
%           Create a random vector x of dimension n with
%           uniformly distributed numbers in the interval [a, b]
%===============================================================
function x = rand_vec(n, a, b, iseed)
  rng(iseed);
  x = a + ( b - a ) * rand(n,1);
end

%===============================================================
% Function: vec_sum(n, vec)
%           Sum up elements of vector vec with dimension n
%===============================================================
function s = vec_sum(n, vec)
  s = 0.0;
  for i = 1:n
    s = s + vec(i,1);
  end
end
```

#### Example Batch-Job Submission Script:

```bash
#!/bin/bash
#SBATCH -J array_test_rnd          # job name
#SBATCH -o array_test_rnd_%a.out   # standard output file
#SBATCH -e array_test_rnd_%a.err   # standard error file
#SBATCH -p test                    # partition
#SBATCH -c 1                       # number of cores
#SBATCH -t 0-00:30                 # time in D-HH:MM
#SBATCH --mem=4000                 # memory in MB
#SBATCH --array=1-3                # array indices

# Load required modules
module load matlab

# Seed for random number generator
iseed=$(($SLURM_ARRAY_JOB_ID+$SLURM_ARRAY_TASK_ID))
echo "iseed = $iseed"

# Run program
srun -c $SLURM_CPUS_PER_TASK matlab -nosplash -nodesktop -nodisplay -r "rnd_test(10, -2, 2, $iseed); exit"
```

#### Example Usage:

```bash
sbatch run.sbatch
```

#### Example Output:

```
cat array_test_1.out
iseed = 65859596

                            < M A T L A B (R) >
                  Copyright 1984-2021 The MathWorks, Inc.
                  R2021a (9.10.0.1602886) 64-bit (glnxa64)
                             February 17, 2021

 
To get started, type doc.
For product information, visit www.mathworks.com.
 
Random vector: [ -0.134309 1.373540 -1.385494 -1.880850 -1.963074 -1.782992 -1.509674 0.931186 -1.617545 1.217141  ]
Sum of elements: -6.752070
```
