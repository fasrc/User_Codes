### Purpose

MATLAB example code illustrating using job arrays and random-vector generation. The specific example creates N random vectors, where N is the job-array dimension, with uniformly distributed elements in a predefined interval. The program takes as arguments the vector dimension (n), the interval boundaries [a, b], and the seed for the random number generator (iseed).

### Contents

* `rnd_test.m`: MATLAB source code
* `run.sbatch`: Batch-job submission script

### Example MATLAB Source Code

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

### Example Batch-Job Submission Script

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

### Example Usage

```bash
sbatch run.sbatch
```

### Example Output

```bash
cat array_test_rnd_1.out
iseed = 37106878

                            < M A T L A B (R) >
                  Copyright 1984-2022 The MathWorks, Inc.
                  R2022b (9.13.0.2049777) 64-bit (glnxa64)
                              August 24, 2022

 
To get started, type doc.
For product information, visit www.mathworks.com.
 
Random vector: [ 0.766367 -1.317896 1.396659 0.497333 -0.065059 0.022468 -1.903101 0.244747 0.966363 -0.962693  ]
Sum of elements: -0.354811
```
