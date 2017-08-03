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
#SBATCH -J array_test
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 0-00:30
#SBATCH -o array_test_%a.out
#SBATCH -e array_test_%a.err
#SBATCH -p serial_requeue
#SBATCH --mem=4000
#SBATCH --array=1-3

# Load required modules
source new-modules.sh
module load matlab/R2016b-fasrc01

# Seed for random number generator
iseed=$(($SLURM_ARRAY_JOB_ID+$SLURM_ARRAY_TASK_ID))
echo "iseed = $iseed"

# Run program
srun -n 1 -c 1 matlab -nosplash -nodesktop -nodisplay -r "rnd_test(10, -2, 2, $iseed); exit"
```

#### Example Usage:

```bash
source new-modules.sh
module load matlab/R2016b-fasrc01
sbatch run.sbatch
```

#### Example Output:

```
iseed = 25603779

                            < M A T L A B (R) >
                  Copyright 1984-2016 The MathWorks, Inc.
                   R2016b (9.1.0.441655) 64-bit (glnxa64)
                             September 7, 2016

 
To get started, type one of these: helpwin, helpdesk, or demo.
For product information, visit www.mathworks.com.
 
Random vector: [ 0.419598 -1.146946 -0.213620 0.857326 0.114036 -0.571077 1.146119 0.984016 1.501505 -0.129344  ]
Sum of elements: 2.961613
```
