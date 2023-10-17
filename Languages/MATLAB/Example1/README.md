#### Purpose:

MATLAB example code computing PI via Monte-Carlo method.

#### Contents:

* <code>pi\_monte\_carlo.m</code>: MATLAB source code
* <code>run.sbatch</code>: Batch-job submission script

#### Example Batch-Job Submission Script:

https://github.com/fasrc/User_Codes/blob/56c0f71f95caf1f3dda10db0bd98f3fa2829043e/Languages/MATLAB/Example1/run.sbatch#L1-L12

#### Example Usage:

```bash
sbatch run.sbatch
```

#### Example Code:

```matlab
%=====================================================================
% Program: pi_monte_carlo.m
%
%          Parallel Monte Carlo calculation of PI
%
% Run:     matlab -nosplash -nodesktop -nodisplay -r "pi_monte_carlo"
%=====================================================================
R = 1.0;
darts = 1e6;
count = 0;
for i = 1:darts
  % Compute the X and Y coordinates of where the dart hit the.........
  % square using Uniform distribution.................................
  x = R*rand(1);
  y = R*rand(1);
  if x^2 + y^2 <= R^2
    % Increment the count of darts that fell inside of the circle.....
    count = count + 1; % Count is a reduction variable.
  end
end
% Compute pi..........................................................
myPI = 4*count/darts;
fprintf('The computed value of pi is %8.7f.\n',myPI);
exit;
```

#### Example Output:

```
cat pi_monte_carlo.out 

                            < M A T L A B (R) >
                  Copyright 1984-2021 The MathWorks, Inc.
                  R2021a (9.10.0.1602886) 64-bit (glnxa64)
                             February 17, 2021

 
To get started, type doc.
For product information, visit www.mathworks.com.
 
The computed value of pi is 3.1399480.
```
