### Using Knitro with Matlab

Artelys Knitro is easily interfaced to MATLAB via its MEX interface. Artelys Knitro uses syntax similar to Optimization Toolbox functions and provides additional flexibility and more options for very difficult and large-scale problems.

### Contents:

* <code>knitro_test.m</code>: Matlab source code
* <code>run.sbatch</code>: Job submission script for the Matlab example
* <code>knitro_test.out</code>: STD output from the Matlab example

The below example illustrates using Knitro with its MATLAB interface in a batch mode on the [FAS Cannon cluster](https://www.rc.fas.harvard.edu/about/cluster-architecture) at Harvard University. 

### Matlab source code:

```matlab
%=====================================================================
% Program: knitro_test.m
%          Program illustrates use of Knitro with MATLAB
%=====================================================================

% objective to minimize
obj = @(x) 1000 - x(1)^2 - 2*x(2)^2 - x(3)^2 - x(1)*x(2) - x(1)*x(3);

% No nonlinear equality constraints.
ceq  = [];

% Specify nonlinear inequality constraint to be nonnegative
c2 =  @(x) x(1)^2 + x(2)^2 + x(3)^2 - 25;

% "nlcon" should return [c, ceq] with c(x) <= 0 and ceq(x) = 0
% so we need to negate the inequality constraint above
nlcon = @(x)deal(-c2(x), ceq);

% Initial point
x0  = [2; 2; 2];

% No linear inequality contraint ("A*x <= b")
A = [];
b = [];

% Since the equality constraint "c1" is linear, specify it here  ("Aeq*x = beq")
Aeq = [8 14 7];
beq = [56];

% lower and upper bounds
lb = zeros(3,1);
ub = [];

% solver call
x = knitro_nlp(obj, x0, A, b, Aeq, beq, lb, ub, nlcon);
```

**Example batch-job submission script**

```bash
#!/bin/bash
#SBATCH -J knitro_test
#SBATCH -o knitro_test.out
#SBATCH -e knitro_test.err
#SBATCH -p test
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 30
#SBATCH --mem=40000

# --- Load required modules ---
module load matlab/R2022b-fasrc01
module load knitro/13.2.0-fasrc01

# --- Run the program ---
srun -n 1 -c 1 matlab -nosplash -nodesktop -nodisplay -r "knitro_test;exit"
```
                       
### Example usage:

```bash
sbatch run.sbatch
```

### Example output:

Upon the job completion the results will be in the <code>knitro_test.out</code> file.

```bash
$ cat knitro_test.out 
                                                     < M A T L A B (R) >
                                           Copyright 1984-2022 The MathWorks, Inc.
                                          R2022b (9.13.0.2049777) 64-bit (glnxa64)
                                                       August 24, 2022

 
To get started, type doc.
For product information, visit www.mathworks.com.
 

=======================================
           Academic License
       (NOT FOR COMMERCIAL USE)
         Artelys Knitro 13.2.0
=======================================

Knitro presolve eliminated 0 variables and 0 constraints.

concurrent_evals:        0
gradopt:                 4

Problem Characteristics                                 (   Presolved)
-----------------------
Objective goal:  Minimize
Objective type:  general
Number of variables:                                  3 (           3)
    bounded below only:                               3 (           3)
    bounded above only:                               0 (           0)
    bounded below and above:                          0 (           0)
    fixed:                                            0 (           0)
    free:                                             0 (           0)
Number of constraints:                                2 (           2)
    linear equalities:                                1 (           1)
    quadratic equalities:                             0 (           0)
    gen. nonlinear equalities:                        0 (           0)
    linear one-sided inequalities:                    0 (           0)
    quadratic one-sided inequalities:                 0 (           0)
    gen. nonlinear one-sided inequalities:            1 (           1)
    linear two-sided inequalities:                    0 (           0)
    quadratic two-sided inequalities:                 0 (           0)
    gen. nonlinear two-sided inequalities:            0 (           0)
Number of nonzeros in Jacobian:                       6 (           6)
Number of nonzeros in Hessian:                        0 (           6)

Knitro using the Interior-Point/Barrier Direct algorithm.

  Iter      Objective      FeasError   OptError    ||Step||    CGits 
--------  --------------  ----------  ----------  ----------  -------
       0    9.760000e+02   1.300e+01
       9    9.360000e+02   0.000e+00   2.017e-07   1.675e-07        0

EXIT: Locally optimal solution found.

Final Statistics
----------------
Final objective value               =   9.36000000000728e+02
Final feasibility error (abs / rel) =   0.00e+00 / 0.00e+00
Final optimality error  (abs / rel) =   2.02e-07 / 1.26e-08
# of iterations                     =          9 
# of CG iterations                  =          0 
# of function evaluations           =         40
# of gradient evaluations           =          0
Total program time (secs)           =       0.25940 (     0.343 CPU time)
Time spent in evaluations (secs)    =       0.05838

===============================================================================
```

### References:

* [Official Knitro User's Manual](https://www.artelys.com/tools/knitro_doc/index.html)
* [Knitro / MATLAB reference](https://www.artelys.com/tools/knitro_doc/3_referenceManual/knitromatlabReference.html)
