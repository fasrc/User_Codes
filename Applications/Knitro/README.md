#### PURPOSE:

Knitro is an optimization software library for finding solutions of both continuous (smooth) optimization models (with or without constraints), 
as well as discrete optimization models with integer or binary variables (i.e. mixed integer programs). Knitro is primarily designed for finding 
local optimal solutions of large-scale, continuous nonlinear problems.

This example illustrates using Knitro with its MATLAB interface in a batch mode on the Odyssey cluster at Harvard University. 

#### CONTENTS:

(1) knitro_test.m: MATLAB source code

(2) run.sbatch: Batch job submission script for sending the job to the queue
                       
#### EXAMPLE USAGE:

	source new-modules.sh
	module load matlab/R2015b-fasrc01
	module load knitro/9.1.0-fasrc01
	sbatch run.sbatch


#### EXAMPLE OUTPUT:

```
======================================
  Academic Ziena License (NOT FOR COMMERCIAL USE)
             KNITRO 9.1.0
          Ziena Optimization
======================================

KNITRO presolve eliminated 0 variables and 0 constraints.

algorithm:            1
gradopt:              4
hessopt:              2
honorbnds:            1
maxit:                10000
outlev:               1
par_concurrent_evals: 0
KNITRO changing bar_switchrule from AUTO to 2.
KNITRO changing bar_murule from AUTO to 4.
KNITRO changing bar_initpt from AUTO to 3.
KNITRO changing bar_penaltyrule from AUTO to 2.
KNITRO changing bar_penaltycons from AUTO to 1.
KNITRO changing bar_switchrule from AUTO to 2.
KNITRO changing linsolver from AUTO to 2.

Problem Characteristics
-----------------------
Objective goal:  Minimize
Number of variables:                     3
    bounded below:                       3
    bounded above:                       0
    bounded below and above:             0
    fixed:                               0
    free:                                0
Number of constraints:                   2
    linear equalities:                   1
    nonlinear equalities:                0
    linear inequalities:                 0
    nonlinear inequalities:              1
    range:                               0
Number of nonzeros in Jacobian:          6
Number of nonzeros in Hessian:           6

EXIT: Locally optimal solution found.

Final Statistics
----------------
Final objective value               =   9.36000000000340e+02
Final feasibility error (abs / rel) =   7.11e-15 / 5.47e-16
Final optimality error  (abs / rel) =   2.25e-09 / 1.41e-10
# of iterations                     =          9 
# of CG iterations                  =          0 
# of function evaluations           =         40
# of gradient evaluations           =          0
Total program time (secs)           =       0.22705 (     0.208 CPU time)
Time spent in evaluations (secs)    =       0.03026

===============================================================================
```

#### REFERENCES:

* [Official Knitro User's Manual](https://www.artelys.com/tools/knitro_doc/index.html)
* [Knitro / MATLAB reference] (https://www.artelys.com/tools/knitro_doc/3_referenceManual/knitromatlabReference.html)

