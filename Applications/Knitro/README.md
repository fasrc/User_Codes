## Knitro
<img src="Images/knitro-logo.png" alt="Knitro-logo" width="250"/>

### What is Knitro?

[Artelys Knitro](https://www.artelys.com/solvers/knitro/) is a leading optimization solver for difficult large-scale nonlinear problems. Four state-of-the-art algorithms and many user options enable users to customize Artelys Knitro to control performance tuning. Each algorithm addresses the full range of continuous or smooth nonlinear optimization problems, and each is constructed for maximal large-scale efficiency. Many respected companies in finance, energy, engineering, telecommunications, and other high-tech and scientific industries rely on Artelys Knitro to solve their most demanding problems.

### Contents:

* <code>knitro_test.m</code>: Matlab source code
* <code>run.sbatch</code>: Job submission script for the Matlab example
* <code>knitro_test.out</code>: STD output from the Matlab example
* <code>knitro_test_python.py</code>: Python source code
* <code>run_python.sbatch</code>: Job submission script for the Python example
* <code>knitro_test_python.out</code>: STD output from the Python example

### Using Knitro in Matlab

Artelys Knitro is easily interfaced to MATLAB via its MEX interface. Artelys Knitro uses syntax similar to Optimization Toolbox functions and provides additional flexibility and more options for very difficult and large-scale problems.

The below example illustrates using Knitro with its MATLAB interface in a batch mode on the [FAS Cannon cluster](https://www.rc.fas.harvard.edu/about/cluster-architecture) at Harvard University. 

**Matlab source code**

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
x = knitromatlab(obj, x0, A, b, Aeq, beq, lb, ub, nlcon);
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
module load knitro/10.1.2-fasrc01

# --- Run the program ---
srun -n 1 -c 1 matlab -nosplash -nodesktop -nodisplay -r "knitro_test;exit"
```
                       
**Example usage**

```bash
sbatch run.sbatch
```

**Example output**

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
         Artelys Knitro 10.1.2
=======================================

Knitro presolve eliminated 0 variables and 0 constraints.

algorithm:            1
gradopt:              4
hessopt:              2
honorbnds:            1
maxit:                10000
outlev:               1
par_concurrent_evals: 0
Knitro changing bar_initpt from AUTO to 3.
Knitro changing bar_murule from AUTO to 4.
Knitro changing bar_penaltycons from AUTO to 1.
Knitro changing bar_penaltyrule from AUTO to 2.
Knitro changing bar_switchrule from AUTO to 2.
Knitro changing linsolver from AUTO to 2.

Problem Characteristics                    ( Presolved)
-----------------------
Objective goal:  Minimize
Number of variables:                     3 (         3)
    bounded below:                       3 (         3)
    bounded above:                       0 (         0)
    bounded below and above:             0 (         0)
    fixed:                               0 (         0)
    free:                                0 (         0)
Number of constraints:                   2 (         2)
    linear equalities:                   1 (         1)
    nonlinear equalities:                0 (         0)
    linear inequalities:                 0 (         0)
    nonlinear inequalities:              1 (         1)
    range:                               0 (         0)
Number of nonzeros in Jacobian:          6 (         6)
Number of nonzeros in Hessian:           6 (         6)

EXIT: Locally optimal solution found.

Final Statistics
----------------
Final objective value               =   9.36000000000340e+02
Final feasibility error (abs / rel) =   7.11e-15 / 5.47e-16
Final optimality error  (abs / rel) =   2.25e-09 / 1.41e-10
# of iterations                     =          9 
# of CG iterations                  =          0 
# of function evaluations           =         44
# of gradient evaluations           =          0
Total program time (secs)           =       0.34302 (     0.296 CPU time)
Time spent in evaluations (secs)    =       0.06101

===============================================================================
```

### Using Knitro in Python

*Note:* Please note that the current Knitro version works only with Python 2.

**Python Source Code**

```python
#*******************************************************
#* Copyright (c) 2015 by Artelys                       *
#* All Rights Reserved                                 *
#*******************************************************

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Knitro example driver using callback mode, defining separate
#  callback functions for each evaluation request.
#
#  This executable invokes Knitro to solve a simple nonlinear
#  optimization test problem.  The purpose is to illustrate how to
#  invoke Knitro using the Python language API.
#
#  Before running, make sure ../../lib is in the load path.
#  To run:
#    python exampleHS15
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


from knitro import *


 ## Solve test problem HS15 from the Hock & Schittkowski collection.
 #
 #  min   100 (x2 - x1^2)^2 + (1 - x1)^2
 #  s.t.  x1 x2 >= 1
 #        x1 + x2^2 >= 0
 #        x1 <= 0.5
 #
 #  The standard start point (-2, 1) usually converges to the standard
 #  minimum at (0.5, 2.0), with final objective = 306.5.
 #  Sometimes the solver converges to another local minimum
 #  at (-0.79212, -1.26243), with final objective = 360.4.
 ##

#----------------------------------------------------------------
#   METHOD evaluateFC
#----------------------------------------------------------------
 ## Compute the function and constraint values at x.
 #
 #  For more information about the arguments, refer to the Knitro
 #  manual, especially the section on the Callable Library.
 ##
def evaluateFC (x, c):
    tmp = x[1] - x[0]*x[0]
    obj = 100.0 * tmp*tmp + (1.0 - x[0])*(1.0 - x[0])
    c[0] = x[0] * x[1]
    c[1] = x[0] + x[1]*x[1]
    return obj


#----------------------------------------------------------------
#   METHOD evaluateGA
#----------------------------------------------------------------
 ## Compute the function and constraint first deriviatives at x.
 #
 #  For more information about the arguments, refer to the Knitro
 #  manual, especially the section on the Callable Library.
 ##
def evaluateGA (x, objGrad, jac):
    tmp = x[1] - x[0]*x[0]
    objGrad[0] = (-400.0 * tmp * x[0]) - (2.0 * (1.0 - x[0]))
    objGrad[1] = 200.0 * tmp
    jac[0] = x[1]
    jac[1] = x[0]
    jac[2] = 1.0
    jac[3] = 2.0 * x[1]


#----------------------------------------------------------------
#   METHOD evaluateH
#----------------------------------------------------------------
 ## Compute the Hessian of the Lagrangian at x and lambda.
 #
 #  For more information about the arguments, refer to the Knitro
 #  manual, especially the section on the Callable Library.
 ##
def evaluateH (x, lambda_, sigma, hess):
    hess[0] = sigma * ( (-400.0 * x[1]) + (1200.0 * x[0]*x[0]) + 2.0)
    hess[1] = (sigma * (-400.0 * x[0])) + lambda_[0]
    hess[2] = (sigma * 200.0) + (lambda_[1] * 2.0)


#----------------------------------------------------------------
#   MAIN METHOD FOR TESTING
#----------------------------------------------------------------

#---- DEFINE THE OPTIMIZATION TEST PROBLEM.
#---- FOR MORE INFORMATION ABOUT THE PROBLEM DEFINITION, REFER
#---- TO THE KNITRO MANUAL, ESPECIALLY THE SECTION ON THE
#---- CALLABLE LIBRARY.
n = 2
objGoal = KTR_OBJGOAL_MINIMIZE
objType = KTR_OBJTYPE_GENERAL;
bndsLo = [ -KTR_INFBOUND, -KTR_INFBOUND ]
bndsUp = [ 0.5, KTR_INFBOUND ]
m = 2
cType = [ KTR_CONTYPE_QUADRATIC, KTR_CONTYPE_QUADRATIC ]
cBndsLo = [ 1.0, 0.0 ]
cBndsUp = [ KTR_INFBOUND, KTR_INFBOUND ]
jacIxConstr = [ 0, 0, 1, 1 ]
jacIxVar    = [ 0, 1, 0, 1 ]
hessRow = [ 0, 0, 1 ]
hessCol = [ 0, 1, 1 ]

xInit = [ -2.0, 1.0 ]


#---- SETUP AND RUN KNITRO TO SOLVE THE PROBLEM.

#---- CREATE A NEW KNITRO SOLVER INSTANCE.
kc = KTR_new()
if kc == None:
    raise RuntimeError ("Failed to find a Knitro license.")

#---- DEMONSTRATE HOW TO SET KNITRO PARAMETERS.
if KTR_set_char_param_by_name(kc, "outlev", "all"):
    raise RuntimeError ("Error setting parameter 'outlev'")
if KTR_set_int_param_by_name(kc, "hessopt", 1):
    raise RuntimeError ("Error setting parameter 'hessopt'")
if KTR_set_int_param_by_name(kc, "hessian_no_f", 1):
    raise RuntimeError ("Error setting parameter 'hessian_no_f'")
if KTR_set_double_param_by_name(kc, "feastol", 1.0E-10):
    raise RuntimeError ("Error setting parameter 'feastol'")

#------------------------------------------------------------------ 
#     FUNCTION callbackEvalFC
#------------------------------------------------------------------
 ## The signature of this function matches KTR_callback in knitro.h.
 #  Only "obj" and "c" are modified.
 ##
def callbackEvalFC (evalRequestCode, n, m, nnzJ, nnzH, x, lambda_, obj, c, objGrad, jac, hessian, hessVector, userParams):
    if evalRequestCode == KTR_RC_EVALFC:
        obj[0] = evaluateFC(x, c)
        return 0
    else:
        return KTR_RC_CALLBACK_ERR

#------------------------------------------------------------------
#     FUNCTION callbackEvalGA
#------------------------------------------------------------------
 ## The signature of this function matches KTR_callback in knitro.h.
 #  Only "objGrad" and "jac" are modified.
 ##
def callbackEvalGA (evalRequestCode, n, m, nnzJ, nnzH, x, lambda_, obj, c, objGrad, jac, hessian, hessVector, userParams):
    if evalRequestCode == KTR_RC_EVALGA:
        evaluateGA(x, objGrad, jac)
        return 0
    else:
        return KTR_RC_CALLBACK_ERR

#------------------------------------------------------------------
#     FUNCTION callbackEvalH
#------------------------------------------------------------------
 ## The signature of this function matches KTR_callback in knitro.h.
 #  Only "hessian" or "hessVector" is modified.
 ##
def callbackEvalH (evalRequestCode, n, m, nnzJ, nnzH, x, lambda_, obj, c, objGrad, jac, hessian, hessVector, userParams):
    if evalRequestCode == KTR_RC_EVALH:
        evaluateH(x, lambda_, 1.0, hessian)
        return 0
    elif evalRequestCode == KTR_RC_EVALH_NO_F:
        evaluateH(x, lambda_, 0.0, hessian)
        return 0
    else:
        return KTR_RC_CALLBACK_ERR

#---- REGISTER THE CALLBACK FUNCTIONS THAT PERFORM PROBLEM EVALUATION.
#---- THE HESSIAN CALLBACK ONLY NEEDS TO BE REGISTERED FOR SPECIFIC
#---- HESSIAN OPTIONS (E.G., IT IS NOT REGISTERED IF THE OPTION FOR
#---- BFGS HESSIAN APPROXIMATIONS IS SELECTED).
if KTR_set_func_callback(kc, callbackEvalFC):
    raise RuntimeError ("Error registering function callback.")
if KTR_set_grad_callback(kc, callbackEvalGA):
    raise RuntimeError ("Error registering gradient callback.")
if KTR_set_hess_callback(kc, callbackEvalH):
    raise RuntimeError ("Error registering hessian callback.")

#---- INITIALIZE KNITRO WITH THE PROBLEM DEFINITION.
ret = KTR_init_problem (kc, n, objGoal, objType, bndsLo, bndsUp,
                                cType, cBndsLo, cBndsUp,
                                jacIxVar, jacIxConstr,
                                hessRow, hessCol,
                                xInit, None)
if ret:
	raise RuntimeError ("Error initializing the problem, "
                                + "Knitro status = "
                                + str(ret))

#---- SOLVE THE PROBLEM.
#----
#---- RETURN STATUS CODES ARE DEFINED IN "knitro.h" AND DESCRIBED
#---- IN THE KNITRO MANUAL.
x       = [0] * n
lambda_ = [0] * (m + n)
obj     = [0]
nStatus = KTR_solve (kc, x, lambda_, 0, obj,
                         None, None, None, None, None, None)

print
print
if nStatus != 0:
    raise RuntimeError ("Knitro failed to solve the problem, final status = %d" % nStatus)
else:
    #---- AN EXAMPLE OF OBTAINING SOLUTION INFORMATION.
    print "Knitro successful, feasibility violation    = %e" % KTR_get_abs_feas_error (kc)
    print "                   KKT optimality violation = %e" % KTR_get_abs_opt_error (kc)

#---- BE CERTAIN THE NATIVE OBJECT INSTANCE IS DESTROYED.
KTR_free(kc)

#+++++++++++++++++++ End of source file +++++++++++++++++++++++++++++
```
**Example batch-job submission script**

```bash
#!/bin/bash
#SBATCH -J knitro_test
#SBATCH -o knitro_test_python.out
#SBATCH -e knitro_test_python.err
#SBATCH -p test
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 30
#SBATCH --mem=40000

# --- Load required modules ---
module load knitro/10.1.2-fasrc01

# --- Run the program ---
srun -n 1 -c 1 python knitro_test_python.py
```
If you name the above script, e.g., <code>run_python.sbatch</code>, it is submited to the queue with:

```bash
sbatch run_python.sbatch
```

**Example output**

```bash
$ cat knitro_test_python.out 

=======================================
           Academic License
       (NOT FOR COMMERCIAL USE)
         Artelys Knitro 10.1.2
=======================================

Knitro presolve eliminated 0 variables and 0 constraints.

feastol:              1e-10
hessian_no_f:         1
outlev:               6
Knitro changing algorithm from AUTO to 1.
Knitro changing bar_initpt from AUTO to 3.
Knitro changing bar_murule from AUTO to 4.
Knitro changing bar_penaltycons from AUTO to 1.
Knitro changing bar_penaltyrule from AUTO to 2.
Knitro changing bar_switchrule from AUTO to 2.
Knitro changing linsolver from AUTO to 2.

Problem Characteristics                    ( Presolved)
-----------------------
Objective goal:  Minimize
Number of variables:                     2 (         2)
    bounded below:                       0 (         0)
    bounded above:                       1 (         1)
    bounded below and above:             0 (         0)
    fixed:                               0 (         0)
    free:                                1 (         1)
Number of constraints:                   2 (         2)
    linear equalities:                   0 (         0)
    nonlinear equalities:                0 (         0)
    linear inequalities:                 0 (         0)
    nonlinear inequalities:              2 (         2)
    range:                               0 (         0)
Number of nonzeros in Jacobian:          4 (         4)
Number of nonzeros in Hessian:           3 (         3)

  Iter     fCount     Objective      FeasError   OptError    ||Step||    CGits 
--------  --------  --------------  ----------  ----------  ----------  -------
       0         2    9.090000e+02   3.000e+00
       1         3    7.992179e+02   2.859e+00   2.191e+01   7.245e-02        0
       2         4    1.865455e+01   9.076e-01   3.917e+01   2.197e+00        0
       3        12    3.211028e+02   8.846e-01   6.751e+00   1.336e+00        8
       4        13    1.437527e+01   5.021e-01   6.570e-01   1.162e+00        2
       5        14    3.543851e+01   3.873e-01   3.873e-01   2.008e-01        0
       6        15    1.144533e+02   2.197e-01   5.820e-01   4.076e-01        0
       7        16    2.342032e+02   7.408e-02   7.408e-02   4.123e-01        0
       8        17    3.011424e+02   6.019e-03   3.302e-02   1.904e-01        0
       9        18    3.064931e+02   9.851e-06   1.490e-04   1.470e-02        0
      10        19    3.065000e+02   0.000e+00   7.038e-10   1.970e-05        0

EXIT: Locally optimal solution found.

Final Statistics
----------------
Final objective value               =   3.06500000151717e+02
Final feasibility error (abs / rel) =   0.00e+00 / 0.00e+00
Final optimality error  (abs / rel) =   7.04e-10 / 4.82e-11
# of iterations                     =         10 
# of CG iterations                  =         10 
# of function evaluations           =         19
# of gradient evaluations           =         12
# of Hessian evaluations            =         10
Total program time (secs)           =       0.01247 (     0.005 CPU time)
Time spent in evaluations (secs)    =       0.00116

Constraint Vector 		     Lagrange Multipliers
----------------- 		     ---------------------
c[       0] =   1.00000000011e+00,   lambda[       0] =  -7.00000000099e+02
c[       1] =   4.50000000152e+00,   lambda[       1] =  -1.69345778619e-08

Solution Vector
---------------
x[       0] =   4.99999999956e-01,   lambda[       2] =   1.75100000054e+03
x[       1] =   2.00000000039e+00,   lambda[       3] =   0.00000000000e+00

===============================================================================



Knitro successful, feasibility violation    = 0.000000e+00
                   KKT optimality violation = 7.038478e-10
```

### References:

* [Official Knitro User's Manual](https://www.artelys.com/tools/knitro_doc/index.html)
* [Knitro / MATLAB reference](https://www.artelys.com/tools/knitro_doc/3_referenceManual/knitromatlabReference.html)

