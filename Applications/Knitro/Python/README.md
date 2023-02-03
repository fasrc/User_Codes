### Using Knitro in Python

The Knitro interface for Python is provided with your Knitro distribution as a Python package.

* <code>knitro.opt</code>: Options file
* <code>knitro_test_python.py</code>: Python source code
* <code>run_python.sbatch</code>: Job submission script for the Python example
* <code>knitro_test_python.out</code>: STD output from the Python example

### Python Source Code:

```python
cat knitro_test_python.py 
from knitro import *

# Create a new Knitro solver instance.
try:
    kc = KN_new ()
except:
    print ("Failed to find a valid license.")
    quit ()

# Illustrate how to override default options by reading from
# the knitro.opt file.
KN_load_param_file (kc, "knitro.opt")

# Initialize Knitro with the problem definition.

# Add the variables and set their bounds.
# Note: unset bounds assumed to be infinite.
xIndices = KN_add_vars (kc, 4)
for x in xIndices:
    KN_set_var_lobnds (kc, x, 0.0)

# Add the constraints and set the rhs and coefficients.
KN_add_cons(kc, 2)
KN_set_con_eqbnds (kc, cEqBnds = [5, 8])

# Add Jacobian structure and coefficients.
# First constraint
jacIndexCons = [0, 0, 0]
jacIndexVars = [0, 1, 2]
jacCoefs = [1.0, 1.0, 1.0]
# Second constraint
jacIndexCons += [1, 1, 1]
jacIndexVars += [0, 1, 3]
jacCoefs += [2.0, 0.5, 1.0]
KN_add_con_linear_struct (kc, jacIndexCons, jacIndexVars, jacCoefs)

# Set minimize or maximize (if not set, assumed minimize).
KN_set_obj_goal (kc, KN_OBJGOAL_MINIMIZE)

# Set the coefficients for the objective.
objIndices = [0, 1]
objCoefs = [-4.0, -2.0]
KN_add_obj_linear_struct (kc, objIndices, objCoefs)

# Solve the problem.
# Return status codes are defined in "knitro.py" and described in the Knitro manual.
nStatus = KN_solve (kc)
print ("Knitro converged with final status = %d" % nStatus)

# An example of obtaining solution information.
nStatus, objSol, x, lambda_ =  KN_get_solution (kc)
print ("  optimal objective value  = %e" % objSol)
print ("  optimal primal values x  = (%e, %e, %e, %e)" % (x[0], x[1], x[2], x[3]))
print ("  feasibility violation    = %e" % KN_get_abs_feas_error (kc))
print ("  KKT optimality violation = %e" % KN_get_abs_opt_error (kc))

# Delete the Knitro solver instance.
KN_free (kc)
```
### Example batch-job submission script:

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
module load knitro/13.2.0-fasrc01 
module load python/3.9.12-fasrc01

# --- Run the program ---
srun -n 1 -c 1 python knitro_test_python.py
```
If you name the above script, e.g., <code>run_python.sbatch</code>, it is submited to the queue with:

```bash
sbatch run_python.sbatch
```

### Example output:

```bash
$ cat knitro_test_python.out 

=======================================
           Academic License
       (NOT FOR COMMERCIAL USE)
         Artelys Knitro 13.2.0
=======================================

No start point provided -- Knitro computing one.

Knitro presolve eliminated 0 variables and 0 constraints.

The problem is identified as an LP.

Problem Characteristics                                 (   Presolved)
-----------------------
Objective goal:  Minimize
Objective type:  linear
Number of variables:                                  4 (           4)
    bounded below only:                               4 (           4)
    bounded above only:                               0 (           0)
    bounded below and above:                          0 (           0)
    fixed:                                            0 (           0)
    free:                                             0 (           0)
Number of constraints:                                2 (           2)
    linear equalities:                                2 (           2)
    quadratic equalities:                             0 (           0)
    gen. nonlinear equalities:                        0 (           0)
    linear one-sided inequalities:                    0 (           0)
    quadratic one-sided inequalities:                 0 (           0)
    gen. nonlinear one-sided inequalities:            0 (           0)
    linear two-sided inequalities:                    0 (           0)
    quadratic two-sided inequalities:                 0 (           0)
    gen. nonlinear two-sided inequalities:            0 (           0)
Number of nonzeros in Jacobian:                       6 (           6)
Number of nonzeros in Hessian:                        0 (           0)

Knitro using the Interior-Point/Barrier Direct algorithm.

  Iter      Objective      FeasError   OptError    ||Step||    CGits 
--------  --------------  ----------  ----------  ----------  -------
       0   -8.256633e+00   6.768e-01
       5   -1.733333e+01   6.999e-13   4.534e-10   2.485e-05        0

EXIT: Optimal solution found.

Final Statistics
----------------
Final objective value               =  -1.73333333324245e+01
Final feasibility error (abs / rel) =   7.00e-13 / 7.00e-13
Final optimality error  (abs / rel) =   4.53e-10 / 1.13e-10
# of iterations                     =          5 
# of CG iterations                  =          0 
# of function evaluations           =          0
# of gradient evaluations           =          0
# of Hessian evaluations            =          0
Total program time (secs)           =       0.00958 (     0.008 CPU time)
Time spent in evaluations (secs)    =       0.00000

===============================================================================

Knitro converged with final status = 0
  optimal objective value  = -1.733333e+01
  optimal primal values x  = (3.666667e+00, 1.333333e+00, 2.941905e-10, 3.871552e-10)
  feasibility violation    = 6.998846e-13
  KKT optimality violation = 4.534322e-10
```

### References:

* [Official Knitro User's Manual](https://www.artelys.com/tools/knitro_doc/index.html)
* [Knitro / Python reference](https://www.artelys.com/docs/knitro/2_userGuide/gettingStarted/startPython.html#)
