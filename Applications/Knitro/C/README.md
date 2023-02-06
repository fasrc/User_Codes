### Using Knitro with C:

The Knitro callable library is used to build a model in pieces while providing special structures to Knitro (e.g. linear structures, quadratic structures), while providing callbacks to handle general, nonlinear structures. Below is a quick example illustrating the use of Knitro in C.

### Contents:

* <code>knitro_test.c</code>: Knitro / C source code
* <code>knitro.opt</code>: Options file
* <code>knitro_test.out</code>: STD output
* <code>run.sbatch</code>: Batch-job submission script

### Example C source code:

```c
#include <stdio.h>
#include <stdlib.h>
#include "knitro.h"

/* main */
int  main (int  argc, char  *argv[]) {
  int  i, nStatus, error;

  /** Declare variables. */
  KN_context   *kc;
  int    n, m;
  double x[3];
  double xLoBnds[3] = {0, 0, 0};
  double xInitVals[3] = {2.0, 2.0, 2.0};
  /** Used to define linear constraint. */
  int    lconIndexVars[3] = {  0,    1,   2};
  double lconCoefs[3]     = {8.0, 14.0, 7.0};
  /** Used to specify quadratic constraint. */
  int    qconIndexVars1[3] = {  0,   1,   2};
  int    qconIndexVars2[3] = {  0,   1,   2};
  double qconCoefs[3]      = {1.0, 1.0, 1.0};
  /** Used to specify quadratic objective terms. */
  int    qobjIndexVars1[5] = {   0,    1,    2,    0,    0};
  int    qobjIndexVars2[5] = {   0,    1,    2,    1,    2};
  double qobjCoefs[5]      = {-1.0, -2.0, -1.0, -1.0, -1.0};
  /** Solution information */
  double objSol;
  double feasError, optError;

  /** Create a new Knitro solver instance. */
  error = KN_new(&kc);
  if (error) exit(-1);
  if (kc == NULL)
    {
      printf ("Failed to find a valid license.\n");
      return( -1 );
    }

  /** Illustrate how to override default options by reading from
   *  the knitro.opt file. */
  error = KN_load_param_file (kc, "knitro.opt");
  if (error) exit(-1);

  /** Initialize Knitro with the problem definition. */

  /** Add the variables and set their bounds and initial values.
   *  Note: unset bounds assumed to be infinite. */
  n = 3;
  error = KN_add_vars(kc, n, NULL);
  if (error) exit(-1);
  error = KN_set_var_lobnds_all(kc, xLoBnds);
  if (error) exit(-1);
  error = KN_set_var_primal_init_values_all(kc, xInitVals);
  if (error) exit(-1);

  /** Add the constraints and set their bounds. */
  m = 2;
  error = KN_add_cons(kc, m, NULL);
  if (error) exit(-1);
  error = KN_set_con_eqbnd(kc, 0, 56.0);
  if (error) exit(-1);
  error = KN_set_con_lobnd(kc, 1, 25.0);
  if (error) exit(-1);

  /** Add coefficients for linear constraint. */
  error = KN_add_con_linear_struct_one (kc, 3, 0, lconIndexVars,
					lconCoefs);
  if (error) exit(-1);

  /** Add coefficients for quadratic constraint */
  error = KN_add_con_quadratic_struct_one (kc, 3, 1, qconIndexVars1,
					   qconIndexVars2, qconCoefs);
  if (error) exit(-1);

  /** Set minimize or maximize (if not set, assumed minimize) */
  error = KN_set_obj_goal(kc, KN_OBJGOAL_MINIMIZE);
  if (error) exit(-1);

  /** Add constant value to the objective. */
  error= KN_add_obj_constant(kc, 1000.0);
  if (error) exit(-1);

  /** Set quadratic objective structure. */
  error = KN_add_obj_quadratic_struct (kc, 5, qobjIndexVars1,
				       qobjIndexVars2, qobjCoefs);
  if (error) exit(-1);

  /** Solve the problem.
   *
   *  Return status codes are defined in "knitro.h" and described
   *  in the Knitro manual. */
  nStatus = KN_solve (kc);

  printf ("\n\n");
  printf ("Knitro converged with final status = %d\n",
	  nStatus);

  /** An example of obtaining solution information. */
  error = KN_get_solution(kc, &nStatus, &objSol, x, NULL);
  if (!error) {
    printf ("  optimal objective value  = %e\n", objSol);
    printf ("  optimal primal values x  = (%e, %e, %e)\n", x[0], x[1], x[2]);
  }
  error = KN_get_abs_feas_error (kc, &feasError);
  if (!error)
    printf ("  feasibility violation    = %e\n", feasError);
  error = KN_get_abs_opt_error (kc, &optError);
  if (!error)
    printf ("  KKT optimality violation = %e\n", optError);

  /** Delete the Knitro solver instance. */
  KN_free (&kc);

  return( 0 );
}
```
### Compile the source code:

The below commands load the <code>knitro</code> software module, and compile and link to the <code>knitro</code> dynamic library.

```bash
module load knitro/13.2.0-fasrc01
gcc -o knitro_test.x knitro_test.c -lm -lknitro -O2
```

### Example batch-job submission script:

```bash
#!/usr/bin/env bash
#SBATCH -J knitro_test
#SBATCH -o knitro_test.out
#SBATCH -e knitro_test.err
#SBATCH -p test
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 0-00:30
#SBATCH --mem=2G

# Load required software modules
module load knitro/13.2.0-fasrc01

# Run program
./knitro_test.x
```

### Example usage:

If the above submission script is named <code>run.sbatch</code>, for instance, it is sent to the queue with:

```bash
sbatch run.sbatch
```

### Example output:

Upon completion the results will be in the file <code>knitro_test.out</code>.

```bash
$ cat knitro_test.out 

=======================================
           Academic License
       (NOT FOR COMMERCIAL USE)
         Artelys Knitro 13.2.0
=======================================

Knitro presolve eliminated 0 variables and 0 constraints.

The problem is identified as a QCQP.

Problem Characteristics                                 (   Presolved)
-----------------------
Objective goal:  Minimize
Objective type:  quadratic
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
    quadratic one-sided inequalities:                 1 (           1)
    gen. nonlinear one-sided inequalities:            0 (           0)
    linear two-sided inequalities:                    0 (           0)
    quadratic two-sided inequalities:                 0 (           0)
    gen. nonlinear two-sided inequalities:            0 (           0)
Number of nonzeros in Jacobian:                       6 (           6)
Number of nonzeros in Hessian:                        5 (           5)

Knitro using the Interior-Point/Barrier Direct algorithm.

  Iter      Objective      FeasError   OptError    ||Step||    CGits 
--------  --------------  ----------  ----------  ----------  -------
       0    9.760000e+02   1.300e+01
      10    9.360000e+02   0.000e+00   1.250e-09   4.838e-08        0

EXIT: Locally optimal solution found.

HINT: Knitro spent   2.3% of solution time (0.000108 secs) checking model
      convexity. To skip the automatic convexity checker for QPs and QCQPs,
      explicity set the user option convex=0 or convex=1.

Final Statistics
----------------
Final objective value               =   9.36000000020340e+02
Final feasibility error (abs / rel) =   0.00e+00 / 0.00e+00
Final optimality error  (abs / rel) =   1.25e-09 / 7.81e-11
# of iterations                     =         10 
# of CG iterations                  =          2 
# of function evaluations           =          0
# of gradient evaluations           =          0
# of Hessian evaluations            =          0
Total program time (secs)           =       0.00471 (     0.004 CPU time)
Time spent in evaluations (secs)    =       0.00000

===============================================================================



Knitro converged with final status = 0
  optimal objective value  = 9.360000e+02
  optimal primal values x  = (1.034193e-09, 3.031926e-10, 8.000000e+00)
  feasibility violation    = 0.000000e+00
  KKT optimality violation = 1.249997e-09
```

### References:

* [Getting started with the callable library](https://www.artelys.com/docs/knitro/2_userGuide/gettingStarted/startCallableLibrary.html)