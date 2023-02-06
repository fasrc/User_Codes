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
