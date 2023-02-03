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

