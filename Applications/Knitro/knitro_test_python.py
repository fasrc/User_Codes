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
