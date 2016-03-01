/*
 * Program: bessel_test.cpp
 *
 * Illustrates calling a C function that uses the GSL numeric library
 *
 *  Computational function that takes a scalar and computes the value 
 * of the Bessel function J_0(x)
 * 
 *
 * This is a MEX-file for MATLAB
 */
#include "mex.h"
#include <iostream>
#include <gsl/gsl_sf_bessel.h>
using namespace std;

// Computational routine.............................................
void bessel_test( double x[] , double y[] ){
  y[0] = gsl_sf_bessel_J0 (x[0]);
}

// The gateway function..............................................
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[]){

  double *x;
  double *y;
  size_t mrows;
  size_t ncols;

  // Get number of rows and coulums of the input matrix...............
  mrows = mxGetM(prhs[0]);
  ncols = mxGetN(prhs[0]);

  // Check for proper number of arguments.............................
  if(nrhs!=1) {
    mexErrMsgIdAndTxt( "MATLAB:bessel_test:invalidNumInputs",
            "One input required.");
  } else if(nlhs>1) {
    mexErrMsgIdAndTxt( "MATLAB:bessel_test:maxlhs",
            "Too many output arguments.");
  }
  
  // The input must be a noncomplex scalar double.....................
  if( !mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]) ||
      !(mrows==1 && ncols==1) ) {
    mexErrMsgIdAndTxt( "MATLAB:bessel_test:inputNotRealScalarDouble",
            "Input must be a noncomplex scalar double.");
  }

  // Create matrix for the return argument............................
  plhs[0] = mxCreateDoubleMatrix((mwSize)mrows, (mwSize)ncols, mxREAL);

  // Assign pointers to each input and output.........................
  x = mxGetPr(prhs[0]);
  y = mxGetPr(plhs[0]);
  
  bessel_test(x,y);

}
