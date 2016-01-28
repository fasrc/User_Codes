###PURPOSE:

Example illustrating calling  C++ functions that use the GSL numerical
library from MATLAB. The specific example first creates a binary MEX-file
from a C++ source MEX-file, and then computes the value of the Bessel
function J_0(x) for x=5.

###Contents:

(1) bessel_test.cpp: C++ source MEX-file

(2) Makefile: Makefile to compile the C++ code

(3) mex_test.m: MATLAB source code using the compiled MEX file 

(4) mex_test.sbatch: SLURM batch-jobs submission scrip for the Odysswy cluster

###Example Usage:
	source new-modules.sh
	module load matlab/R2015a-fasrc01
	module load gcc/4.7.4-fasrc01
	module load gsl/2.1-fasrc02
	make
	sbatch mex_test.sbatch
