#### Purpose:

Example of using GSL libraries on the cluster. The specific example solves the integral

$$\int_0^1 x^{-1/2} log(x) dx = -4$$

to a relative accuracy bound of 1e-7. 

#### Contents:

(1) gsl_int_test.c: C source file

(2) Makefile: Makefile to compile the source code 

(3) gsl_int_test.sbatch: Btach-job submission script to send the job to the queue

#### Example Usage:

	source new-modules.sh
	module load gsl/1.16-fasrc02
	make
	sbatch gsl_int_test.sbatch

#### Example Output:

```
result          = -4.000000000000085265
exact result    = -4.000000000000000000
estimated error =  0.000000000000135447
actual error    = -0.000000000000085265
intervals       = 8
```
