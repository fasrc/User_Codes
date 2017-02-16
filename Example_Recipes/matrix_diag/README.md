### Purpose:

Example of using MKL libraries on the cluster. The specific example creates 3000X3000 random matrix and diagonalizes it.
The program prints out the first 10 eigen values.

### Contents:

(1) matrix_diag.f90: Fortran source code

(2) Makefile: Makefile to compile the source code

(3) run.sbatch: Btach-job submission script to send the job to the queue

### Example Usage:

	source new-modules.sh
	module load intel/15.0.0-fasrc01
	module load intel-mkl/11.0.0.079-fasrc02
	make
	sbatch run.sbatch
    
NOTE: To change matrix dimension adjust the parameter n (line 11) in matrix_diag.f90 and recompile the code
by "make clean; make".

### Example Output:

```
 First M eigen values of h:
           1  -31.5535401215023     
           2  -31.3893450362036     
           3  -31.3287443137241     
           4  -31.1211986413599     
           5  -31.0084719275123     
           6  -30.9443787874679     
           7  -30.9043033720232     
           8  -30.8070563099242     
           9  -30.7319332144362     
          10  -30.6680361297837
```
