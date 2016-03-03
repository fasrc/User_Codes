### Purpose:

Program illustrates use of MPI-IO. It generates 3 random vectors of dimension 20 and
writes them to disk with MPI IO. Then, it reads them from disk with MPI IO and prints 
them out to screen.

### Contents:

(1) mpi_IO_test1.f90: Fortran source code

(2) Makefile: Makefile to compile the source code

(3) run.sbatch: Btach-job submission script to send the job to the queue.

### Example Usage:

	source new-modules.sh
	module load intel/15.0.0-fasrc01
	module load openmpi/1.8.3-fasrc02
	make
	sbatch run.sbatch
    
### Example Output:

```
Vector  1:
  1    0.2982
  2    0.7151
  3    0.0330
  4    0.8744
  5    0.5342
  6    0.6316
  7    0.8910
  8    0.2575
  9    0.9316
 10    0.2772
 11    0.7158
 12    0.4834
 13    0.5311
 14    0.1834
 15    0.2713
 16    0.6031
 17    0.8334
 18    0.2281
 19    0.6684
 20    0.5291
Vector  2:
  1    0.5342
  2    0.1522
  3    0.0801
  4    0.5339
  5    0.7612
  6    0.7909
  7    0.6761
  8    0.3839
  9    0.2481
 10    0.7321
 11    0.1342
 12    0.5210
 13    0.3486
 14    0.9983
 15    0.3846
 16    0.8159
 17    0.6175
 18    0.7962
 19    0.9339
 20    0.0321
Vector  3:
  1    0.9934
  2    0.9222
  3    0.9429
  4    0.0703
  5    0.0667
  6    0.8935
  7    0.8314
  8    0.0901
  9    0.1268
 10    0.6222
 11    0.0295
 12    0.8502
 13    0.9582
 14    0.7396
 15    0.4929
 16    0.7772
 17    0.3665
 18    0.0348
 19    0.4898
 20    0.7183
```
