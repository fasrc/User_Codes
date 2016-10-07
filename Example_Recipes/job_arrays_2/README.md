#### PURPOSE:

Example workflow illustrating running multiple jobs with the
help of job arrays. The specific example computes the integer sum from 1 to
N where N is a number read from the command line. 

#### CONTENTS:

(1) main_program.cpp: C++ source code

(2) run.sbatch: Batch job submission script for sending the array job
                to the queue.

(3) Makefile

#### EXAMPLE USAGE:
	sbatch run.sbatch

#### EXAMPLE OUTPUT:

```
./pro.x 100
Sum of 1 to 100 is 5050.
End of program.
```
