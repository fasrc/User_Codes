#### PURPOSE:

Example workflow illustrating running multiple parallel MATLAB jobs with the
help of job arrays. The specific example computes the integer sum from 1 to
N where N is a number read from an input file. 

#### CONTENTS:

(1) parallel_sum.m: Parallel MATLAB code.

(2) run.sbatch: Batch job submission script for sending the array job
                to the queue.

(3) input_1, input_2, input_3: Input files containing the numbers 100, 200, and 300 respectively.
                       
#### EXAMPLE USAGE:
	source new-modules.sh
	module load matlab/R2015b-fasrc01
	sbatch run.sbatch


#### EXAMPLE OUTPUT:

```
[pkrastev@sa01 liaoshiwugmailcom]$ cat output_1.out
MATLAB is selecting SOFTWARE OPENGL rendering.

                            < M A T L A B (R) >
                  Copyright 1984-2015 The MathWorks, Inc.
                   R2015b (8.6.0.267246) 64-bit (glnxa64)
                              August 20, 2015


To get started, type one of these: helpwin, helpdesk, or demo.
For product information, visit www.mathworks.com.


        Academic License


pc =

 Local Cluster

    Properties:

                          Profile: local
                         Modified: true
                             Host: holy2a13303
                       NumWorkers: 4

               JobStorageLocation: /scratch/pkrastev/1
 RequiresMathWorksHostedLicensing: false

    Associated Jobs:

                   Number Pending: 0
                    Number Queued: 0
                   Number Running: 0
                  Number Finished: 0

Starting parallel pool (parpool) using the 'local' profile ... connected to 4 workers.

ans =

 Pool with properties:

            Connected: true
           NumWorkers: 4
              Cluster: local
        AttachedFiles: {}
          IdleTimeout: 30 minute(s) (30 minutes remaining)
          SpmdEnabled: true

Sum of numbers from 1 to 100 is 5050.
Parallel pool using the 'local' profile is shutting down.
```
