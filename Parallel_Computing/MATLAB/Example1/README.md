### Purpose:

Program computes PI with parallel Monte-Carlo mathod.

### Contents:

(1) parallel_monte_carlo.m: MATLAB source code

(2) run.sbatch: Batch-job submission script to send the job to the queue.

### Example Usage:

	source new-modules.sh
	module load matlab/R2016a-fasrc01
	sbatch run.sbatch
    
### Example Output:

```
MATLAB is selecting SOFTWARE OPENGL rendering.

                            < M A T L A B (R) >
                  Copyright 1984-2015 The MathWorks, Inc.
                   R2015b (8.6.0.267246) 64-bit (glnxa64)
                              August 20, 2015

 
To get started, type one of these: helpwin, helpdesk, or demo.
For product information, visit www.mathworks.com.
 

	Academic License

Starting parallel pool (parpool) using the 'local' profile ... connected to 8 workers.

ans = 

 Pool with properties: 

            Connected: true
           NumWorkers: 8
              Cluster: local
        AttachedFiles: {}
          IdleTimeout: 30 minute(s) (30 minutes remaining)
          SpmdEnabled: true

The computed value of pi is 3.1409472.
The parallel Monte-Carlo method is executed in    33.86 seconds.
Parallel pool using the 'local' profile is shutting down.
```

