### Purpose:

Program illustrates solving a 1D time independent Schrodinger equation (SE) 
using a Finite Difference Method. The specific example solves the SE for 4
different energies (Ev = [-295, -200, -150, -100]) in parallel with the MATLAB 
Parallel Computing Toolbox (PCT).

### Contents:

(1) se_fdm.m: Function solving a 1D time independent Schrodinger equation.

(2) simpson1d.m: 1D Simpson integration.

(3) prun.m: Parallel driver.

(2) run.sbatch: Batch-job submission script to send the job to the queue.

### Example Usage:

	source new-modules.sh
	module load matlab/R2016a-fasrc01
	sbatch run.sbatch
    
### Example Output:

```
                            < M A T L A B (R) >
                  Copyright 1984-2016 The MathWorks, Inc.
                   R2016a (9.0.0.341360) 64-bit (glnxa64)
                             February 11, 2016


To get started, type one of these: helpwin, helpdesk, or demo.
For product information, visit www.mathworks.com.


        Academic License


pc =

 Local Cluster

    Properties:

                          Profile: local
                         Modified: false
                             Host: holy2a02103
                       NumWorkers: 4

               JobStorageLocation: /n/home06/pkrastev/.matlab/local_cluster_jobs/R2016a
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

Prosess 4:
Prosess 3:
Prosess 2:
Prosess 1:
Ev = -100
Number of crossing for Psi = 3
End value of Psi  =  -3.2e+05


Ev = -150
Number of crossing for Psi = 3
End value of Psi  =  -3.46e+05


Ev = -200
Number of crossing for Psi = 2
End value of Psi  =  3.73e+05


Ev = -295
Number of crossing for Psi = 2
End value of Psi  =  2.29e+05


Parallel pool using the 'local' profile is shutting down.
```

### Example Output Figures:

![Process 1:]
(fig_1.png)

![Process 2:]
(fig_2.png)

![Process 3:]
(fig_3.png)

![Process 4:]
(fig_4.png)

