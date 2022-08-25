### Purpose:

Program computes PI with parallel Monte-Carlo method directly on the cluster (i.e., when you ssh to the cluster) and on [VDI](vdi.rc.fas.harvard.edu).

### Contents:

(1) parallel_monte_carlo.m: MATLAB source code

(2) run.sbatch: Batch-job submission script to send the job to the queue.

### Example Usage directly on the cluster:

	module load matlab/R2022a-fasrc01
	sbatch run.sbatch

### Example Output directly on the cluster:

```
MATLAB is selecting SOFTWARE OPENGL rendering.

                            < M A T L A B (R) >
                  Copyright 1984-2022 The MathWorks, Inc.
                  R2022a (9.12.0.1884302) 64-bit (glnxa64)
                             February 16, 2022


To get started, type doc.
For product information, visit www.mathworks.com.

Starting parallel pool (parpool) using the 'local' profile ...
Connected to the parallel pool (number of workers: 8).

ans =

 ProcessPool with properties:

            Connected: true
           NumWorkers: 8
                 Busy: false
              Cluster: local
        AttachedFiles: {}
    AutoAddClientPath: true
            FileStore: [1x1 parallel.FileStore]
           ValueStore: [1x1 parallel.ValueStore]
          IdleTimeout: 30 minutes (30 minutes remaining)
          SpmdEnabled: true

The computed value of pi is 3.1408536.
The parallel Monte-Carlo method is executed in     0.85 seconds.
Parallel pool using the 'local' profile is shutting down.
```

### Example Usage on VDI

On VDI, you can use the same `parallel_monte_carlo.m` code by using the command (note: for 8 cores, you need ~12 GB of memory):

```
parallel_monte_carlo(str2num(getenv('SLURM_CPUS_PER_TASK')))
```

### Example Output on VDI:

```matlab
Starting parallel pool (parpool) using the 'local' profile ...
Connected to the parallel pool (number of workers: 8).

ans = 

 ProcessPool with properties:

            Connected: true
           NumWorkers: 8
                 Busy: false
              Cluster: local
        AttachedFiles: {}
    AutoAddClientPath: true
            FileStore: [1x1 parallel.FileStore]
           ValueStore: [1x1 parallel.ValueStore]
          IdleTimeout: 30 minutes (30 minutes remaining)
          SpmdEnabled: true

The computed value of pi is 3.1407408.
The parallel Monte-Carlo method is executed in     0.89 seconds.
Parallel pool using the 'local' profile is shutting down.
```

