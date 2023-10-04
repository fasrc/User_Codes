### Purpose:

Program computes PI with parallel Monte-Carlo method.

### Contents:

* `parallel_pi.m`: Mathematica source code
* `run.sbatch`: Btach-job submission script to send the job to the queue.

### Source code:

```mathematica
(* Compute PI in via parallel Monte-Carlo method *)
tstart = AbsoluteTime[];
Print [ "Parallel calculation of PI via Monte-Carlo method." ];

nproc = 8;
LaunchKernels[nproc];

Print[ " Number of kernels: ", nproc];

n = 10^8;
m = n / nproc;
Print [ " Total number of hits: ", n ];
Print [ " Number of hits per core: ", m ];

acceptpoint[j_] := Total[Table[ 1 - Floor[ Sqrt[ (Random[])^2 + (Random[])^2 ] ], {i,1,m} ] ];
DistributeDefinitions[n,m,acceptpoint];
t1 = AbsoluteTime[];
hits = ParallelTable[acceptpoint[j], {j,1,nproc} ];
t2 = AbsoluteTime[];
tt = t2 - t1;
hits = Total[ hits ];
pi = hits / ( nproc * m ) * 4.0;
Print [ " Computed PI = ", pi ];
Print [ " Time in parallel calculation: ", tt ] ;
tend = AbsoluteTime[];
ttotal = tend - tstart;
Print [ " Total time: ", ttotal  ];
Quit [ ]
```

### Example Batch-Job Submission Script:

```bash
#!/bin/bash
#SBATCH -J parallel_pi
#SBATCH -o parallel_pi.out
#SBATCH -e parallel_pi.err
#SBATCH -p test
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -t 0-00:30
#SBATCH --mem=4000

# Load required software modules
module load mathematica/13.3.0-fasrc01

# Run program
srun -n 1 -c 8 math -script parallel_pi.m
```

### Example Usage:

```bash
sbatch run.sbatch
```
 
### Example Output:

```
Parallel calculation of PI via Monte-Carlo method.
Updating from Wolfram Research server ...
 Number of kernels: 8
 Total number of hits: 100000000
 Number of hits per core: 12500000
 Computed PI = 3.1410542
 Time in parallel calculation: 2.467308`6.8437683603959725
 Total time: 9.950124`7.449373486521115
```
