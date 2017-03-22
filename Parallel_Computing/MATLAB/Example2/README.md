### Purpose:

Program illustrates using implicit multi-threading capabilities of MATLAB. Specific
example performs matrix-matrix multilication with increasing number of threads (1, 2, 4, 8)
and computes speedup and efficiency.

### Contents:

(1) thread_test.m: MATLAB source code

(2) run.sbatch: Batch-job submission script to send the job to the queue.

### Example Usage:

	source new-modules.sh
	module load matlab/R2016a-fasrc01
	sbatch run.sbatch
    
### Example Output:

```

                                                                            < M A T L A B (R) >
                                                                  Copyright 1984-2015 The MathWorks, Inc.
                                                                  R2015b (8.6.0.267246) 64-bit (glnxa64)
                                                                              August 20, 2015

 
To get started, type one of these: helpwin, helpdesk, or demo.
For product information, visit www.mathworks.com.
 

	Academic License

Number of threads: 1
Number of threads: 2
Number of threads: 4
Number of threads: 8

     Nproc  Walltime  Speedup  Efficiency (%)
       1     28.33      1.00      100.00
       2     14.27      1.99       99.29
       4      7.84      3.61       90.35
       8      4.95      5.72       71.55
```


### Speedup

![Speedup](/speedup.png)

### REFERENCES:

[Multi-threading examples](http://www.bu.edu/tech/support/research/training-consulting/online-tutorials/matlab-pct/implicit-parallelism)

