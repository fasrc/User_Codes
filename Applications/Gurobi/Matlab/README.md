### Using Gurobi with Matlab

This section will work through a simple example in order to illustrate the use of the Gurobi MATLAB API. The example builds a simple Mixed Integer Programming model, solves it, and prints the optimal solution. We use the`mip1.m` example provided with the [Gurobi distribution](https://docs.gurobi.com/projects/examples/en/current/examples/matlab/mip1.html).

>**Note:** Gurobi works with these modules:
```module load matlab/R2025b-fasrc01 gurobi/12.0.1-fasrc01```


### Contents:

* <code>mip1.m</code>: Matlab source code
* <code>run.sbatch</code>: Job submission script for the Matlab example
* <code>gurobi_test.out</code>: STD output from the Matlab example

The below example illustrates using Gurobi with its MATLAB interface in a batch mode on the [FAS Cannon cluster](https://www.rc.fas.harvard.edu/about/cluster-architecture) at Harvard University. 

### Matlab source code:

```matlab
function mip1()
% Copyright 2026, Gurobi Optimization, LLC
% This example formulates and solves the following simple MIP model:
%  maximize
%        x +   y + 2 z
%  subject to
%        x + 2 y + 3 z <= 4
%        x +   y       >= 1
%        x, y, z binary

names = {'x'; 'y'; 'z'};

model.A = sparse([1 2 3; 1 1 0]);
model.obj = [1 1 2];
model.rhs = [4; 1];
model.sense = '<>';
model.vtype = 'B';
model.modelsense = 'max';
model.varnames = names;

gurobi_write(model, 'mip1.lp');

params.outputflag = 0;

result = gurobi(model, params);

disp(result);

for v=1:length(names)
    fprintf('%s %d\n', names{v}, result.x(v));
end

fprintf('Obj: %e\n', result.objval);
end
```

### Example batch-job submission script:

```bash
#!/bin/bash
#SBATCH -J gurobi_test          # job name
#SBATCH -o gurobi_test.out      # standard output file
#SBATCH -e gurobi_test.err      # standard error file
#SBATCH -p test                 # partition
#SBATCH -c 1                    # number of cores
#SBATCH -t 0-00:30              # time in D-HH:MM
#SBATCH --mem=4000              # memory in MB

# --- Load required software modules ---
module load matlab/R2025b-fasrc01 gurobi/12.0.1-fasrc01

# --- Run the Matlab code ---
srun -c $SLURM_CPUS_PER_TASK matlab -nosplash -nodesktop -nodisplay -r "mip1; exit;"
```

### Example usage:

```bash
sbatch run.sbatch
```

### Example output:

Upon the job completion the results will be in the <code>gurobi_test.out</code> file.

```bash
$ cat gurobi_test.out

                            < M A T L A B (R) >
                  Copyright 1984-2025 The MathWorks, Inc.
             R2025b Update 1 (25.2.0.3042426) 64-bit (glnxa64)
                              October 3, 2025

 
To get started, type doc.
For product information, visit www.mathworks.com.
 
          status: 'OPTIMAL'
     versioninfo: [1×1 struct]
         runtime: 0.0188
            work: 1.3061e-05
         memused: 3.8800e-06
      maxmemused: 3.2860e-04
          objval: 3
               x: [3×1 double]
           slack: [2×1 double]
    poolobjbound: 3
            pool: [1×2 struct]
          mipgap: 0
        objbound: 3
       objboundc: 3
       itercount: 0
    baritercount: 0
       nodecount: 0
          maxvio: 0

x 1
y 0
z 1
Obj: 3.000000e+00
``` 

### References:

* [Gurobi / MATLAB Reference](https://docs.gurobi.com/projects/optimizer/en/current/reference/matlab.html)
* [Tutorial](https://support.gurobi.com/hc/en-us/articles/17307706649617-Tutorial-Getting-Started-with-the-Gurobi-MATLAB-API)
