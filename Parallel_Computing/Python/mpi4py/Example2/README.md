## Purpose:

This example illustrates how to minimize a target function in parallel using the `mpi4py` package. The specific example first distributes a set of initial function values to all MPI processes, and then each MPI process minimizes an instance of the objective function for a given (local) initial value. At the end, results are gathered at the root MPI process.

## Contents:

* <code>optimize_mpi.py</code>: Python MPI source code.
* <code>run.sbatch</code>: Btach-job submission script to send the job to the queue.

### Python Source Code:

```python
#!/usr/bin/env python3
from scipy.optimize import minimize
import numpy as np
from mpi4py import MPI

# Function to minimize
def f(x):
    r = x*x
    return r

# Main program
if __name__ == '__main__':
    comm  = MPI.COMM_WORLD
    nproc = comm.Get_size() 
    iproc = comm.Get_rank()

    # Define initial data on the root MPI process
    if iproc == 0:
        data = [(i+1)**2 for i in np.arange(nproc)]
    else:
        data = None

    # Scatter data to all MPI processes
    x0 = comm.scatter(data, root=0)

    # Minimize the function
    comm.Barrier()
    res = minimize(f, x0, method='Nelder-Mead', tol=1e-8)
    x = res.x[0]

    # Gather results on the root process
    x1 = comm.gather(x, root=0)
    if iproc == 0:
        print (x1)
```

### Example Batch-Job Submission Script:

```bash
#!/bin/bash
#SBATCH -J optimize_mpi
#SBATCH -o optimize_mpi.out
#SBATCH -e optimize_mpi.err
#SBATCH -p test
#SBATCH -n 8
#SBATCH -t 30
#SBATCH --mem-per-cpu=4000

# Set up environment
module load python/3.10.9-fasrc01
source activate python3_env1

# Run the program
srun -n 8 --mpi=pmi2 python optimize_mpi.py
```

If you opted out for installing mpi4py with your MPI flavor and/or version of choice via pip as explained [here](../README.md). You will need to modify the "Set up environment" section in the above script as follows:

```bash
# Set up environment
module load python/3.10.9-fasrc01
module load gcc/12.2.0-fasrc01
module load openmpi/4.1.5-fasrc01
source activate python3_env2
```
and the "Run the program" section as follows:

```bash
srun -n 8 --mpi=pmix python optimize_mpi.py
```

**Note:** You can use as an example the included <code>run_ompi.sbatch</code> batch-job submission script for running with **OpenMPI**. Please, notice that you need to replace <code>--mpi=pmi2</code> with <code>--mpi=pmix</code> in this case.

### Example Usage:

```
sbatch run.sbatch
```

### Example Output:

```bash
(python3_env1) [pkrastev@holy2a01301 Example2]$ cat optimize_mpi.out 
[-8.881784197001252e-16, -3.552713678800501e-15, -2.1316282072803006e-14, -1.4210854715202004e-14, 0.0, -8.526512829121202e-14, -5.684341886080802e-14, -5.684341886080802e-14]
```
