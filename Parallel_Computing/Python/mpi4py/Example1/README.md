## Purpose:

Program illustrates using MPI for Python (mpi4py) on the FAS cluster at Harvard University.

## Contents:

* <code>mpi4py_test.py</code>: Python MPI source code.
* <code>run.sbatch</code>: Batch-job submission script to send the job to the queue.

## Python Source Code:

```python
#!/usr/bin/env python3
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Parallel Python test program: mpi4py  
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from mpi4py import MPI

nproc = MPI.COMM_WORLD.Get_size()   # Size of communicator 
iproc = MPI.COMM_WORLD.Get_rank()   # Ranks in communicator
inode = MPI.Get_processor_name()    # Node where this MPI process runs

if iproc == 0: print ("This code is a test for mpi4py.")

for i in range(0,nproc):
    MPI.COMM_WORLD.Barrier()
    if iproc == i:
        print ( 'Rank %d out of %d' % (iproc,nproc) )
        
MPI.Finalize()
```

## Example Batch-Job Submission Script:

If you have installed <code>mpi4py</code> with <code>conda</code> and use the default MPICH instance in the conda environment you could use the below batch-job submission script to send your job to the queue: 

```bash
#!/bin/bash
#SBATCH -J mpi4py_test
#SBATCH -o mpi4py_test.out
#SBATCH -e mpi4py_test.err
#SBATCH -p shared
#SBATCH -n 16
#SBATCH -t 30
#SBATCH --mem-per-cpu=4000

# Set up environment
module load python/3.10.12-fasrc01
source activate python3_env1

# Run the program
srun -n 16 --mpi=pmi2 python mpi4py_test.py
```

If you opted out for installing mpi4py with your MPI flavor and/or version of choice via pip as explained [here](../README.md). You will need to modify the "Set up environment" section in the above script as follows:

```bash
# Set up environment
module load python/3.10.12-fasrc01
module load gcc/12.2.0-fasrc01
module load openmpi/4.1.5-fasrc03
source activate python3_env2
```
and the "Run the program" section as follows:

```bash
srun -n 16 --mpi=pmix python mpi4py_test.py
```

**Note:** You can use as an example the included <code>run_ompi.sbatch</code> batch-job submission script for running with **OpenMPI**. Please, notice that you need to replace <code>--mpi=pmi2</code> with <code>--mpi=pmix</code> in this case.

## Example Usage:

```bash
sbatch run.sbatch
```

## Example Output:

```
> cat mpi4py_test.out
This code is a test for mpi4py.
Rank 0 out of 16
Rank 1 out of 16
Rank 2 out of 16
Rank 3 out of 16
Rank 4 out of 16
Rank 5 out of 16
Rank 6 out of 16
Rank 7 out of 16
Rank 8 out of 16
Rank 9 out of 16
Rank 10 out of 16
Rank 11 out of 16
Rank 12 out of 16
Rank 13 out of 16
Rank 14 out of 16
Rank 15 out of 16
```