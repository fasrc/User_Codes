### Purpose:

Program illustrates use of MPI for Python (mpi4py) on the Odyssey cluster.

### Install mpi4py:

To run this example you need to install mpi4py following the below instructions.

(1) You may first need to create a conda environment, e.g.,

```
module load python/3.6.3-fasrc02
conda create -n python3_env1 python=3.6 numpy scipy pip wheel
source activate python3_env1 
```
This creates the <code>python3_env1</code> environment with the specified packages.

(2) Next, you need to install the mpi4py module. 

```
module load gcc/8.2.0-fasrc01
module load openmpi/3.1.1-fasrc01
pip install mpi4py
```

### Contents:

* <code>mpi4py_test.py</code>: Python MPI source code.
* <code>run.sbatch</code>: Btach-job submission script to send the job to the queue.

### Python Source Code:

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

### Example Batch-Job Submission Script:

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
module load python/3.6.3-fasrc02
module load gcc/8.2.0-fasrc01
module load openmpi/3.1.1-fasrc01
source activate python3_env1

# Run the program
srun -n 16 --mpi=pmi2 python mpi4py_test.py
```

### Example Usage:

```
sbatch run.sbatch
```
 
### Example Output:

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
