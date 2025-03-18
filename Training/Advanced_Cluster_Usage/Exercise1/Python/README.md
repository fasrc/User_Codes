# Exercise 1: Job Efficiency - Memory per Node ( `--mem` )

We use a Python code, `mem_test.py`, to generate a random matrix of dimension 30,000.  

## Step 1: Environment set up
To run the code we need to load one of the available Python modules, e.g.,

```bash
module load python/3.12.8-fasrc01
```

 The Python source code used in this example  is included  below:

```python
import random

def create_symmetric_matrix(n):
    """
    Creates a symmetric random matrix of size n x n using pure Python.
    """
    # Initialize an n x n matrix filled with zeros
    matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    
    # Fill the matrix with random numbers and ensure symmetry
    for i in range(n):
        for j in range(i + 1):  # Only fill the lower triangular part
            value = random.random()  # Random float between 0.0 and 1.0
            matrix[i][j] = value
            matrix[j][i] = value  # Symmetric element
    
    return matrix

def main():
    n = 30000  # Matrix dimension
    
    print(f"Creating a symmetric random matrix of size {n}x{n}...")
    
    # Create the symmetric matrix
    h = create_symmetric_matrix(n)
    
    print("Hamiltonian matrix created successfully!")
    
    # Print a small portion of the matrix (optional)
    print("Top-left 5x5 corner of the matrix:")
    for row in h[:5]:
        print(row[:5])

if __name__ == "__main__":
    main()
```

## Step 2: Create a job submission  script

The below job-submission script intentionally requests less memory than what the job
actually needs:

```bash
#!/bin/bash
#SBATCH -p test
#SBATCH -J mem_test
#SBATCH -o mem_test.out
#SBATCH -e mem_test.err
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 20
#SBATCH --mem=20G 

# Load required modules
module load python/3.12.8-fasrc01

# Run the code
srun -n 1 -c 1 python mem_test.py
```

## Step 3: Submit the Job

If the job-submission script is named `run_python.sbatch`, for instance, the job 
is submitted to the queue with:

```bash
sbatch run_python.sbatch
```
>**NOTE:** The job should fail due to insufficient memory. 

## Step 4: Diagnose the Issue

You can check the job status with:

```bash
sacct -j 3851291
JobID           JobName  Partition    Account  AllocCPUS      State ExitCode 
------------ ---------- ---------- ---------- ---------- ---------- -------- 
3851291        mem_test       test   rc_admin          1     FAILED      1:0 
3851291.bat+      batch              rc_admin          1     FAILED      1:0 
3851291.ext+     extern              rc_admin          1  COMPLETED      0:0 
3851291.0        python              rc_admin          1 OUT_OF_ME+    0:125 
```

You can also check the STD error file with:

```bash
cat mem_test.err 
slurmstepd: error: Detected 1 oom_kill event in StepId=3851291.0. Some of the step tasks have been OOM Killed.
srun: error: holy8a26601: task 0: Out Of Memory
```

## Step 5: Adjust the Memory Request and Resubmit the Job

Modify the job-submission script to request more memory, e.g., double the memory,

```bash
#SBATCH --mem=40G  # Double the original memory request 
```

and resubmit the job:

```bash
sbatch run_python.sbatch
Submitted batch job 3852229
```

>**NOTE:** This time the job should complete successfully.

## Step 6: Check the Job Status and Memory Efficiency

First, check the job status, e.g.,
```bash
sacct -j 3852229
JobID           JobName  Partition    Account  AllocCPUS      State ExitCode 
------------ ---------- ---------- ---------- ---------- ---------- -------- 
3852229        mem_test       test   rc_admin          1  COMPLETED      0:0 
3852229.bat+      batch              rc_admin          1  COMPLETED      0:0 
3852229.ext+     extern              rc_admin          1  COMPLETED      0:0 
3852229.0        python              rc_admin          1  COMPLETED      0:0 
```
You can also check the STD output, e.g.,

```bash
cat mem_test.out 
Creating a symmetric random matrix of size 30000x30000...
Hamiltonian matrix created successfully!
Top-left 5x5 corner of the matrix:
[0.20198983178230456, 0.38972129034631353, 0.4448011747040722, 0.014762384864909328, 0.5207878870066227]
[0.38972129034631353, 0.5804732716534929, 0.9279578334633064, 0.3665711283159755, 0.4736537334788029]
[0.4448011747040722, 0.9279578334633064, 0.27196057633304604, 0.06226103290658924, 0.4203840596810722]
[0.014762384864909328, 0.3665711283159755, 0.06226103290658924, 0.3781459667385185, 0.6048320364367925]
[0.5207878870066227, 0.4736537334788029, 0.4203840596810722, 0.6048320364367925, 0.8051804647208598]
```

Second, check the memory efficiency with the `seff` command:

```bash
seff 3852229
Job ID: 3852229
Cluster: odyssey
User/Group: pkrastev/rc_admin
State: COMPLETED (exit code 0)
Cores: 1
CPU Utilized: 00:01:56
CPU Efficiency: 100.00% of 00:01:56 core-walltime
Job Wall-clock time: 00:01:56
Memory Utilized: 18.90 GB
Memory Efficiency: 47.24% of 40.00 GB (40.00 GB/node)
```

The Memory Efficiency is about 47%. The job used 18.90 GB while the requested memory 
is 40 GB. Adjust the requested memory so that the efficiency
is at least 80%, and resubmit the job, e.g.,

```bash
#SBATCH --mem=21G
```

Submit the job with the updated batch-job submission script,

```bash
sbatch run.sbatch
Submitted batch job 852752
```

check the job status,

```bash
sacct -j 3852752
JobID           JobName  Partition    Account  AllocCPUS      State ExitCode 
------------ ---------- ---------- ---------- ---------- ---------- -------- 
3852752        mem_test       test   rc_admin          1  COMPLETED      0:0 
3852752.bat+      batch              rc_admin          1  COMPLETED      0:0 
3852752.ext+     extern              rc_admin          1  COMPLETED      0:0 
3852752.0        python              rc_admin          1  COMPLETED      0:0 
```

and memory efficiency:

```bash
seff 3852752
Job ID: 3852752
Cluster: odyssey
User/Group: pkrastev/rc_admin
State: COMPLETED (exit code 0)
Cores: 1
CPU Utilized: 00:01:53
CPU Efficiency: 98.26% of 00:01:55 core-walltime
Job Wall-clock time: 00:01:55
Memory Utilized: 19.20 GB
Memory Efficiency: 91.45% of 21.00 GB (21.00 GB/node)
```

We see that the Memory Efficiency is ~91%, while the CPU Efficiency is 98.26%.
