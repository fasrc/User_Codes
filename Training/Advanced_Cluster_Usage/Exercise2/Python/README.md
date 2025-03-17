# Exercise 2: Job Efficiency - Memory per CPU/core ( `--mem-per-cpu` )

We use a Python code, `mp_mem_test.py`, to generate a random matrix of dimension 30,000 in parallel via the `threading` Python package. The specific example uses 2 threads.  

## Step 1: Environment set up
To run the code we need to load one of the available Python modules, e.g.,

```bash
module load python/3.12.8-fasrc01
```

 The Python source code used in this example  is included  below:

```python
import random
import threading

def create_symmetric_matrix_parallel(n, num_threads=2):
    """
    Creates a symmetric random matrix of size n x n using Python threading.
    """
    matrix = [[0.0 for _ in range(n)] for _ in range(n)]

    def worker(start_row, end_row):
        for i in range(start_row, end_row):
            for j in range(i + 1):
                value = random.random()
                matrix[i][j] = value
                matrix[j][i] = value

    # Calculate rows for each thread
    rows_per_thread = n // num_threads
    threads = []

    for i in range(num_threads):
        start_row = i * rows_per_thread
        # Ensure the last thread covers any remaining rows
        end_row = (start_row + rows_per_thread) if i < num_threads - 1 else n
        thread = threading.Thread(target=worker, args=(start_row, end_row))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return matrix

def main():
    n = 30000  # Matrix dimension

    print(f"Creating a symmetric random matrix of size {n}x{n} using 2 threads...")

    h = create_symmetric_matrix_parallel(n, num_threads=2)

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
#SBATCH -J mp_mem_test
#SBATCH -o mp_mem_test.out
#SBATCH -e mp_mem_test.err
#SBATCH -N 1
#SBATCH -c 2
#SBATCH -t 30
#SBATCH --mem-per-cpu=10G 

# --- Load required modules ---
module load python/3.12.8-fasrc01

# --- Run the code ---
srun -c ${SLURM_CPUS_PER_TASK} python mp_mem_test.py
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
cat mp_mem_test.err
slurmstepd: error: Detected 1 oom_kill event in StepId=7077405.0. Some of the step tasks have been OOM Killed.
srun: error: holy8a24101: task 0: Out Of Memory
```

## Step 5: Adjust the Memory Request and Resubmit the Job

Modify the job-submission script to request more memory, e.g., double the memory,

```bash
#SBATCH --mem-per-cpu=20G  # Double the original memory request 
```

and resubmit the job:

```bash
sbatch run.sbatch
Submitted batch job 7077928
```

>**NOTE:** This time the job should complete successfully.

## Step 6: Check the Job Status and Memory Efficiency

First, check the job status, e.g.,
```bash
sacct -j 7077928
JobID           JobName  Partition    Account  AllocCPUS      State ExitCode 
------------ ---------- ---------- ---------- ---------- ---------- -------- 
7077928      mp_mem_te+       test   rc_admin          2  COMPLETED      0:0 
7077928.bat+      batch              rc_admin          2  COMPLETED      0:0 
7077928.ext+     extern              rc_admin          2  COMPLETED      0:0 
7077928.0        python              rc_admin          2  COMPLETED      0:0
```
You can also check the STD output, e.g.,

```bash
cat mp_mem_test.out 
Creating a symmetric random matrix of size 30000x30000 using 2 threads...
Hamiltonian matrix created successfully!
Top-left 5x5 corner of the matrix:
[0.4764070046734181, 0.025972894669556434, 0.9720445271164556, 0.7909413983237963, 0.008352624349525994]
[0.025972894669556434, 0.02128212264819529, 0.2928492964984176, 0.3940689713878712, 0.18561313974686156]
[0.9720445271164556, 0.2928492964984176, 0.17940697489944968, 0.5939830740667027, 0.3526881704261495]
[0.7909413983237963, 0.3940689713878712, 0.5939830740667027, 0.04711547150892437, 0.47637168702175325]
[0.008352624349525994, 0.18561313974686156, 0.3526881704261495, 0.47637168702175325, 0.1318266890059513]
```

Second, check the memory efficiency with the `seff` command:

```bash
seff 7077928
Job ID: 7077928
Cluster: odyssey
User/Group: pkrastev/rc_admin
State: COMPLETED (exit code 0)
Nodes: 1
Cores per node: 2
CPU Utilized: 00:01:57
CPU Efficiency: 49.58% of 00:03:56 core-walltime
Job Wall-clock time: 00:01:58
Memory Utilized: 19.28 GB
Memory Efficiency: 48.20% of 40.00 GB (20.00 GB/core)
```

The Memory Efficiency is about 50%. The job used 19.28 GB while the requested memory 
is 40 GB. Please, notice that the requested memory is per core (20.00 GB/core). The requested memory is shared between all threads (shared-memory parallelism).

Adjust the requested memory so that the efficiency is at least 80%, and resubmit the job, e.g.,

```bash
#SBATCH --mem-per-cpu=11G
```

Submit the job with the updated batch-job submission script,

```bash
sbatch run.sbatch
Submitted batch job 7079194
```

check the job status,

```bash
sacct -j 7079194
JobID           JobName  Partition    Account  AllocCPUS      State ExitCode 
------------ ---------- ---------- ---------- ---------- ---------- -------- 
7079194      mp_mem_te+       test   rc_admin          2  COMPLETED      0:0 
7079194.bat+      batch              rc_admin          2  COMPLETED      0:0 
7079194.ext+     extern              rc_admin          2  COMPLETED      0:0 
7079194.0        python              rc_admin          2  COMPLETED      0:0 
```

and memory efficiency:

```bash
seff 7079194
Job ID: 7079194
Cluster: odyssey
User/Group: pkrastev/rc_admin
State: COMPLETED (exit code 0)
Nodes: 1
Cores per node: 2
CPU Utilized: 00:01:57
CPU Efficiency: 48.75% of 00:04:00 core-walltime
Job Wall-clock time: 00:02:00
Memory Utilized: 19.23 GB
Memory Efficiency: 87.40% of 22.00 GB (11.00 GB/core)
```

We see that the Memory Efficiency is 87.40%, while the CPU Efficiency is 48.75%.
