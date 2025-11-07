# Exercise 2: Job Efficiency - Memory per CPU/core ( `--mem-per-cpu` )

We use a R code, `mp_mem_test.R`, to generate a random matrix of dimension 20,000 in parallel via the `parallel` R package. The specific example uses 2 threads. 

## Step 1: Environment set up
To run the code we need to load one of the available R modules, e.g.,

```bash
module load R/4.4.3-fasrc01
```

 The R source code used in this example  is included  below:

```r
library(parallel)

# Parallel function to create a symmetric random matrix
create_symmetric_matrix_parallel <- function(n) {
  matrix <- matrix(0.0, nrow = n, ncol = n)

  # Set number of threads
  cores <- 2  # explicitly set number of threads to 2
  cat("Using", cores, "cores for parallel execution\n")

  # Split the indices for parallel execution
  indices <- split(1:n, cut(1:n, cores, labels = FALSE))

  # Define parallel task
  parallel_task <- function(rows) {
    partial_matrix <- matrix(0.0, nrow = length(rows), ncol = n)
    for (idx in seq_along(rows)) {
      i <- rows[idx]
      for (j in 1:i) {
        value <- runif(1)
        partial_matrix[idx, j] <- value
        partial_matrix[idx, j] <- value
      }
    }
    return(partial_matrix)
  }

  # Execute in parallel
  results <- mclapply(indices, parallel_task, mc.cores = cores)

  # Combine results into the matrix
  current_row <- 1
  for (partial_result in results) {
    rows_count <- nrow(partial_result)
    matrix[current_row:(current_row + rows_count - 1), ] <- partial_result
    current_row <- current_row + rows_count
  }

  # Mirror lower triangular to upper triangular part
  matrix[upper.tri(matrix)] <- t(matrix)[upper.tri(matrix)]

  return(matrix)
}

# Main function
main_parallel <- function() {
  n <- 20000
  cat("Creating a symmetric random matrix of size", n, "x", n, "in parallel...\n")

  h <- create_symmetric_matrix_parallel(n)

  cat("Hamiltonian matrix created successfully!\n")

  cat("Top-left 5x5 corner of the matrix:\n")
  print(h[1:5, 1:5])
}

# Run the parallel main function
main_parallel()
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
#SBATCH --mem-per-cpu=5G 

# --- Load required modules ---
module load R/4.4.3-fasrc01

# --- Run the code ---
srun -c ${SLURM_CPUS_PER_TASK} Rscript mp_mem_test.R
```

## Step 3: Submit the Job

If the job-submission script is named `run.sbatch`, for instance, the job 
is submitted to the queue with:

```bash
sbatch run.sbatch
```
>**NOTE:** The job should fail due to insufficient memory. 

## Step 4: Diagnose the Issue

You can check the job status with:

```bash
sacct -j 7083729
JobID           JobName  Partition    Account  AllocCPUS      State ExitCode 
------------ ---------- ---------- ---------- ---------- ---------- -------- 
7083729      mp_mem_te+       test   rc_admin          2     FAILED      1:0 
7083729.bat+      batch              rc_admin          2     FAILED      1:0 
7083729.ext+     extern              rc_admin          2  COMPLETED      0:0 
7083729.0       Rscript              rc_admin          2 OUT_OF_ME+    0:125 
```

You can also check the STD error file with:

```bash
cat mp_mem_test.err 
Error in current_row:(current_row + rows_count - 1) : 
  argument of length 0
Calls: main_parallel -> create_symmetric_matrix_parallel
In addition: Warning message:
In mclapply(indices, parallel_task, mc.cores = cores) :
  scheduled core 2 did not deliver a result, all values of the job will be affected
Execution halted
slurmstepd: error: Detected 2 oom_kill events in StepId=7083729.0. Some of the step tasks have been OOM Killed.
srun: error: holy8a24101: task 0: Out Of Memory
```

## Step 5: Adjust the Memory Request and Resubmit the Job

Modify the job-submission script to request more memory, e.g., double the memory,

```bash
#SBATCH --mem-per-cpu=10G  # Double the original memory request 
```

and resubmit the job:

```bash
sbatch run.sbatch
Submitted batch job 7084142
```

>**NOTE:** This time the job should complete successfully.

## Step 6: Check the Job Status and Memory Efficiency

First, check the job status, e.g.,
```bash
sacct -j 7084142
JobID           JobName  Partition    Account  AllocCPUS      State ExitCode 
------------ ---------- ---------- ---------- ---------- ---------- -------- 
7084142      mp_mem_te+       test   rc_admin          2  COMPLETED      0:0 
7084142.bat+      batch              rc_admin          2  COMPLETED      0:0 
7084142.ext+     extern              rc_admin          2  COMPLETED      0:0 
7084142.0       Rscript              rc_admin          2  COMPLETED      0:0 
```
You can also check the STD output, e.g.,

```bash
cat mp_mem_test.out 
Creating a symmetric random matrix of size 20000 x 20000 in parallel...
Using 2 cores for parallel execution
Hamiltonian matrix created successfully!
Top-left 5x5 corner of the matrix:
          [,1]      [,2]      [,3]      [,4]      [,5]
[1,] 0.3920189 0.4611903 0.2228508 0.8211259 0.3817897
[2,] 0.4611903 0.8291229 0.5537287 0.5074117 0.0433086
[3,] 0.2228508 0.5537287 0.1846014 0.3344495 0.8323349
[4,] 0.8211259 0.5074117 0.3344495 0.5606949 0.2630786
[5,] 0.3817897 0.0433086 0.8323349 0.2630786 0.3534443
```

Second, check the memory efficiency with the `seff` command:

```bash
seff 7084142
Job ID: 7084142
Cluster: odyssey
User/Group: pkrastev/rc_admin
State: COMPLETED (exit code 0)
Nodes: 1
Cores per node: 2
CPU Utilized: 00:03:21
CPU Efficiency: 64.01% of 00:05:14 core-walltime
Job Wall-clock time: 00:02:37
Memory Utilized: 12.00 GB
Memory Efficiency: 60.01% of 20.00 GB (10.00 GB/core)
```

The Memory Efficiency is about 60%. The job used 12 GB while the requested memory 
is 20.00 GB. Please, notice that the requested memory is per core (10.00 GB/core).


Adjust the requested memory so that the efficiency is at least 80%, and resubmit the 
job, e.g.,

```bash
#SBATCH --mem-per-cpu=9G
```

Submit the job with the updated batch-job submission script,

```bash
sbatch run.sbatch
Submitted batch job 7085868
```

check the job status,

```bash
sacct -j 7085868
JobID           JobName  Partition    Account  AllocCPUS      State ExitCode 
------------ ---------- ---------- ---------- ---------- ---------- -------- 
7085868      mp_mem_te+       test   rc_admin          2  COMPLETED      0:0 
7085868.bat+      batch              rc_admin          2  COMPLETED      0:0 
7085868.ext+     extern              rc_admin          2  COMPLETED      0:0 
7085868.0       Rscript              rc_admin          2  COMPLETED      0:0 
```

and memory efficiency:

```bash
seff 7085868
Job ID: 7085868
Cluster: odyssey
User/Group: pkrastev/rc_admin
State: COMPLETED (exit code 0)
Nodes: 1
Cores per node: 2
CPU Utilized: 00:03:14
CPU Efficiency: 62.99% of 00:05:08 core-walltime
Job Wall-clock time: 00:02:34
Memory Utilized: 15.23 GB
Memory Efficiency: 84.62% of 18.00 GB (9.00 GB/core)
```

We see that the Memory Efficiency is ~85%, while the CPU Efficiency is ~63%.
