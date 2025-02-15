# Exercise 1: Job Efficiency - Memory per Node ( `--mem` )

We use a R code, `mem_test.R`, to generate a random matrix of dimension 20,000.  

## Step 1: Environment set up
To run the code we need to load one of the available R modules, e.g.,

```bash
module load R/4.4.1-fasrc01
```

 The R source code used in this example  is included  below:

```r
# Function to create a symmetric random matrix
create_symmetric_matrix <- function(n) {
  # Initialize an n x n matrix filled with zeros
  matrix <- matrix(0.0, nrow = n, ncol = n)
  
  # Fill the matrix with random numbers and ensure symmetry
  for (i in 1:n) {
    for (j in 1:i) {  # Only fill the lower triangular part
      value <- runif(1)  # Random float between 0 and 1
      matrix[i, j] <- value
      matrix[j, i] <- value  # Symmetric element
    }
  }
  
  return(matrix)
}

# Main function
main <- function() {
  n <- 20000  # Matrix dimension
  
  cat("Creating a symmetric random matrix of size", n, "x", n, "...\n")
  
  # Create the symmetric matrix
  h <- create_symmetric_matrix(n)
  
  cat("Hamiltonian matrix created successfully!\n")
  
  # Print a small portion of the matrix (optional)
  cat("Top-left 5x5 corner of the matrix:\n")
  print(h[1:5, 1:5])
}

# Run the main function
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
#SBATCH -t 30
#SBATCH --mem=3G 

# Load required modules
module load R/4.4.1-fasrc01

# Run the code
srun -n 1 -c 1 Rscript mem_test.R
```

## Step 3: Submit the Job

If the job-submission script is named `run_r.sbatch`, for instance, the job 
is submitted to the queue with:

```bash
sbatch run_r.sbatch
```
>**NOTE:** The job should fail due to insufficient memory. 

## Step 4: Diagnose the Issue

You can check the job status with:

```bash
sacct -j 3855840
JobID           JobName  Partition    Account  AllocCPUS      State ExitCode 
------------ ---------- ---------- ---------- ---------- ---------- -------- 
3855840        mem_test       test   rc_admin          1     FAILED      1:0 
3855840.bat+      batch              rc_admin          1     FAILED      1:0 
3855840.ext+     extern              rc_admin          1  COMPLETED      0:0 
3855840.0       Rscript              rc_admin          1 OUT_OF_ME+    0:125 
```

You can also check the STD error file with:

```bash
cat mem_test.err 
slurmstepd: error: Detected 1 oom_kill event in StepId=3855840.0. Some of the step tasks have been OOM Killed.
srun: error: holy8a26601: task 0: Out Of Memory
```

## Step 5: Adjust the Memory Request and Resubmit the Job

Modify the job-submission script to request more memory, e.g., double the memory,

```bash
#SBATCH --mem=6G  # Double the original memory request 
```

and resubmit the job:

```bash
sbatch run_r.sbatch 
Submitted batch job 3856281
```

>**NOTE:** This time the job should complete successfully.

## Step 6: Check the Job Status and Memory Efficiency

First, check the job status, e.g.,
```bash
sacct -j 3856281
JobID           JobName  Partition    Account  AllocCPUS      State ExitCode 
------------ ---------- ---------- ---------- ---------- ---------- -------- 
3856281        mem_test       test   rc_admin          1  COMPLETED      0:0 
3856281.bat+      batch              rc_admin          1  COMPLETED      0:0 
3856281.ext+     extern              rc_admin          1  COMPLETED      0:0 
3856281.0       Rscript              rc_admin          1  COMPLETED      0:0 
```
You can also check the STD output, e.g.,

```bash
cat mem_test.out 
Creating a symmetric random matrix of size 20000 x 20000 ...
Hamiltonian matrix created successfully!
Top-left 5x5 corner of the matrix:
           [,1]       [,2]      [,3]       [,4]       [,5]
[1,] 0.44621553 0.42900839 0.2862648 0.01262636 0.63216435
[2,] 0.42900839 0.49104579 0.5034983 0.72901940 0.01145602
[3,] 0.28626480 0.50349831 0.5535101 0.10061161 0.79721984
[4,] 0.01262636 0.72901940 0.1006116 0.63738541 0.51724142
[5,] 0.63216435 0.01145602 0.7972198 0.51724142 0.04589379
```

Second, check the memory efficiency with the `seff` command:

```bash
seff 3856281
Job ID: 3856281
Cluster: odyssey
User/Group: pkrastev/rc_admin
State: COMPLETED (exit code 0)
Cores: 1
CPU Utilized: 00:02:31
CPU Efficiency: 98.05% of 00:02:34 core-walltime
Job Wall-clock time: 00:02:34
Memory Utilized: 3.04 GB
Memory Efficiency: 50.62% of 6.00 GB (6.00 GB/node)
```

The Memory Efficiency is about 47%. The job used 3.04 GB while the requested memory 
is 6.00 GB. Adjust the requested memory so that the efficiency
is at least 80%, and resubmit the job, e.g.,

```bash
#SBATCH --mem=4G
```

Submit the job with the updated batch-job submission script,

```bash
sbatch run_r.sbatch 
Submitted batch job 3856774
```

check the job status,

```bash
sacct -j 3856774
JobID           JobName  Partition    Account  AllocCPUS      State ExitCode 
------------ ---------- ---------- ---------- ---------- ---------- -------- 
3856774        mem_test       test   rc_admin          1  COMPLETED      0:0 
3856774.bat+      batch              rc_admin          1  COMPLETED      0:0 
3856774.ext+     extern              rc_admin          1  COMPLETED      0:0 
3856774.0       Rscript              rc_admin          1  COMPLETED      0:0  
```

and memory efficiency:

```bash
seff 3856774 
Job ID: 3856774
Cluster: odyssey
User/Group: pkrastev/rc_admin
State: COMPLETED (exit code 0)
Cores: 1
CPU Utilized: 00:02:34
CPU Efficiency: 99.35% of 00:02:35 core-walltime
Job Wall-clock time: 00:02:35
Memory Utilized: 3.04 GB
Memory Efficiency: 75.93% of 4.00 GB (4.00 GB/node)
```

We see that the Memory Efficiency is ~76%, while the CPU Efficiency is 99.35%.
