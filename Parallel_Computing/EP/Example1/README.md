#### Purpose:

Example illustrating use of job-arrays on the Harvard University FASRC cluster. The specific example computes the sum of integers from 1 through N, where N is the value of the job array index (100, 200, 300).

#### Contents:

* <code>serial\_sum.R</code>: R source code
* <code>run.sbatch</code>: Batch-job submission script
* <code>serial\_sum.R.100.out</code>: Example output (N=100)
* <code>serial\_sum.R.200.out</code>: Example output (N=200)
* <code>serial\_sum.R.300.out</code>: Example output (N=300)

#### Source Code:

```r
#================================================
# Function: serial_sum(N)
#           Returns the sum of integers from 1
#           through N
#================================================
serial_sum <- function(x){
  k <- 0
  s <- 0
  while (k < x){
    k <- k + 1
    s <- s + k
  }
  return(s)
}

# +++ Main program +++
tid <- as.integer(Sys.getenv('SLURM_ARRAY_TASK_ID'))
res <- serial_sum(x=tid)
print(res)
```

#### Batch-Jobs Submission Script:

```bash
#!/bin/bash
#SBATCH -J array_test
#SBATCH -p shared
#SBATCH -c 1
#SBATCH -t 00:10:00
#SBATCH --mem=4G
#SBATCH -o %A-%a.o
#SBATCH -e %A-%a.e
#SBATCH --array=100,200,300

# Load software environment
module load R/3.5.1-fasrc01

input=serial_sum.R

# Execute code
srun -n 1 -c 1 R CMD BATCH $input $input.$SLURM_ARRAY_TASK_ID.out
```

#### Usage:

```bash
sbatch run.sbatch
```

### Example Output:

```
$ cat serial_sum.R.100.out

R version 3.5.1 (2018-07-02) -- "Feather Spray"
Copyright (C) 2018 The R Foundation for Statistical Computing
Platform: x86_64-pc-linux-gnu (64-bit)

R is free software and comes with ABSOLUTELY NO WARRANTY.
You are welcome to redistribute it under certain conditions.
Type 'license()' or 'licence()' for distribution details.

  Natural language support but running in an English locale

R is a collaborative project with many contributors.
Type 'contributors()' for more information and
'citation()' on how to cite R or R packages in publications.

Type 'demo()' for some demos, 'help()' for on-line help, or
'help.start()' for an HTML browser interface to help.
Type 'q()' to quit R.

[Previously saved workspace restored]

> #================================================
> # Function: serial_sum(N)
> #           Returns the sum of integers from 1
> #           through N
> #================================================
> serial_sum <- function(x){
+   k <- 0
+   s <- 0
+   while (k < x){
+     k <- k + 1
+     s <- s + k
+   }
+   return(s)
+ }
> 
> # +++ Main program +++
> tid <- as.integer(Sys.getenv('SLURM_ARRAY_TASK_ID'))
> res <- serial_sum(x=tid)
> print(res)
[1] 5050
> 
> proc.time()
   user  system elapsed 
  0.228   0.067   0.832 
```