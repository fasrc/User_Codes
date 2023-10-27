#### Purpose:

Example code to illustrate R use on the FASRC cluster. The specific example prins out integers from 10 down to 1.

#### Contents:

* <code>count\_down.R</code>: R source code
* <code>run.sbatch</code>: Batch-job submission script
* <code>count_down.Rout</code> : Output file

#### R source code:

```r
#===========================================================
# Program: count_down.R
#
# Run:     R --vanilla < count_down.R         
#===========================================================

# Function CountDown........................................
CountDown <- function(x)
{
  print( x )
  while( x != 0 )
  {
    Sys.sleep(1)
    x <- x - 1
    print( x )
  }
}

# Call CountDown............................................
CountDown( 10 )
```

#### Example Batch-Job Submission Script:

```bash
#!/bin/bash
#SBATCH -J count_down         # job name
#SBATCH -o count_down.out     # standard output file
#SBATCH -e count_down.err     # standard error file
#SBATCH -p shared             # partition
#SBATCH -c 1                  # number of cores
#SBATCH -t 0-00:30            # time in D-HH:MM
#SBATCH --mem=4000            # memory in MB

# Load required software modules
module load R

# Run program
srun -c 1 R --vanilla < count_down.R
```

#### Example Usage:

```bash
sbatch run.sbatch
```
#### Example Output:

Content of file `count_down.Rout`:

```
R version 4.3.1 (2023-06-16) -- "Beagle Scouts"
Copyright (C) 2023 The R Foundation for Statistical Computing
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

> #===========================================================
> # Program: count_down.R
> #
> # Run:     R --vanilla < count_down.R         
> #===========================================================
> 
> # Function CountDown........................................
> CountDown <- function(x)
+ {
+   print( x )
+   while( x != 0 )
+   {
+     Sys.sleep(1)
+     x <- x - 1
+     print( x )
+   }
+ }
> 
> # Call CountDown............................................
> CountDown( 10 )
[1] 10
[1] 9
[1] 8
[1] 7
[1] 6
[1] 5
[1] 4
[1] 3
[1] 2
[1] 1
[1] 0
> 
> proc.time()
   user  system elapsed 
  0.138   0.038  10.386
```
