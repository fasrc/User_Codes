#### Purpose:

Example code to illustrate R use on the Odyssey cluster. The specific example prins out integers from 10 down to 1.

#### Contents:

* <code>count\_down.R</code>: R source code
* <code>run.sbatch</code>: Batch-job submission script
* <code>count_down.Rout</code>

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
#SBATCH -J count_down
#SBATCH -o count_down.out
#SBATCH -e count_down.err
#SBATCH -p shared
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 0-00:30
#SBATCH --mem=4000

# Load required software modules
module load R/3.5.1-fasrc01

# Run program
srun -n 1 -c 1 R --vanilla < count_down.R 
```

#### Example Usage:

```bash
module load R/3.5.1-fasrc01
sbatch run.sbatch
```
#### Example Output:

```
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
  0.264   0.093  12.301 
```
