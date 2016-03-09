### Purpose:

Program illustrates use of R-MPI on the Odyssey cluster.

### Contents:

(1) mpi_test.R: R source code.

(2) run.sbatch: Btach-job submission script to send the job to the queue.

### Example Usage:

	source new-modules.sh
	module load intel/15.0.0-fasrc01
	module load mvapich2/2.0-fasrc03
	module load R/3.2.2-fasrc02	
	sbatch run.sbatch
 
### Example Output:

```
R version 3.2.2 (2015-08-14) -- "Fire Safety"
Copyright (C) 2015 The R Foundation for Statistical Computing
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

master (rank 0, comm 1) of size 8 is running on: holy2a13205 
slave1 (rank 1, comm 1) of size 8 is running on: holy2a13306 
slave2 (rank 2, comm 1) of size 8 is running on: holy2a13306 
slave3 (rank 3, comm 1) of size 8 is running on: holy2a13306 
slave4 (rank 4, comm 1) of size 8 is running on: holy2a13306 
slave5 (rank 5, comm 1) of size 8 is running on: holy2a13306 
slave6 (rank 6, comm 1) of size 8 is running on: holy2a13306 
slave7 (rank 7, comm 1) of size 8 is running on: holy2a13306 
> # Load the R MPI package if it is not already loaded.
> if (!is.loaded("mpi_initialize")) {
+     library("Rmpi")
+     }
> 
> # In case R exits unexpectedly, have it automatically clean up
> # resources taken up by Rmpi (slaves, memory, etc...)
> .Last <- function(){
+     if (is.loaded("mpi_initialize")){
+         if (mpi.comm.size(1) > 0){
+             print("Please use mpi.close.Rslaves() to close slaves.")
+             mpi.close.Rslaves()
+         }
+         print("Please use mpi.quit() to quit R")
+         .Call("mpi_finalize")
+     }
+ }
> 
> # Tell all slaves to return a message identifying themselves
> mpi.remote.exec(paste("I am",mpi.comm.rank(),"of",mpi.comm.size(),system("hostname",intern=T)))
$slave1
[1] "I am 1 of 8 holy2a13306.rc.fas.harvard.edu"

$slave2
[1] "I am 2 of 8 holy2a13306.rc.fas.harvard.edu"

$slave3
[1] "I am 3 of 8 holy2a13306.rc.fas.harvard.edu"

$slave4
[1] "I am 4 of 8 holy2a13306.rc.fas.harvard.edu"

$slave5
[1] "I am 5 of 8 holy2a13306.rc.fas.harvard.edu"

$slave6
[1] "I am 6 of 8 holy2a13306.rc.fas.harvard.edu"

$slave7
[1] "I am 7 of 8 holy2a13306.rc.fas.harvard.edu"

> 
> # Tell all slaves to close down, and exit the program
> mpi.close.Rslaves()
[1] 1
> 
> 
[1] "Please use mpi.quit() to quit R"
> proc.time()
   user  system elapsed 
  2.803   0.210   7.239
```
