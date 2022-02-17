### Purpose:

This example illustrates using [Rmpi](https://cran.r-project.org/web/packages/Rmpi/index.html) on the Harvard University FASRC cluster.

### Contents:

* <code>mpi_test.R</code>: R source code
* <code>run.sbatch</code>: Example batch-job submission script
* <code>mpi\_test.out</code>: Example output

### Install and set up Rmpi in user environment:

Load required software modules.

```bash
# Compiler, MPI, and R libraries
module load gcc/9.3.0-fasrc01 openmpi/4.0.5-fasrc01 R/4.0.5-fasrc03
```

Create directory for customized R packages and set it up as a local R-library location.

```bash
mkdir -p $HOME/apps/R_4.0.5_fasrc03
export R_LIBS_USER=$HOME/apps/R_4.0.5_fasrc03:$R_LIBS_USER
```

Create a <code>$HOME/.R/Makevars</code> file with the below contents.

```bash
CC=mpicc
SHLIB_LD=mpicc
```

Install <code>Rmpi</code>.


```bash
export RMPI_TYPE="OPENMPI"
echo 'install.packages("Rmpi", repos="http://cran.us.r-project.org", configure.args=c("--with-Rmpi-include=${MPI_INCLUDE} --with-Rmpi-libpath=${MPI_LIB} --with-Rmpi-type=${RMPI_TYPE}"), configure.vars=c("CPPFLAGS=-I${MPI_INCLUDE} LDFLAGS=-L${MPI_LIB}"))' | R --vanilla
```

**Note**: After the installation you may remove/rename the file <code>$HOME/.R/Makevars</code>

### Example Usage:

```bash	
sbatch run.sbatch
```

### R source code:

```r
# Load the R MPI package if it is not already loaded.
if (!is.loaded("mpi_initialize")) {
    library("Rmpi")
    }
 
#
# In case R exits unexpectedly, have it automatically clean up
# resources taken up by Rmpi (slaves, memory, etc...)
.Last <- function(){
       if (is.loaded("mpi_initialize")){
           if (mpi.comm.size(1) > 0){
               print("Please use mpi.close.Rslaves() to close slaves.")
               mpi.close.Rslaves()
           }
           print("Please use mpi.quit() to quit R")
           .Call("mpi_finalize")
       }
}
# Tell all slaves to return a message identifying themselves
mpi.bcast.cmd( id <- mpi.comm.rank() )
mpi.bcast.cmd( ns <- mpi.comm.size() )
mpi.bcast.cmd( host <- mpi.get.processor.name() )
mpi.remote.exec(paste("I am",mpi.comm.rank(),"of",mpi.comm.size()))
 
# Test computations
x <- 5
x <- mpi.remote.exec(rnorm, x)
length(x)
print(x)
 
# Tell all slaves to close down, and exit the program
mpi.close.Rslaves(dellog = FALSE)
mpi.quit()
```

### Batch-Job Submission Script:

```bash
#!/bin/bash
#SBATCH -J mpi_test
#SBATCH -o mpi_test.o
#SBATCH -e mpi_test.e
#SBATCH -p shared
#SBATCH -n 8
#SBATCH -t 30
#SBATCH --mem-per-cpu=4000

# Load required software modules 
module load gcc/9.3.0-fasrc01 openmpi/4.0.5-fasrc01 R/4.0.5-fasrc03

# Set up Rmpi package
export R_LIBS_USER=$HOME/apps/R_4.0.5_fasrc03:$R_LIBS_USER
export R_PROFILE=$HOME/apps/R_4.0.5_fasrc03/Rmpi/Rprofile

# Run program
export OMPI_MCA_mpi_warn_on_fork=0
srun -n 8 --mpi=pmix R CMD BATCH --no-save --no-restore mpi_test.R mpi_test.out
```
**Note:** Please notice the line <code>export R_PROFILE=$HOME/apps/R_4.0.5_fasrc03/Rmpi/Rprofile</code> in the above batch-job submission script. It is very important to set the <code>R\_PROFILE</code> environment variable to point to the correct <code>Rprofile</code> file for <code>Rmpi</code> to work correctly.

### Example Output:

```r
$ cat mpi_test.out 

R version 4.0.5 (2021-03-31) -- "Shake and Throw"
Copyright (C) 2021 The R Foundation for Statistical Computing
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

master (rank 0, comm 1) of size 8 is running on: holy7c16407 
slave1 (rank 1, comm 1) of size 8 is running on: holy7c16407 
slave2 (rank 2, comm 1) of size 8 is running on: holy7c16407 
slave3 (rank 3, comm 1) of size 8 is running on: holy7c16408 
slave4 (rank 4, comm 1) of size 8 is running on: holy7c16408 
slave5 (rank 5, comm 1) of size 8 is running on: holy7c16409 
slave6 (rank 6, comm 1) of size 8 is running on: holy7c16409 
slave7 (rank 7, comm 1) of size 8 is running on: holy7c16410 
> # Load the R MPI package if it is not already loaded.
> if (!is.loaded("mpi_initialize")) {
+     library("Rmpi")
+     }
>  
> #
> # In case R exits unexpectedly, have it automatically clean up
> # resources taken up by Rmpi (slaves, memory, etc...)
> .Last <- function(){
+        if (is.loaded("mpi_initialize")){
+            if (mpi.comm.size(1) > 0){
+                print("Please use mpi.close.Rslaves() to close slaves.")
+                mpi.close.Rslaves()
+            }
+            print("Please use mpi.quit() to quit R")
+            .Call("mpi_finalize")
+        }
+ }
> # Tell all slaves to return a message identifying themselves
> mpi.bcast.cmd( id <- mpi.comm.rank() )
> mpi.bcast.cmd( ns <- mpi.comm.size() )
> mpi.bcast.cmd( host <- mpi.get.processor.name() )
> mpi.remote.exec(paste("I am",mpi.comm.rank(),"of",mpi.comm.size()))
$slave1
[1] "I am 1 of 8"

$slave2
[1] "I am 2 of 8"

$slave3
[1] "I am 3 of 8"

$slave4
[1] "I am 4 of 8"

$slave5
[1] "I am 5 of 8"

$slave6
[1] "I am 6 of 8"

$slave7
[1] "I am 7 of 8"

>  
> # Test computations
> x <- 5
> x <- mpi.remote.exec(rnorm, x)
> length(x)
[1] 7
> print(x)
           X1         X2         X3          X4         X5         X6
1 -0.51796698  2.0623506  1.9722837  0.40322662  0.6707728 -0.1150194
2  1.22351449 -0.2613692  0.5460643  0.06039536  0.4644074 -0.5661581
3 -0.06609504  0.1705551  0.7206776 -1.39576360 -0.5939334  1.2169408
4 -0.66172839 -0.7379895 -0.8374896 -1.51531698  1.3193620 -0.8606493
5 -0.68540647 -0.9281437 -1.3998907 -0.83743830 -1.9481434  0.4069756
          X7
1 -0.7803866
2  0.1022698
3 -1.1203312
4  0.2049703
5 -1.5066558
>  
> # Tell all slaves to close down, and exit the program
> mpi.close.Rslaves(dellog = FALSE)
[1] 1
> mpi.quit()
```
