### Purpose:

This example illustrates using Rmpi on the Research Computing cluster.

### Contents:

* <code>mpi_test.R</code>: R source code
* <code>run.sbatch</code>: Example batch-job submission script
* <code>mpi_test.Rout</code>: Example output

### Install and set up Rmpi in user environment:

Load required software modules.

```bash
module load gcc/7.1.0-fasrc01 
module load openmpi/3.1.3-fasrc01 
module load R/3.5.1-fasrc02
```

Create directory for customized R packages and set it up as a local R-library location.

```bash
mkdir $HOME/software/R/3.5.1
export R_LIBS_USER=$HOME/software/R/3.5.1:$R_LIBS_USER
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
module load gcc/7.1.0-fasrc01 
module load openmpi/3.1.3-fasrc01 
module load R/3.5.1-fasrc02		
sbatch run.sbatch
```

### R source code:

```r
# Load the R MPI package if it is not already loaded.
if (!is.loaded("mpi_initialize")) {
    library("Rmpi")
    }

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
mpi.remote.exec(paste("I am",mpi.comm.rank(),"of",mpi.comm.size(),system("hostname",intern=T)))

# Tell all slaves to close down, and exit the program
mpi.close.Rslaves()
```

### Batch-Job Submission Script:

```bash
#!/bin/bash
#SBATCH -J mpi_test
#SBATCH -o mpi_test.out
#SBATCH -e mpi_test.err
#SBATCH -p shared
#SBATCH -n 8
#SBATCH -t 30
#SBATCH --mem-per-cpu=4000

# Load required software modules
module load gcc/7.1.0-fasrc01 
module load openmpi/3.1.3-fasrc01 
module load R/3.5.1-fasrc02

# Set up Rmpi package
export R_LIBS_USER=$HOME/software/R/3.5.1:$R_LIBS_USER
export R_PROFILE=$HOME/software/R/3.5.1/Rmpi/Rprofile

# Run program
export OMPI_MCA_mpi_warn_on_fork=0
srun -n 8 --mpi=pmix R CMD BATCH --no-save --no-restore mpi_test.R
```
**Note:** Please notice the line <code>export R_PROFILE=$HOME/software/R/3.5.1/Rmpi/Rprofile</code> in the above batch-job submission script. It is very important to set the <code>R\_PROFILE</code> environment variable to point to the correct <code>Rprofile</code> file for <code>Rmpi</code> to work correctly.

### Example Output:

```r
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

master (rank 0, comm 1) of size 8 is running on: holy7c01101 
slave1 (rank 1, comm 1) of size 8 is running on: holy7c01102 
slave2 (rank 2, comm 1) of size 8 is running on: holy7c01102 
slave3 (rank 3, comm 1) of size 8 is running on: holy7c01103 
slave4 (rank 4, comm 1) of size 8 is running on: holy7c01103 
slave5 (rank 5, comm 1) of size 8 is running on: holy7c01103 
slave6 (rank 6, comm 1) of size 8 is running on: holy7c01103 
slave7 (rank 7, comm 1) of size 8 is running on: holy7c01103 
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
[1] "I am 1 of 8 holy7c01102.rc.fas.harvard.edu"

$slave2
[1] "I am 2 of 8 holy7c01102.rc.fas.harvard.edu"

$slave3
[1] "I am 3 of 8 holy7c01103.rc.fas.harvard.edu"

$slave4
[1] "I am 4 of 8 holy7c01103.rc.fas.harvard.edu"

$slave5
[1] "I am 5 of 8 holy7c01103.rc.fas.harvard.edu"

$slave6
[1] "I am 6 of 8 holy7c01103.rc.fas.harvard.edu"

$slave7
[1] "I am 7 of 8 holy7c01103.rc.fas.harvard.edu"

> 
> # Tell all slaves to close down, and exit the program
> mpi.close.Rslaves()
[1] 1
> 
> 
[1] "Please use mpi.quit() to quit R"
> proc.time()
   user  system elapsed 
  2.581   0.170   4.181
```
