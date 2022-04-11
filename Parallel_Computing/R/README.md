### Purpose:

This example illustrates using [Rmpi](https://cran.r-project.org/web/packages/Rmpi/index.html) on the Harvard University FASRC cluster.

### Contents:

* <code>mpi_test.R</code>: R source code
* <code>run.sbatch</code>: Example batch-job submission script
* <code>mpi\_test.Rout</code>: Example output

### Install and set up Rmpi in user environment:

Request an interactive node

````bash
salloc -p test --time=0:30:00 --mem=1000
````

Load required software modules.

```bash
# Compiler, MPI, and R libraries
module load R/3.5.1-fasrc01
module load gcc/10.2.0-fasrc01 openmpi/4.1.1-fasrc01
```

Create directory for customized R packages and set it up as a local R-library location.

```bash
mkdir -p $HOME/apps/R/3.5.1
export R_LIBS_USER=$HOME/apps/R/3.5.1:$R_LIBS_USER
```

Install <code>Rmpi</code>
(you must be in an interactive node for the install to be successful).

```bash
export RMPI_TYPE="OPENMPI"
srun Rscript -e 'install.packages("Rmpi", repos="http://cran.us.r-project.org", configure.args=c("--with-Rmpi-include=${MPI_INCLUDE} --with-Rmpi-libpath=${MPI_LIB} --with-Rmpi-type=${RMPI_TYPE}"), configure.vars=c("CPPFLAGS=-I${MPI_INCLUDE} LDFLAGS=-L${MPI_LIB}"))'
```

Output

````bash
Installing package into ‘/n/home05/username/apps/R/3.5.1’
(as ‘lib’ is unspecified)
trying URL 'http://cran.us.r-project.org/src/contrib/Rmpi_0.6-9.2.tar.gz'
Content type 'application/x-gzip' length 106030 bytes (103 KB)
==================================================
downloaded 103 KB

* installing *source* package ‘Rmpi’ ...
** package ‘Rmpi’ successfully unpacked and MD5 sums checked

... omitted output ...

* DONE (Rmpi)

The downloaded source packages are in
	‘/tmp/RtmpGytOWX/downloaded_packages’
````
Exit interactive node

```bash	
exit
```

**Note**: If you attempted a previous installation with the file <code>$HOME/.R/Makevars</code>, you must remove/rename the file <code>$HOME/.R/Makevars</code> otherwise it will conflict with future installs.

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
#SBATCH -o mpi_test.out
#SBATCH -e mpi_test.err
#SBATCH -p test
#SBATCH -n 8
#SBATCH -t 30
#SBATCH --mem-per-cpu=4000

# Load required software modules 
module load R/3.5.1-fasrc01
module load gcc/10.2.0-fasrc01 openmpi/4.1.1-fasrc01

# Set up Rmpi package
export R_LIBS_USER=$HOME/apps/R/3.5.1:$R_LIBS_USER
export R_PROFILE=$HOME/apps/R/3.5.1/Rmpi/Rprofile

# Run program
export OMPI_MCA_mpi_warn_on_fork=0
srun -n 8 --mpi=pmix R CMD BATCH --no-save --no-restore mpi_test.R
```
**Note:** Please notice the line <code>export R_PROFILE=$HOME/apps/R/3.5.1/Rmpi/Rprofile</code> in the above batch-job submission script. It is very important to set the <code>R\_PROFILE</code> environment variable to point to the correct <code>Rprofile</code> file for <code>Rmpi</code> to work correctly.

### Example Output:

```r
$ cat mpi_test.Rout 

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

master (rank 0, comm 1) of size 8 is running on: holy7c18312 
slave1 (rank 1, comm 1) of size 8 is running on: holy7c18312 
slave2 (rank 2, comm 1) of size 8 is running on: holy7c18312 
slave3 (rank 3, comm 1) of size 8 is running on: holy7c18312 
slave4 (rank 4, comm 1) of size 8 is running on: holy7c18312 
slave5 (rank 5, comm 1) of size 8 is running on: holy7c18312 
slave6 (rank 6, comm 1) of size 8 is running on: holy7c18312 
slave7 (rank 7, comm 1) of size 8 is running on: holy7c18312 
> # Load the R MPI package if it is not already loaded.
> if (!is.loaded("mpi_initialize")) {
+     library("Rmpi")
+     }
>  
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
1 -2.29725577 -1.2114942  1.6391021  0.40414602 -0.1129386  1.2655687
2  1.61999298  0.2420147 -1.2218427 -0.47842102 -2.7758085  0.4352998
3 -0.04977144  0.1748042 -1.3156919  0.71658806 -1.8217445 -0.4598137
4 -0.25964969  0.3763296  0.8649287 -0.04017663  0.6134354 -1.4181552
5  0.54973583  1.1025789  1.9306896  0.79261701 -1.2906490 -0.6792234
          X7
1 -0.8693797
2 -1.0577793
3  1.1263427
4  1.1208240
5  0.8360957
>  
> # Tell all slaves to close down, and exit the program
> mpi.close.Rslaves(dellog = FALSE)
[1] 1
> mpi.quit()
```
