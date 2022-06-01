## Purpose

This example illustrates using [pbdMPI](https://cran.r-project.org/web/packages/pbdMPI/index.html) on the Harvard University FASRC Cannon cluster.

## Contents

* Example of function `apply`
  * <code>pbdApply.R</code>: R source code
  * <code>run_apply.sbatch</code>: Example batch-job submission script
  * <code>pbdApply.Rout</code>: Example output
* Example of function `Lapply`
  * <code>pbdLApply.R</code>: R source code
  * <code>run_Lapply.sbatch</code>: Example batch-job submission script
  * <code>pbdLApply.Rout</code>: Example output

For more examples, you can go to your package install forlder `$HOME/apps/R/4.0.5/pbdMPI/demo` where you can find various `pbdMPI` examples

## Install and set up pbdMPI in user environment

Request an interactive node

````bash
salloc -p test --time=0:30:00 --mem=4000
````

Load required software modules.

```bash
# Compiler, MPI, and R libraries
module load gcc/9.3.0-fasrc01 openmpi/4.0.5-fasrc01 R/4.0.5-fasrc02
```

Create directory for customized R packages and set it up as a local R-library location.

```bash
mkdir -p $HOME/apps/R/4.0.5
export R_LIBS_USER=$HOME/apps/R/4.0.5:$R_LIBS_USER
```
Create a `$HOME/.R/Makevars` file with the below contents.

```bash
CC=mpicc
SHLIB_LD=mpicc
```

Install `pbdMPI`
(you must be in an interactive node for the install to be successful).

```bash
export PBDMPI_TYPE="OPENMPI"
srun Rscript -e 'install.packages("pbdMPI", repos="http://cran.us.r-project.org", configure.args=c("--with-mpi-include=${MPI_INCLUDE} --with-mpi-libpath=${MPI_LIB} --with-mpi-type=${PBDMPI_TYPE} --with-mpi=${MPI_HOME}"), configure.vars=c("CPPFLAGS=-I${MPI_INCLUDE} LDFLAGS=-L${MPI_LIB}"))'
```

Output

````bash
Installing package into ‘/n/home05/username/apps/R/4.0.5’
(as ‘lib’ is unspecified)
also installing the dependencies ‘rlecuyer’, ‘float’

... omitted output ...

* DONE (pbdMPI)

The downloaded source packages are in
	‘/tmp/RtmpSHAYPL/downloaded_packages’
````
Exit interactive node

```bash	
exit
```

### Important **note**

You must delete or rename the file <code>$HOME/.R/Makevars</code> otherwise it will conflict with future package installs.

```bash
# rename file
mv $HOME/.R/Makevars $HOME/.R/Makevars_pbdMPI_install

# delete file
rm $HOME/.R/Makevars
```

## Function `apply` example

### R source code:

For the differences between each model, refer to the `pbdMPI` [documentation](https://rdrr.io/cran/pbdMPI/man/yy_api_apply.html).

```r
# This is example is a slight modification of pbdMPI's demo `pbdApply.r`

### Initial.
suppressMessages(library(pbdMPI, quietly = TRUE))
init()
.comm.size <- comm.size()
.comm.rank <- comm.rank()

### Examples.
N <- 100
x <- matrix((1:N) + N * .comm.rank, ncol = 10)
comm.print(x)

# compute sum using "master-worker" (mw) model
y <- pbdApply(x, 1, sum, pbd.mode = "mw")
comm.print(y)

# compute sum using "Single Program Multiple Data" (spmd) model
y <- pbdApply(x, 1, sum, pbd.mode = "spmd")
comm.print(y)

comm.print("This is a print statement in pbdMPI")
comm.print(.comm.size)

### Finish.
finalize()
```

### Batch-Job Submission Script:

```bash
#!/bin/bash
#SBATCH --job-name=pbdMPI_apply
#SBATCH --output=apply_%j.out
#SBATCH --error=apply_%j.err
#SBATCH --partition=test
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=3
#SBATCH --mem-per-cpu=4000
#SBATCH -t 0-00:10               # Runtime in D-HH:MM, minimum of 10 minutes

# load modules
module load gcc/9.3.0-fasrc01 openmpi/4.0.5-fasrc01 R/4.0.5-fasrc02

# set R lirbary path
export R_LIBS_USER=$HOME/apps/R/4.0.5:$R_LIBS_USER

# choose one of the mpirun statements:
# run R program and keep output and error messages in pbdApply.Rout
mpirun Rscript --vanilla pbdApply.R > pbdApply.Rout 2>&1

# run R program and keep output in pbdApply.Rout and error messages in error file
#mpirun Rscript --vanilla pbdApply.R > pbdApply.Rout
```

Submit a job to run `pbdApply.R` in a compute node:

```bash	
sbatch run_apply.sbatch
```

### Example Output:

```r
$ cat pbdApply.Rout 
COMM.RANK = 0
      [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9] [,10]
 [1,]    1   11   21   31   41   51   61   71   81    91
 [2,]    2   12   22   32   42   52   62   72   82    92
 [3,]    3   13   23   33   43   53   63   73   83    93
 [4,]    4   14   24   34   44   54   64   74   84    94
 [5,]    5   15   25   35   45   55   65   75   85    95
 [6,]    6   16   26   36   46   56   66   76   86    96
 [7,]    7   17   27   37   47   57   67   77   87    97
 [8,]    8   18   28   38   48   58   68   78   88    98
 [9,]    9   19   29   39   49   59   69   79   89    99
[10,]   10   20   30   40   50   60   70   80   90   100
COMM.RANK = 0
 [1] 460 470 480 490 500 510 520 530 540 550
COMM.RANK = 0
[1] 460
[1] "This is a print statement in pbdMPI"
COMM.RANK = 0
[1] 6
```

## Function `Lapply` example

### R source code:

For the differences between each model, refer to the `pbdMPI` [documentation](https://rdrr.io/cran/pbdMPI/man/yy_api_apply.html).

```r
# This is example is a slight modification of pbdMPI's demo `pbdLApply.r`

# Initial
library(pbdMPI, quietly = TRUE)
init()
.comm.size <- comm.size()
.comm.rank <- comm.rank()

# Examples
N <- 100
x <- split((1:N) + N * .comm.rank, rep(1:10, each = 10))

comm.print("Master-worker model")
y <- pbdLapply(x, sum, pbd.mode = "mw")
comm.print(unlist(y))

comm.print("Single program multiple data")
y <- pbdLapply(x, sum, pbd.mode = "spmd")
comm.print(unlist(y))

comm.print("Distributed model")
y <- pbdLapply(x, sum, pbd.mode = "dist")
comm.print(unlist(y))

# Finish
finalize()
```

### Batch-Job Submission Script:

```bash
#!/bin/bash
#SBATCH --job-name=pbdMPI_Lapply
#SBATCH --output=Lapply_%j.out
#SBATCH --error=Lapply_%j.err
#SBATCH --partition=test
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=3
#SBATCH --mem-per-cpu=4000
#SBATCH -t 0-00:10               # Runtime in D-HH:MM, minimum of 10 minutes

# load modules
module load gcc/9.3.0-fasrc01 openmpi/4.0.5-fasrc01 R/4.0.5-fasrc02

# set R lirbary path
export R_LIBS_USER=$HOME/apps/R/4.0.5:$R_LIBS_USER

# choose one of the mpirun statements:
# run R program and keep output and error messages in pbdApply.Rout
mpirun Rscript --vanilla pbdLapply.R > pbdLapply.Rout 2>&1

# run R program and keep output in pbdApply.Rout and error messages in error file
#mpirun Rscript --vanilla pbdLApply.R > pbdLApply.Rout
```

Submit a job to run `pbdLApply.R` in a compute node:

```bash	
sbatch run_Lapply.sbatch
```

### Example Output:

```r
$ cat pbdLApply.Rout 
[1] "Master-worker model"
  1   2   3   4   5   6   7   8   9  10
 55 155 255 355 455 555 655 755 855 955
[1] "Single program multiple data"
 1
55
[1] "Distributed model"
  1   2   3   4   5   6   7   8   9  10
 55 155 255 355 455 555 655 755 855 955
```

## Function `scatter` example

### R source code:

For the differences between each model, refer to the `pbdMPI` [documentation](https://rdrr.io/cran/pbdMPI/man/yy_api_apply.html).

```r
# Initial.
suppressMessages(library(pbdMPI, quietly = TRUE))
init()
.comm.size <- comm.size()
.comm.rank <- comm.rank()

# Examples
N <- 5
x.total <- (.comm.size + 1) * .comm.size / 2
x <- 1:x.total
x.count <- 1:.comm.size
comm.cat("Original x:\n", quiet = TRUE)
comm.print(x)

y <- scatter(split(x, rep(x.count, x.count)))    ### return the element of list.
comm.cat("\nScatter list:\n", quiet = TRUE)
comm.print(y)

y <- scatter(as.integer(x), integer(.comm.rank + 1), as.integer(x.count))
comm.cat("\nScatterv integer:\n", quiet = TRUE)
comm.print(y, rank.print = 1)

y <- scatter(as.double(x), double(.comm.rank + 1), as.integer(x.count))
comm.cat("\nScatterv double:\n", quiet = TRUE)
comm.print(y, rank.print = 1)

### Finish.
finalize()
```

### Batch-Job Submission Script:

```bash
#!/bin/bash
#SBATCH --job-name=pbdMPI_scatter
#SBATCH --output=test_%j.out
#SBATCH --error=test_%j.err
#SBATCH --partition=test
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=3
#SBATCH --mem-per-cpu=4000
#SBATCH -t 0-00:10               # Runtime in D-HH:MM, minimum of 10 minutes

# load modules
module load gcc/9.3.0-fasrc01 openmpi/4.0.5-fasrc01 R/4.0.5-fasrc02

# set R lirbary path
export R_LIBS_USER=$HOME/apps/R/4.0.5:$R_LIBS_USER

# choose one of the mpirun statements:
# run R program and keep output and error messages in pbdApply.Rout
mpirun Rscript --vanilla scatter.R > scatter.Rout 2>&1

# run R program and keep output in pbdApply.Rout and error messages in error file
#mpirun Rscript --vanilla scatter.R > scatter.Rout
```

Submit a job to run `pbdLApply.R` in a compute node:

```bash	
sbatch run_scatter.sbatch
```

### Example Output:

```r
$ cat scatter.Rout 
Original x:
COMM.RANK = 0
 [1]  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21

Scatter list:
COMM.RANK = 0
[1] 1

Scatterv integer:
COMM.RANK = 1
[1] 2 3

Scatterv double:
COMM.RANK = 1
[1] 2 3
```


