## Purpose

These examples show how to use the package `future` on the Harvard University FASRC Cannon cluster.
* [future documentation](https://future.futureverse.org/)
* [future GitHub](https://github.com/HenrikBengtsson/future)

## Contents

* Example of `future` plans `sequential`, `multisession`, and `multicore`
  * `future_slow_square.R`: R source code
  * `run_future.sbatch`: Example batch-job submission script
  * `future_slow_square.Rout`: Example output 
* Example nested futures with `batchtools_slurm` and `multicore`
  * `future_hybrid.R`: R source code
  * `future_slurm.tmpl`: SLURM batch-job submit script template used by `batchtools_slurm`
  * `run_hybrid.sbatch`: Example batch-job submission script
  * `future_hybrid.Rout`: Example output from R code
  * `Rhybrid.log`: Example output from `batchtools_slurm`

For more examples, refer to:
* [future examples](https://github.com/HenrikBengtsson/future)
* [bootstrapping example](https://grantmcdermott.com/ds4e/parallel.html#example-2-bootstrapping)

## `future` basics

The package `future` has various `plan` to parallelize an R code:

* `sequential`
  * synchronous
  * no parallelization
  * single node
* `multisession`
  * asynchronous
  * background parallelization (no forking): "futures" are evaluated in independent processes as if you are running multiple R/RStudio sessions
  * single node
* `multicore`
  * asynchronous
  * parallelization with forked R processes
  * shared memory
  * single node
* `cluster`
  * asynchronous
  * multi node
  * external R sessions on current, local, and/or remote nodes
  * uses `ssh` to communicate with remote nodes
* `batchtools`
  * asynchronous
  * multi node
  * uses batch submit scripts to submit jobs to remote nodes; specifically on Cannon, we use `batchtools_slurm`

In addition to the `plan` listed above, `future` also allows nested futures. For example, to achieve a hybrid job with multi-node and shared memory (analogous to the more traditional `MPI`+`OpenMP` hybrid configuration), we can use `plan(list(batchtools_slurm,multicore))` as shown in the example below. 
 
## Install and set up future user environment

Request an interactive node

````bash
salloc -p test --time=0:30:00 --mem=4000
````

Load required software modules.

```bash
# R libraries
module load R/4.1.0-fasrc01
```

Create directory for customized R packages and set it up as a local R-library location.

```bash
mkdir -p $HOME/apps/R/4.1.0
export R_LIBS_USER=$HOME/apps/R/4.1.0:$R_LIBS_USER
unset R_LIBS_SITE
```

Install `future` inside the R shell

```r
> install.packages("future")
```

Output

````bash
Installing package into ‘/n/home05/username/apps/R/4.1.0’
(as ‘lib’ is unspecified)

... omitted output ...

Selection: 78
also installing the dependencies ‘globals’, ‘listenv’, ‘parallelly’

... omitted output ...

* DONE (future)

The downloaded source packages are in
	‘/tmp/RtmpB3XhHu/downloaded_packages’
````

Install packages necessary for examples presented here:

```r
> install.packages("data.table")
> install.packages("future.apply")
> install.packages("tictoc")
> install.packages("doFuture")
> install.packages("future.batchtools")

... omitted output ...

# Exit R shell
> q()
Save workspace image? [y/n/c]: n
```

## Examples: `sequential`, `multisession`, and `multicore`

`slow_square` example with function `future_lapply`

### R source code:

```r
# This example was adapted from https://grantmcdermott.com/ds4e/parallel.html

library("data.table")
library("future")
library("future.apply")
library("tictoc")

# comment out one future plan
#plan(sequential)    # synchronous
#plan(multisession)  # asynchronous, no forking
plan(multicore)     # asynchronous, with forking

slow_square =
  function(x = 1) {
    x_sq = x^2
    d = data.frame(value = x, value_squared = x_sq)
    Sys.sleep(2)
    return(d)
    }

tic()
availableCores()
future_ex = future_lapply(1:12, slow_square)
toc(log = TRUE)
```

### Batch-Job Submission Script:

```bash
#!/bin/bash
#SBATCH --job-name=Rfuture
#SBATCH --output=test_%j.out
#SBATCH --error=test_%j.err
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=4000
#SBATCH -t 0-00:10               # Runtime in D-HH:MM, minimum of 10 minutes

# load modules
module load R/4.1.0-fasrc01

# set R lirbary path
export R_LIBS_USER=$HOME/apps/R/4.1.0:$R_LIBS_USER

### slow_square_multisession.R
# run R program and keep output and error messages in slow_square_multisession.Rout
Rscript --vanilla future_slow_square.R > future_slow_square.Rout 2>&1

# run R program and keep output in future_slow_square.Rout
# and error messages in error file
Rscript --vanilla future_slow_square.R > future_slow_square.Rout
```

Submit a job to run `future_slow_square.R` in a compute node:

```bash	
sbatch run_future.sbatch
```

### Example output:

```bash
# sequential plan
$ cat future_slow_square.Rout
cgroups.cpuset
             4
24.144 sec elapsed

# multisession plan
$ cat future_slow_square.Rout
cgroups.cpuset
             4
7.178 sec elapsed

# multicore plan
$ cat future_slow_square.Rout
cgroups.cpuset
             4
6.225 sec elapsed
```

## Example: nested futures with `batchtools_slurm` and `multicore`

`batchtools_slurm` submits jobs that will run `future_lapply` using the `multicore`

### R source code:

```r
library("data.table")
library("future")
library("future.apply")
library("future.batchtools")
library("doFuture")
library("tictoc")
options(parallelly.debug = TRUE)

# simple square fucntion to show how to use future
slow_square =
  function(x = 1) {
    x_sq = x^2
    d = data.frame(value = x, value_squared = x_sq)
    Sys.sleep(2)
    return(d)
    }

# sbatch contains the details of the job submission script that will populate
# the template "future_slurm.tmpl"
sbatch <- tweak(batchtools_slurm,
                template = "future_slurm.tmpl",             # file that will be populated
                resources = list(job_name = "Rhybrid",      # name of the job
                                 n_cpu    = 8,              # number of cores per node
                                 queue    = "test",         # which partition
                                 walltime = "00:10:00",     # walltime <hh:mm:ss>
                                 mem_cpu  = 1000,           # memory per core
                                 nodes    = 1,              # number of nodes
                                 log_file = "Rhybrid.log")) # name of log file

# The first level of futures submits jobs to the cluster using batchtools.
# The second level of futures uses multicore, where the number of parallel
# processes is automatically decided based on what the cluster grants to
# each compute node.
plan(list(sbatch,multicore))

# function to allow doFuture to recognize the number available cores
myCores <- registerDoFuture()

# The first level of futures is parallelized using "dopar". future recognizes
# "dopar" as the first level of parallelization and it uses batchtools to
# submit all iterations as jobs to the cluster. In this example, 3 jobs are
# simultaneously submitted
results <- foreach(i=1:3) %dopar% {
  tic()
  c <- availableCores()
  print(c)
  # second level of future uses future_apply to run a multicore process
  future_results <- future_lapply(1:12, slow_square)
  toc(log = TRUE)
}
```

### SLURM batch-job submit script template:

This is the file that is populated and submitted by `batchtools_slurm`.

```bash
## Default resources can be set in your .batchtools.conf.R by defining the variable
## 'default.resources' as a named list.

#!/bin/sh
#SBATCH --job-name <%= resources$job_name %>       ## Name of the job
#SBATCH --ntasks-per-node <%= resources$n_cpu %>   ## number of processes per node
#SBATCH --partition <%= resources$queue %>         ## Job queue
#SBATCH --time <%= resources$walltime %>           ## walltime in hh:mm:ss
#SBATCH --mem-per-cpu <%=resources$mem_cpu %>      ## min memory per core
#SBATCH --nodes <%= resources$nodes %>             ## if 1 put load on one node
#SBATCH --output <%= resources$log_file %>         ## Output is sent to logfile, stdout + stderr by default

## Export value of DEBUGME environment var to slave
export DEBUGME=<%= Sys.getenv("DEBUGME") %>

# Load required software modules
module load R/4.1.0-fasrc01

# Set up R library
export R_LIBS_USER=$HOME/apps/R/4.1.0:$R_LIBS_USER

# Run R process
Rscript -e 'batchtools::doJobCollection("<%= uri %>")'
```

### Batch-Job Submission Script:

This is the file that you submit.

```bash
#!/bin/bash
#SBATCH -J multinode
#SBATCH -o test.out
#SBATCH -e test.err
#SBATCH -p test
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 30
#SBATCH --mem-per-cpu=4000

# Load required software modules
module load R/4.1.0-fasrc01

# Set up R library
export R_LIBS_USER=$HOME/apps/R/4.1.0:$R_LIBS_USER

# Run program
Rscript --vanilla future_hybrid.R > future_hybrid.Rout 2>&1
```

Submit a job to run `future_hybrid.R`:

```bash	
sbatch run_hybrid.sbatch
```

### Example output:

The output from the R code `future_hybrid.R`:

```bash
$ cat future_hybrid.Rout
Loading required package: foreach
cgroups.cpuset
             8
4.51 sec elapsed
cgroups.cpuset
             8
4.428 sec elapsed
cgroups.cpuset
             8
4.432 sec elapsed
```

And the output from `batchtools_slurm`:

```bash
$ cat Rhybrid.log
/n/home_rc/paulasan/myDocNotes/R/future/multinode/.future/20220523_121847-7EZ8pY/doFuture-3_759656388/jobs/job4926a5fda534b607e0d66991d6be0b0a.rds
### [bt]: This is batchtools v0.9.15
### [bt]: Starting calculation of 1 jobs
### [bt]: Setting working directory to '/n/home_rc/paulasan/myDocNotes/R/future/multinode'
Loading required package: doFuture
Loading required package: foreach
Loading required package: future
Loading required package: tictoc
### [bt]: Memory measurement disabled
### [bt]: Starting job [batchtools job.id=1]
### [bt]: Setting seed to 10696 ...

### [bt]: Job terminated successfully [batchtools job.id=1]
### [bt]: Calculation finished!

### [bt]: Job terminated successfully [batchtools job.id=1]
### [bt]: Calculation finished!

### [bt]: Job terminated successfully [batchtools job.id=1]
### [bt]: Calculation finished!
```

Note that the script `run_hybrid.sbatch` that you submit may request less `--ntasks-per-node` than the script submitted by `batchtools_slurm`. In this example, `run_hybrid.sbatch` requests `--ntasks-per-node=1`, but the template `future_slurm.tmpl` is populated with `--ntasks-per-node=8` (`n_cpu=8` in `future_hybrid.R`). We can use `sacct` to see how each job requests the appropriate number of cores:

```bash
$ sacct
JobID           JobName  Partition    Account  AllocCPUS      State ExitCode
------------ ---------- ---------- ---------- ---------- ---------- --------
11411804      multinode       test   rc_admin          1    RUNNING      0:0
11411804.ba+      batch              rc_admin          1    RUNNING      0:0
11411804.ex+     extern              rc_admin          1    RUNNING      0:0
11411805        Rhybrid       test   rc_admin          8  COMPLETED      0:0
11411805.ba+      batch              rc_admin          8  COMPLETED      0:0
11411805.ex+     extern              rc_admin          8  COMPLETED      0:0
11411806        Rhybrid       test   rc_admin          8  COMPLETED      0:0
11411806.ba+      batch              rc_admin          8  COMPLETED      0:0
11411806.ex+     extern              rc_admin          8  COMPLETED      0:0
11411807        Rhybrid       test   rc_admin          8  COMPLETED      0:0
11411807.ba+      batch              rc_admin          8  COMPLETED      0:0
11411807.ex+     extern              rc_admin          8  COMPLETED      0:0
```

