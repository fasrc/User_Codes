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

