# R Parallel

## Purpose

Here, we briefly explain different ways to use R in parallel on the Harvard University FASRC Cannon cluster.


* For R basics, refer to [R-Basics](https://docs.rc.fas.harvard.edu/kb/r-basics/)
* For R Parallel, refer to [R-Parallel](https://docs.rc.fas.harvard.edu/kb/r-parallel/)

### Processing large datasets

* [Working with large data that does not fit into memory](Large_Data_Processing_R/data_format/NYC_lyft.md)
* [Processing Single instruction multiple data problem on shared and distributed memory systems](Large_Data_Processing_R/parallel_computation/R_embarrassingly_parallel.md)


### Single-node, multi-core (shared memory)

* Package `parallel`
  * FAS RC embarrassingly parallel [documentation](Large_Data_Processing_R/parallel_computation/R_embarrassingly_parallel.md)
  * FAS RC embarrassingly parallel [Cannon example](Large_Data_Processing_R/parallel_computation/R/2_compute_pi_parLapply.R) (using `parLapply`)
  * FAS RC Embarrassingly parallel [VDI example](Large_Data_Processing_R/parallel_computation/R_parLapply_vdi.md) (using `parLapply`)
  * [parallel documentation](https://rdrr.io/r/parallel/parallel-package.html)

* Package `future`
  * [Install future on Cannon](future/README.md)
  * [Example](future/future_slow_square.R) of `multisession` (not shared memory) and `multicore` (shared memory) and its [submit script](future/run_future.sbatch)
  * [future documentation](https://future.futureverse.org/)
 
### Multi-node, distributed memory

* Package `Rmpi`
  * [Install Rmpi on Cannon](Rmpi/README.md)
  * [Example](Rmpi/mpi_test.R) and its [submit script](Rmpi/run.sbatch)
  * [Rmpi documentation](https://cran.r-project.org/web/packages/Rmpi/index.html)

* Package `pbdMPI` (programming big data MPI) 
  * [Install pbdMPI on Cannon](pbdMPI/README.md)
  * [Examples](pbdMPI/README.md) based on the `pbdMPI` demos – after installing `pbdMPI` package, all demos can be found in your R library folder `$HOME/apps/R/4.0.5/pbdMPI/demo`
  * [pbdMPI documentation](https://rdrr.io/cran/pbdMPI/) and [GitHub](https://github.com/RBigData/pbdMPI)
  * [pbdR website](https://pbdr.org/packages.html)

### Hybrid: Multi-node + shared-memory

Using nested futures and package `future.batchtools`, we can perform a multi-node and multi-core job.

* Package `future` and `future.batchtools`
  * [Install future and future.batchtools on Cannon](future/README.md)
  * [Example](future/future_hybrid.R) and its [submit script](future/run_hybrid.sbatch)
  * [future documentation](https://future.futureverse.org/) and [GitHub](https://github.com/HenrikBengtsson/future)
  * [future.batchtools documentation](https://future.batchtools.futureverse.org/) and [GitHub](https://github.com/HenrikBengtsson/future.batchtools)

