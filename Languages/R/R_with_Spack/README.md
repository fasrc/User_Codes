# R packages with Spack 

> [!WARNING]
> These instructions apply for R from the **command line**. If you use R in
> RStudio, refer to FASRC [RStudio
> docs](https://docs.rc.fas.harvard.edu/kb/r-and-rstudio/).

## Purpose

Examples of how to install geospatial and bioinformatics R packages using Spack.

## Requirements

Refer to Steps 1 and 2 in the [R with
Spack](https://docs.rc.fas.harvard.edu/kb/r-and-rstudio/#R_with_Spack) main
documentation for requirements.

## Content

* How to install
  * [`glmnet`](#glmnet)
  * [`rstan`](#rstan)
  * [`sf`](#sf)
* Install many R packages in [Spack environment](#spack-environments)
* Install [specific version of R](#installing-specific-versions-of-r)
* [Submit a slurm batch job](#submit-a-slurm-batch-job)

## Examples

> [!IMPORTANT]
> Packages `rgdal`, `raster`, and `rgeos` have been deprecated. Instead, use:
>   * `rgdal` -> `sf` or `terra`
>   * `raster` -> `terra`
>   * `rgeos` -> `sf` or `terra`

We added some examples below. See [Spack
list](https://spack.readthedocs.io/en/latest/package_list.html) for a full list
of software offered by Spack.

If you have trouble installing packages, change Spack's [default
architecture](../../../Documents/Software/Spack.md#default-architecture).

### `glmnet`

```bash
# install R packages with spack
[jharvard@holy2c02302 spack]$ spack install r-glmnet

# load spack packages
[jharvard@holy2c02302 spack]$ spack load r-glmnet

# launch R and load libraries
[jharvard@holy2c02302 spack]$ R
> library(glmnet)
Loading required package: Matrix
Loaded glmnet 4.1-7
```

### `rstan`

This software requires ~12G of memory and may take ~1 hour to install. Make sure
you request enough memory and time in your interactive job to install the R
package `rstan`.

```bash
# install R packages with spack
[jharvard@holy2c02302 spack]$ spack install r-rstan
[jharvard@holy2c02302 spack]$ spack install r-codetools

# load spack packages
[jharvard@holy2c02302 spack]$ spack load r-rstan
[jharvard@holy2c02302 spack]$ spack load r-codetools

# launch R and load libraries
[jharvard@holy2c02302 spack]$ R
> library(rstan)
Loading required package: StanHeaders
Loading required package: ggplot2
rstan (Version 2.21.8, GitRev: 2e1f913d3ca3)
For execution on a local, multicore CPU with excess RAM we recommend calling
options(mc.cores = parallel::detectCores()).
To avoid recompilation of unchanged Stan programs, we recommend calling
rstan_options(auto_write = TRUE)
```

### `seurat`

For Seurat, we recommend using [RStudio
Server](https://docs.rc.fas.harvard.edu/kb/r-and-rstudio/#RStudio_Server)
instructions.

### `sf`

This software may take ~60 min to install

```bash
# install R packages with spack
[jharvard@holy2c02302 spack]$ spack install r-sf

# load spack packages
[jharvard@holy2c02302 spack]$ spack load r-sf

# launch R and load libraries
[jharvard@holy2c02302 spack]$ R
> library(sf)
Linking to GEOS 3.12.2, GDAL 3.9.0, PROJ 9.2.1; sf_use_s2() is TRUE
```

## Spack environments

Alternatively, if you would like to install many R packages, you can create a
[Spack environment](https://spack.readthedocs.io/en/latest/environments.html) --
Spack environment are similar to conda/mamba environments. After cloning and
sourcing spack, then:

```bash
# create environment
spack env create R_packages
==> Created environment R_packages in: /n/holyscratch01/jharvard_lab/Lab/jharvard/spack_installs/R_spack/var/spack/environments/R_packages
==> Activate with: spack env activate R_packages

# activate environment
spack env activate -p R_packages

# add packages to environment
spack add r-glmnet r-rstan r-sf
==> Adding r-glmnet to environment R_packages
==> Adding r-rstan to environment R_packages
==> Adding r-sf to environment R_packages

# install packages (this can take 2+ hours)
spack install

# use R packages
R
> library(rstan)
Loading required package: StanHeaders
Loading required package: ggplot2
rstan (Version 2.21.8, GitRev: 2e1f913d3ca3)
For execution on a local, multicore CPU with excess RAM we recommend calling
options(mc.cores = parallel::detectCores()).
To avoid recompilation of unchanged Stan programs, we recommend calling
rstan_options(auto_write = TRUE)
```

## Installing specific versions of R

To install a specific version of r, add `@` and the version to the `spack
install r` command. For example, to install version 3.5.0:

```bash
# install R with spack
[jharvard@holy2c02302 spack]$ spack install r@3.5.0

# load specific R version
[jharvard@holy2c02302 spack]$ spack load r@3.5.0

# launch R
[jharvard@holy2c02302 spack]$ R

R version 3.5.0 (2018-04-23) -- "Joy in Playing"
Copyright (C) 2018 The R Foundation for Statistical Computing
Platform: x86_64-pc-linux-gnu (64-bit)
```

> [!IMPORTANT]
> If you get a similar error:
> ```bash
> ==> Error: concretization failed for the following reasons:
>
>   1. Cannot satisfy 'curl@:7'
>   2. Cannot satisfy 'curl@:7'
>        required because r depends on curl@:7 when @:4.2
>          required because r@3.5.0 requested explicitly
> ```
> This is because `curl` version < 8 has been labeled as deprecated on Spack.
> Instead, you can use the operating system's `curl`. To use the operating
> system's `curl`, follow Spack's instructions on [Finding external
> packages](../../../Documents/Software/Spack.md#finding-external-packages)

## Submit a slurm batch job

When you submit a slurm job and you need to use an R package that was installed
with Spack, you need to:

1. Source Spack within the slurm submission script

```bash
. /n/holylabs/LABS/jharvard_lab/Users/jharvard/spack/share/spack/setup-env.sh
```

2. Load necessary R packages

```bash
spack load r-codetools
spack load r-rstan
```

3. Run the program with either `Rscript` or `R CMD BATCH`

```bash
Rscript --vanilla r_spack_load_libs.R > r_spack_load_libs.Rout
```

Putting steps 1-3 together in [`runscript_r_spack.sh`](runscript_r_spack.sh), the slurm batch script becomes:

https://github.com/fasrc/User_Codes/blob/c9b0e1d6bb45750252ad2fb42639618142d8d3c3/Languages/R/runscript_r_spack.sh#L1-L30

To load R packages installed with Spack in an R script, it works as usual:

https://github.com/fasrc/User_Codes/blob/d4e6c06b8160b2df44601397700f1caec05260b6/Languages/R/r_spack_load_libs.R#L1-L15

Finally, submit the job that executes the R script [`r_spack_load_libs.R`](r_spack_load_libs.R), submit a slurm job with:

```bash
sbatch runscript_r_spack.sh
```
