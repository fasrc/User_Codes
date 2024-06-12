# R packages with Spack 

> [!WARNING]
> These instructions apply for R from the command line. If you use R in RStudio, refer to FASRC [RStudio docs](https://docs.rc.fas.harvard.edu/kb/r-and-rstudio/).

## Purpose

Examples of how to install geospatial and bioinformatics R packages using Spack.

## Requirements

Refer to Steps 1 and 2 in the [R with
Spack](https://docs.rc.fas.harvard.edu/kb/r-and-rstudio/#R_with_Spack) main
documentation for requirements.

## Examples

> [!IMPORTANT]
> Packages `rgdal, `raster`, and `rgeos` have been deprecated. Instead, use:

For all of these examples, ensure you are in a compute node by requesting an
interactive job (Step 2 of [R with
Spack](https://docs.rc.fas.harvard.edu/kb/r-and-rstudio/#R_with_Spack).

We added some examples below. See [Spack
list](https://spack.readthedocs.io/en/latest/package_list.html) for a full list
of software offered by Spack.

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
Loaded glmnet 4.1-4
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
rstan (Version 2.21.7, GitRev: 2e1f913d3ca3)
For execution on a local, multicore CPU with excess RAM we recommend calling
options(mc.cores = parallel::detectCores()).
To avoid recompilation of unchanged Stan programs, we recommend calling
rstan_options(auto_write = TRUE)
```

### `seurat`

This software may take ~30 min to install

```bash
# install R packages with spack
[jharvard@holy2c02302 spack]$ spack install r-seurat

# load spack packages
[jharvard@holy2c02302 spack]$ spack load r-seurat

# launch R and load libraries
[jharvard@holy2c02302 spack]$ R
> library(Seurat)
Attaching SeuratObject
```

### `sf`

This software may take ~20 min to install

```bash
# install R packages with spack
[jharvard@holy2c02302 spack]$ spack install r-sf

# load spack packages
[jharvard@holy2c02302 spack]$ spack load r-sf

# launch R and load libraries
[jharvard@holy2c02302 spack]$ R
> library(sf)
Linking to GEOS 3.9.1, GDAL 3.5.3, PROJ 8.2.1; sf_use_s2() is TRUE
```

## Installing specific versions of R

To install a specific version of r, simply add `@` and the version to the `spack
install r` command:

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

## Submitting a slurm job

When you submit a slurm job and you need to use an R package that was installed
with Spack, you need to:

1. Source Spack within the slurm submission script

```bash
. /n/holylabs/LABS/jharvard_lab/Users/jharvard/spack/share/spack/setup-env.sh
```

2. Load necessary R packages

```bash
spack load r-codetools
spack load r-raster
```

3. Run the program with either `Rscript` or `R CMD BATCH`

```bash
Rscript --vanilla r_spack_load_libs.R > r_spack_load_libs.Rout
```

Putting items 1-3 together in `runscript_r_spack.sh`, the slurm batch script becomes:

https://github.com/fasrc/User_Codes/blob/c9b0e1d6bb45750252ad2fb42639618142d8d3c3/Languages/R/runscript_r_spack.sh#L1-L30

To load R packages installed with Spack in an R script, it works as usual:

https://github.com/fasrc/User_Codes/blob/d4e6c06b8160b2df44601397700f1caec05260b6/Languages/R/r_spack_load_libs.R#L1-L15

Finally, submit the job that executes the R script `r_spack_load_libs.R`, submit a slurm job with:

```bash
sbatch runscript_r_spack.sh
```
