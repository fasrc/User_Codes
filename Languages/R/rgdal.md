## Installing *sp*, *rgdal*, *rgeos*, and *sf*

The below instructions are intended to help you install the R packages [rgdal](https://cran.r-project.org/web/packages/rgdal/index.html), [rgeos](https://cran.r-project.org/web/packages/rgeos/index.html), and [sf](https://r-spatial.github.io/sf) on the FAS cluster.

### Load the required software modules:

First, you will need to load the required software modules.

```bash
module load R/4.1.0-fasrc01
module load gdal/2.3.0-fasrc01
module load geos/3.6.2-fasrc01
module load proj/5.0.1-fasrc01
module load gcc/9.3.0-fasrc01 udunits/2.2.26-fasrc01
```

### Set the required environmental variables:

Next, you will need to set the required environment variables. Please, notice that first you will need to unset the <code>R\_LIBS\_SITE</code> environmental variable, which points to preinstalled R packages compiled with the default (system) GNU compilers on the cluster. For best results, the <code>rgdal</code>, <code>rgeos</code>, and <code>sf</code> need to be compiled consistently together with any dependencies with a newer GNU compilers, e.g., GCC version 9.3.0. The below code also creates space for a customized R library under the <code>~/apps/R/4.1.0</code> directory. 

```bash
unset R_LIBS_SITE
mkdir -p $HOME/apps/R/4.1.0
export R_LIBS_USER=$HOME/apps/R/4.1.0:$R_LIBS_USER
```

### Installing:

To install the packages from within the R shell run:

```r
> install.packages("sp")
> install.packages("rgdal")
> install.packages("rgeos")
> install.packages("sf")
```

### Troubleshooting:

If while installing `sf` you get an error

```bash
Error e1071: e1071 is not a valid installed package
```
then, install this package in the R shell:

```r
> install.packages("e1071")
```
