## Installing *ENMTools*, *ecospat*, *raster*, *rJava*

The below instructions are intended to help you install the R packages [ENMTools](https://github.com/danlwarren/ENMTools), [rgeos](https://cran.r-project.org/web/packages/rgeos/index.html), [ecospat](https://r-spatial.github.io/sf), [raster](https://cran.r-project.org/web/packages/raster/index.html), and [rJava](https://cran.r-project.org/web/packages/rJava/index.html) on the FAS Cannon cluster.

### Load the required software modules:

First, you will need to load the required software modules.

```bash
module load gcc/9.3.0-fasrc01 R/4.1.0-fasrc01
module load gdal/3.2.2-fasrc01 geos/3.9.1-fasrc01 udunits/2.2.26-fasrc01
```

### Set the required environmental variables:

Next, you will need to set the required environment variables. Please, notice that first you will need to unset the <code>R\_LIBS\_SITE</code> environmental variable, which points to preinstalled R packages compiled with the default (system) GNU compilers on the cluster. For best results, the <code>rgdal</code>, <code>rgeos</code>, and <code>sf</code> need to be compiled consistently together with any dependencies with a newer GNU compilers, e.g., GCC version 9.3.0. The below code also creates space for a customized R library under the <code>~/apps/R/4.1.0</code> directory. 

```bash
unset R_LIBS_SITE
mkdir -p $HOME/apps/R/4.1.0
export R_LIBS_USER=$HOME/apps/R/4.1.0:$R_LIBS_USER
```

You will need to create a `Makevars` file that will point out to the correct compiler and compiler flags for `CXX14`

```bash
$ mkdir $HOME/.R
$ echo -e '## C++ flags \nCXX14=g++ \nCXX14PICFLAGS=-fPIC \nCXX14STD=-std=c++14' >> $HOME/.R/Makevars
$ cat $HOME/.R/Makevars
## C++ flags
CXX14=g++
CXX14PICFLAGS=-fPIC
CXX14STD=-std=c++14
```

### Installing:

First, request a compute node because this installation takes a long time and requires considerable memory:

```bash
salloc -p test --time=4:00:00 --mem=8000
```

Install the packages from within the R shell:

```r
# Install ENMTools and its dependencies
> install.packages("sp")
> install.packages("rgdal")
> install.packages("rgeos")
> install.packages("sf")                 # this takes a while (~10 min)
> install.packages("fields")             # this takes a while (~10 min)
> install.packages("glmnet")
> install.packages("gridExtra")
> install.packages("spThin")             # this takes a while (~5 min)
> install.packages("rangeModelMetadata") # this takes more than 1h!
> install.packages("ENMeval")
> install.packages("ENMTools")

# Install ecostap and its dependencies
> install.packages(https://cran.r-project.org/src/contrib/Archive/randomForest/randomForest_4.6-14.tar.gz, repos=NULL, type="source")
> install.packages("biomod2")
> install.packages("ecospat")            # this takes a while (~20 min)

# Install raster
> install.packages("raster")

# exit R shell
> q()
Save workspace image? [y/n/c]: n
```

To install `rJava`, first you need to setup the Java environement in the cluster (i.e., outside of the R shell)

```bash
module load Java/1.8
R CMD javareconf -e
```

Back in the R shell:

```r
# Install rJava
> install.packages("rJava")

# Exit R shell
> q()
Save workspace image? [y/n/c]: n
```

### Remove `Makevars` file

You need to delete `$HOME/.R/Makevars` so that it will not interfere with future package installations

```bash
rm $HOME/.R/Makevars
```

