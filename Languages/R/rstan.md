## Installing *rstan*

The below instructions are intended to help you install the R package [rstan](https://cran.r-project.org/web/packages/rstan/index.html).

### Load the required software modules:

First, you will need to load the required software modules.

```bash
module load R/4.1.0-fasrc01 gcc/10.2.0-fasrc01 automake/1.16-fasrc01
```

### Set the required environmental variables:

Next, you will need to set the required environment variables. Please, notice that first you will need to unset the <code>R\_LIBS\_SITE</code> environmental variable, which points to preinstalled R packages compiled with the default (system) GNU compilers on the cluster. The below code also creates space for a customized R library under the <code>$HOME/apps/R/4.1.0</code> directory. 

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

First, request a compute node because this installation takes more than 1h:

```bash
salloc -p test --time=2:00:00 --mem=4000
```

Install the packages from within the R shell. 

```r
install.packages("Rcpp")
install.packages("miniUI")
install.packages("rstan")
# exit R shell
> q()
Save workspace image? [y/n/c]: n
```

### Remove `Makevars` file

You probably need to delete `$HOME/.R/Makevars` so that it will not interfere with future package installations. If you need to keep this file for some reason, then delete the lines that were added for this specific package installation.

```bash
# this deletes the file $HOME/.R/Makevars
rm $HOME/.R/Makevars
```

Alternatively, open the file `$HOME/.R/Makevars` and delete the following lines:

```bash
## C++ flags
CXX14=g++
CXX14PICFLAGS=-fPIC
CXX14STD=-std=c++14
```

