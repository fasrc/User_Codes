# Using Packrat environment in R 

This document provides you with necessary steps to set up a Packrat
environment in R that will allow you to install the desired R packages
in an isolated, portable, and reproducible way. Packrat is a
dependency management system for projects and their R package
[dependencies](https://cran.r-project.org/web/packages/packrat/index.html). Packrat
can be easily setup in R using its command line interface (cli) or the
[OpenOnDemand VDI portal](https://rcood.rc.fas.harvard.edu).

The steps below are applicable for both types of interfaces. For more
information on packrat and its walkthrough guide, see [rstudio
packrat](https://rstudio.github.io/packrat/).

## Setting up Packrat and installing an R package using that

Complete the below steps to install Packrat and use that to install an R package:

* Go to a compute node, if on cli:
```bash
salloc -p test -t 60 -n1 --mem 4000
```

* Load the required R module, if on cli. On the RStudio app on VDI,
  the R module is loaded in the form:
```bash
module load R/4.3.1-fasrc01
```

* Set up a local [R library](https://docs.rc.fas.harvard.edu/kb/r-packages/):
```bash
mkdir $HOME/apps/R_4.3.1
export R_LIBS_USER=$HOME/apps/R_4.3.1:$R_LIBS_USER 
```

* Create a Packrat project directory:
```bash
cd $HOME/apps/R_4.3.1
mkdir packrat_r
```

* Start R, if on cli, and install Packrat:
```R
# --- Install Packrat in R ---
install.packages("packrat")
# --- Choose 70 for CRAN mirror ---
```

* Initialize the Packrat project:
```bash
packrat::init("~/apps/R_4.3.1/packrat_r")
```

* Use snapshot feature to save changes in Packrat:
```bash
packrat::snapshot() 
```
* Check the status of Packrat:
```bash
packrat::status()
```

* Turn Packrat on and ensure that the directory path shown matches
  with the desired Packrat project location:
```bash
packrat::on()
```

* Install the desired R package into your Packrat location, e.g.:
```bash
install.packages(“viridis”)
```
