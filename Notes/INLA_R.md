# INLA

INLA needs to be installed with the correct version, otherwise the installation
may look correct, but loading the package will crash. For more details, see
[INLA install doc](https://www.r-inla.org/download-install) under section
"Version Compatibility".

| R version | INLA version |
| --------  | ------------ |
| 4.3       | 23.05.30     |
| 4.2       | 22.05.07     |
| 4.1       | 22.05.03     |

## Example install

For RStudio Server with R version 4.2.3, Bioconductior 3.16, RStudio 2023.03.0

```R
> library(remotes)
> remotes::install_version("INLA", version="22.05.07",repos=c(getOption("repos"),INLA="https://inla.r-inla-download.org/R/testing"), dep=TRUE)
> library(INLA)
Loading required package: Matrix
Loading required package: foreach
Loading required package: parallel
Loading required package: sp
This is INLA_22.05.07 built 2022-05-07 09:52:03 UTC.
 - See www.r-inla.org/contact-us for how to get help.
 - To enable PARDISO sparse library; see inla.pardiso()
```
