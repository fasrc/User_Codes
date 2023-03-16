# R packages with Spack 

**NOTE**: These instructions apply to users who use R from the command line. If
you use R in RStudio, refer to FASRC [RStudio
docs](https://docs.rc.fas.harvard.edu/kb/rstudio-server-vs-rstudio-desktop/)

## R changes from CentOS 7 to Rocky 8

- On Rocky 8, FASRC will no longer have software, such as `udunits`, `gdal`,
`geos`, available through `module load` command that are needed to install R
packages. Instead, we will be using Spack to install these R packages.

- There is no need to set `R_LIBS_USER` neither to unset `R_LIBS_SITE`

- A basic R module is available through HeLmod (e.g. `module load R`). However,
  whenever you install R packages with Spack, R is also installed within Spack.

- For more details about the operating system changes, see the [Rocky 8
    Transition
    Guide](https://docs.rc.fas.harvard.edu/kb/rocky-8-transition-guide/)

## Requirements

First, to install spack, follow our [Spack Install and
Setup](User_Codes/Documents/Software/Spack.md) instructions.

Once Spack is installed, then you can install the R packages with Spack from
the command line.

## Examples

For all of these examples, ensure you are in a compute node by requesting an
interactive job:

```bash
[jharvard@holylogin01 spack]$ salloc -p test -t 2:00:00 --mem 8000
```

Installing R packages with spack is fairly simple. The main steps are:

```bash
[jharvard@holy2c02302 spack]$ spack install package_name  # install software
[jharvard@holy2c02302 spack]$ spack load package_name     # load software to your environment
[jharvard@holy2c02302 spack]$ R                           # launch R
> library(package_name)                                   # load package within R
```

We added some examples below. See [Spack
list](https://spack.readthedocs.io/en/latest/package_list.html) for a full list
of software offered by Spack.

### `raster`

```bash
# install R packages with spack
[jharvard@holy2c02302 spack]$ spack install r-codetools
[jharvard@holy2c02302 spack]$ spack install r-raster

# load spack packages
[jharvard@holy2c02302 spack]$ spack load r-codetools
[jharvard@holy2c02302 spack]$ spack load r-raster

# launch R and load libraries
[jharvard@holy2c02302 spack]$ R
> library(raster)
Loading required package: sp
```

### `rgdal`

`rgdal` is a heavy package and can take several minutes (~50 min) to install

```bash
# install R packages with spack
[jharvard@holy2c02302 spack]$ spack install r-rgdal

# load spack packages
[jharvard@holy2c02302 spack]$ spack load r-rgdal

# launch R and load libraries
[jharvard@holy2c02302 spack]$ R
> library(rgdal)
Please note that rgdal will be retired by the end of 2023,
plan transition to sf/stars/terra functions using GDAL and PROJ
at your earliest convenience.

rgdal: version: 1.5-32, (SVN revision 1176)
Geospatial Data Abstraction Library extensions to R successfully loaded
Loaded GDAL runtime: GDAL 3.5.3, released 2022/10/21
Path to GDAL shared files: /n/holylabs/LABS/jharvard_lab/Users/jharvard/spack/opt/spack/linux-rocky8-skylake_avx512/gcc-8.5.0/gdal-3.5.3-uecjo2dxdphgs6kwb7h3m7dnwim2t4cv/share/gdal
 GDAL does not use iconv for recoding strings.
GDAL binary built with GEOS: TRUE
Loaded PROJ runtime: Rel. 8.2.1, January 1st, 2022, [PJ_VERSION: 821]
Path to PROJ shared files: /n/holylabs/LABS/jharvard_lab/Users/jharvard/spack/opt/spack/linux-rocky8-skylake_avx512/gcc-8.5.0/proj-8.2.1-nx7ka5a6mb4gncxqhhecu5ujsqso7iz4/share/proj
PROJ CDN enabled: FALSE
Linking to sp version:1.5-1
To mute warnings of possible GDAL/OSR exportToProj4() degradation,
use options("rgdal_show_exportToProj4_warnings"="none") before loading sp or rgdal.
```

### `rgeos`

```bash
# install R packages with spack
[jharvard@holy2c02302 spack]$ spack install r-rgeos

# load spack packages
[jharvard@holy2c02302 spack]$ spack load r-rgeos

# launch R and load libraries
[jharvard@holy2c02302 spack]$ R
> library(rgeos)
Loading required package: sp
rgeos version: 0.5-9, (SVN revision 684)
 GEOS runtime version: 3.9.1-CAPI-1.14.2
 Please note that rgeos will be retired by the end of 2023,
plan transition to sf functions using GEOS at your earliest convenience.
 GEOS using OverlayNG
 Linking to sp version: 1.5-1
 Polygon checking: TRUE
```

### `rstan`

This software may take ~20 min to install

```bash
# install R packages with spack
[jharvard@holy2c02302 spack]$ spack install r-rstan

# load spack packages
[jharvard@holy2c02302 spack]$ spack load r-rstan

# launch R and load libraries
[jharvard@holy2c02302 spack]$ R
> library(rstan)
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


