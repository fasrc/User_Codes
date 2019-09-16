<p><h3> <img style="float" src="WRF_logo.jpeg" align="middle"> Weather Research and Forecasting (WRF) Model</h4></p>

[The Weather Research and Forecasting (WRF) Model](https://www.mmm.ucar.edu/weather-research-and-forecasting-model) is a next-generation mesoscale numerical weather prediction system designed for both atmospheric research and operational forecasting applications. It features two dynamical cores, a data assimilation system, and a software architecture supporting parallel computation and system extensibility. The model serves a wide range of meteorological applications across scales from tens of meters to thousands of kilometers.

### Configure and compile WRF on the FASRC cluster

(1) Load required software modules (the specific example uses Intel compiler and MPI libraries) 

```bash
module load intel/17.0.4-fasrc01module load impi/2017.2.174-fasrc01module load netcdf/4.1.3-fasrc02module load libpng/1.6.25-fasrc01module load jasper/1.900.1-fasrc02 
```

(2) Define required environment variables

```bash
export NETCDF=${NETCDF_HOME}export JASPERLIB=${JASPER_LIB}export JASPERINC=${JASPER_INCLUDE}unset MPI_LIB
```

(3) Create top-level directory for code and clone the official code repository from Github, e.g.,

```
mkdir WRF_Model
cd WRF_Model/
git clone https://github.com/wrf-model/WRF.git
```

(4) Configure WRF

```bash
cd WRF/
./configure 
checking for perl5... no
checking for perl... found /usr/bin/perl (perl)
Will use NETCDF in dir: /n/helmod/apps/centos7/MPI/intel/17.0.4-fasrc01/impi/2017.2.174-fasrc01/netcdf/4.1.3-fasrc02
HDF5 not set in environment. Will configure WRF for use without.
PHDF5 not set in environment. Will configure WRF for use without.
Will use 'time' to report timing information


If you REALLY want Grib2 output from WRF, modify the arch/Config.pl script.
Right now you are not getting the Jasper lib, from the environment, compiled into WRF.

------------------------------------------------------------------------
Please select from among the following Linux x86_64 options:

  1. (serial)   2. (smpar)   3. (dmpar)   4. (dm+sm)   PGI (pgf90/gcc)
  5. (serial)   6. (smpar)   7. (dmpar)   8. (dm+sm)   PGI (pgf90/pgcc): SGI MPT
  9. (serial)  10. (smpar)  11. (dmpar)  12. (dm+sm)   PGI (pgf90/gcc): PGI accelerator
 13. (serial)  14. (smpar)  15. (dmpar)  16. (dm+sm)   INTEL (ifort/icc)
                                         17. (dm+sm)   INTEL (ifort/icc): Xeon Phi (MIC architecture)
 18. (serial)  19. (smpar)  20. (dmpar)  21. (dm+sm)   INTEL (ifort/icc): Xeon (SNB with AVX mods)
 22. (serial)  23. (smpar)  24. (dmpar)  25. (dm+sm)   INTEL (ifort/icc): SGI MPT
 26. (serial)  27. (smpar)  28. (dmpar)  29. (dm+sm)   INTEL (ifort/icc): IBM POE
 30. (serial)               31. (dmpar)                PATHSCALE (pathf90/pathcc)
 32. (serial)  33. (smpar)  34. (dmpar)  35. (dm+sm)   GNU (gfortran/gcc)
 36. (serial)  37. (smpar)  38. (dmpar)  39. (dm+sm)   IBM (xlf90_r/cc_r)
 40. (serial)  41. (smpar)  42. (dmpar)  43. (dm+sm)   PGI (ftn/gcc): Cray XC CLE
 44. (serial)  45. (smpar)  46. (dmpar)  47. (dm+sm)   CRAY CCE (ftn $(NOOMP)/cc): Cray XE and XC
 48. (serial)  49. (smpar)  50. (dmpar)  51. (dm+sm)   INTEL (ftn/icc): Cray XC
 52. (serial)  53. (smpar)  54. (dmpar)  55. (dm+sm)   PGI (pgf90/pgcc)
 56. (serial)  57. (smpar)  58. (dmpar)  59. (dm+sm)   PGI (pgf90/gcc): -f90=pgf90
 60. (serial)  61. (smpar)  62. (dmpar)  63. (dm+sm)   PGI (pgf90/pgcc): -f90=pgf90
 64. (serial)  65. (smpar)  66. (dmpar)  67. (dm+sm)   INTEL (ifort/icc): HSW/BDW
 68. (serial)  69. (smpar)  70. (dmpar)  71. (dm+sm)   INTEL (ifort/icc): KNL MIC
 72. (serial)  73. (smpar)  74. (dmpar)  75. (dm+sm)   FUJITSU (frtpx/fccpx): FX10/FX100 SPARC64 IXfx/Xlfx

Enter selection [1-75] :
```


Choose, e.g., option 15 to compile a MPI version of WRF with Intel compilers, and then option 1 to select the default nesting:

```
Enter selection [1-75] : 15
------------------------------------------------------------------------
Compile for nesting? (1=basic, 2=preset moves, 3=vortex following) [default 1]: 1

Configuration successful! 
------------------------------------------------------------------------
```

(5) Modify the file “configure.wrf” (around lines 154-155) to read the following. Note that you have to do this each time you run ./configure, because the "configure.wrf" script is overwritten each time:

```bash
DM_FC  =  mpiifort -f90=$(SFC)DM_CC  =  mpiicc -cc=$(SCC) -DMPI2_SUPPORT
```

(6) Compile WRF before WPS!! Compilation will take a while. If you're on an interactive shell, remove the "&" to avoid timing out:

```bash
./compile em_real &> compile_wrf.log
```

This generates the **[compile_wrf.log](compile_wrf.log)** file with details of the build process.

### Configure and compile WPS on the FASRC cluster

The WRF Pre-Processing System (WPS) is a collection
of Fortran and C programs that provides data used as
input to the real.exe and real_nmm.exe programs. There 
are three main programs and a number of auxiliary 
programs that are part of WPS.  Both the ARW and NMM 
dynamical cores in WRF are supported by WPS.

One needs to configure and compile WRF following the above example steps, before attempting to build WPS.

(1) Clone WPS from the official Github repository

```bash
git clone https://github.com/wrf-model/WPS.git
```
(2) Configure WPS. Choose, e.g., option 19 to compile a MPI version with GRIB2 capabilities:

```bash
cd WPS\
./configure
Will use NETCDF in dir: /n/helmod/apps/centos7/MPI/intel/17.0.4-fasrc01/impi/2017.2.174-fasrc01/netcdf/4.1.3-fasrc02
Found what looks like a valid WRF I/O library in ../WRF
Found Jasper environment variables for GRIB2 support...
  $JASPERLIB = /n/helmod/apps/centos7/Comp/intel/17.0.4-fasrc01/jasper/1.900.1-fasrc02/lib64
  $JASPERINC = /n/helmod/apps/centos7/Comp/intel/17.0.4-fasrc01/jasper/1.900.1-fasrc02/include
------------------------------------------------------------------------
Please select from among the following supported platforms.

   1.  Linux x86_64, gfortran    (serial)
   2.  Linux x86_64, gfortran    (serial_NO_GRIB2)
   3.  Linux x86_64, gfortran    (dmpar)
   4.  Linux x86_64, gfortran    (dmpar_NO_GRIB2)
   5.  Linux x86_64, PGI compiler   (serial)
   6.  Linux x86_64, PGI compiler   (serial_NO_GRIB2)
   7.  Linux x86_64, PGI compiler   (dmpar)
   8.  Linux x86_64, PGI compiler   (dmpar_NO_GRIB2)
   9.  Linux x86_64, PGI compiler, SGI MPT   (serial)
  10.  Linux x86_64, PGI compiler, SGI MPT   (serial_NO_GRIB2)
  11.  Linux x86_64, PGI compiler, SGI MPT   (dmpar)
  12.  Linux x86_64, PGI compiler, SGI MPT   (dmpar_NO_GRIB2)
  13.  Linux x86_64, IA64 and Opteron    (serial)
  14.  Linux x86_64, IA64 and Opteron    (serial_NO_GRIB2)
  15.  Linux x86_64, IA64 and Opteron    (dmpar)
  16.  Linux x86_64, IA64 and Opteron    (dmpar_NO_GRIB2)
  17.  Linux x86_64, Intel compiler    (serial)
  18.  Linux x86_64, Intel compiler    (serial_NO_GRIB2)
  19.  Linux x86_64, Intel compiler    (dmpar)
  20.  Linux x86_64, Intel compiler    (dmpar_NO_GRIB2)
  21.  Linux x86_64, Intel compiler, SGI MPT    (serial)
  22.  Linux x86_64, Intel compiler, SGI MPT    (serial_NO_GRIB2)
  23.  Linux x86_64, Intel compiler, SGI MPT    (dmpar)
  24.  Linux x86_64, Intel compiler, SGI MPT    (dmpar_NO_GRIB2)
  25.  Linux x86_64, Intel compiler, IBM POE    (serial)
  26.  Linux x86_64, Intel compiler, IBM POE    (serial_NO_GRIB2)
  27.  Linux x86_64, Intel compiler, IBM POE    (dmpar)
  28.  Linux x86_64, Intel compiler, IBM POE    (dmpar_NO_GRIB2)
  29.  Linux x86_64 g95 compiler     (serial)
  30.  Linux x86_64 g95 compiler     (serial_NO_GRIB2)
  31.  Linux x86_64 g95 compiler     (dmpar)
  32.  Linux x86_64 g95 compiler     (dmpar_NO_GRIB2)
  33.  Cray XE/XC CLE/Linux x86_64, Cray compiler   (serial)
  34.  Cray XE/XC CLE/Linux x86_64, Cray compiler   (serial_NO_GRIB2)
  35.  Cray XE/XC CLE/Linux x86_64, Cray compiler   (dmpar)
  36.  Cray XE/XC CLE/Linux x86_64, Cray compiler   (dmpar_NO_GRIB2)
  37.  Cray XC CLE/Linux x86_64, Intel compiler   (serial)
  38.  Cray XC CLE/Linux x86_64, Intel compiler   (serial_NO_GRIB2)
  39.  Cray XC CLE/Linux x86_64, Intel compiler   (dmpar)
  40.  Cray XC CLE/Linux x86_64, Intel compiler   (dmpar_NO_GRIB2)

Enter selection [1-40] : 19
------------------------------------------------------------------------
Configuration successful. To build the WPS, type: compile
------------------------------------------------------------------------

Testing for NetCDF, C and Fortran compiler

This installation NetCDF is 64-bit
C compiler is 64-bit
Fortran compiler is 64-bit
```

(3) Modify the "configure.wps" around lines 65-66 to read the following:

```bash
DM_FC               = mpiifort
DM_CC               = mpiicc
```

(4) Compile WPS. If you're on an interactive shell, remove the "&" to avoid timing out:

```bash
./compile &> compile_wps.log
```

This generates the **[compile_wps.log](compile_wps.log)** file with details of the build process.


### References:

1. [The Weather Research and Forecasting (WRF) Model (official website)] (https://www.mmm.ucar.edu/weather-research-and-forecasting-model)
2. [WRF user's page](http://www2.mmm.ucar.edu/wrf/users)
3. [WRF - Github repo](https://github.com/wrf-model/WRF)
4. [WRF - User Guide](http://www2.mmm.ucar.edu/wrf/users/docs/user_guide_v4/contents.html)
5. [WPS - Github repo](https://github.com/wrf-model/WPS)