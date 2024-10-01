# OpenFOAM and RheoTool

### Introduction

[OpenFOAM](https://www.openfoam.com/) (Open Field Operation and Manipulation) is an open-source Computational Fluid Dynamics (CFD) toolbox that is extensively used for the simulation of fluid flow, heat transfer, and associated physical phenomena. Developed primarily in C++, OpenFOAM allows users to customize their simulations through the development of custom solvers and utilities, providing a versatile environment for tackling a wide range of complex problems in fluid dynamics and beyond. It is particularly valued in academic research and industrial applications for its flexibility, extensive library of solvers, and the ability to handle both steady and unsteady flows, multiphase flows, turbulence modeling, and more.

[RheoTool](https://github.com/fppimenta/rheoTool) is a specialized OpenFOAM toolbox designed to simulate and analyze complex rheological behaviors in fluid flow, particularly for non-Newtonian fluids. Developed to extend the capabilities of OpenFOAM, RheoTool provides a suite of solvers and utilities tailored for the study of various rheological models, including those for viscoelastic, shear-thinning, and shear-thickening fluids. It allows researchers and engineers to perform detailed simulations of fluid dynamics in scenarios where traditional CFD tools may fall short, such as in the processing of polymers, food products, or biological materials. RheoTool is highly customizable, enabling users to modify and adapt the toolset to their specific needs. The integration with OpenFOAM makes it a powerful addition to the open-source CFD community, providing enhanced capabilities for studying complex fluid behaviors in both academic and industrial applications.

### Setting up OpenFOAM and RheoTool on the FASRC Cannon cluster

The below instructions are intended to setup OpenFOAM and RheoTool on the cluster.

* Create an OpenFOAM Singularity image, `of90.sif`,  with the provided definition. file `of90.def`

```bash
singularity build of90.sif of90.def
```
**Note:** This will result in the image `of90.sif`

* Open a singularity shell with the image.

```bash
singularity shell of90.sif
Singularity> 
```

* Set up OpenFOAM 9

```bash
Singularity> source /opt/openfoam9/etc/bashrc
```

* Clone the GitHub rheoTool repo 
```bash
Singularity> git clone https://github.com/fppimenta/rheoTool.git
```

*  Install Eigen

```bash
Singularity> cd rheoTool/of90/
Singularity> ./downloadEigen -j $(nproc)
```

* Install PETSc
```bash
Singularity> ./installPetsc -j $(nproc)
```

* Install rheoTool

```bash
Singularity> export EIGEN_RHEO=${HOME}/OpenFOAM/${USER}-9/ThirdParty/Eigen3.2.9
Singularity> export PETSC_DIR=${HOME}/OpenFOAM/${USER}-9/ThirdParty/petsc-3.16.5 
Singularity> export PETSC_ARCH=arch-linux-c-opt
Singularity> cd src/
Singularity> ./Allwmake -j $(nproc)
```
**Note:** `RheoTool` and its dependencies, `PETSc` and `Eigen` are installed in the directory `${HOME}/OpenFOAM/${USER}-9`


### Using the software

* Create a setup file `setup.sh` with the below contents.

```bash
source /opt/openfoam9/etc/bashrc
export EIGEN_RHEO=$HOME/OpenFOAM/$USER-9/ThirdParty/Eigen3.2.9
export PETSC_DIR=$HOME/OpenFOAM/$USER-9/ThirdParty/petsc-3.16.5
export PETSC_ARCH=arch-linux-c-opt
export PATH=$HOME/OpenFOAM/$USER-9/platforms/linux64GccDPInt32Opt/bin:$PATH
export LD_LIBRARY_PATH=$HOME/OpenFOAM/$USER-9/platforms/linux64GccDPInt32Opt/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$HOME/OpenFOAM/$USER-9/platforms/linux64GccDPInt32Opt/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$PETSC_DIR/$PETSC_ARCH/lib:$LD_LIBRARY_PATH
```

* Start a singularity shell with the OpenFOAM image and setup the environment.

```bash
singularity exec of90.sif bash
Singularity> source ./setup.sh
```

* Test RheoTools

```bash
Singularity>  rheoFoam -help

Usage: rheoFoam [OPTIONS]
options:
  -case <dir>       specify alternate case directory, default is the cwd
  -fileHandler <handler>
                    override the fileHandler
  -hostRoots <(((host1 dir1) .. (hostN dirN))>
                    slave root directories (per host) for distributed running
  -libs <(lib1 .. libN)>
                    pre-load libraries
  -listFunctionObjects
                    List functionObjects
  -listFvConstraints
                    List fvConstraints
  -listFvModels     List fvModels
  -listScalarBCs    List scalar field boundary conditions (fvPatchField<scalar>)
  -listSwitches     List all available debug, info and optimisation switches
  -listVectorBCs    List vector field boundary conditions (fvPatchField<vector>)
  -noFunctionObjects
                    do not execute functionObjects
  -parallel         run in parallel
  -postProcess      Execute functionObjects only
  -roots <(dir1 .. dirN)>
                    slave root directories for distributed running
  -srcDoc           display source code in browser
  -doc              display application documentation in browser
  -help             print the usage

Using: OpenFOAM-9 (see https://openfoam.org)
Build: 9-b456138dc4bc

Singularity> 
```

### OpenFOAM Singularity definition file `of90.def`

```bash
Bootstrap: docker
From: ubuntu:20.04

%help
    This container provides OpenFOAM
    
%environment
    # Set the timezone
    export TZ=Etc/UTC
    
%post
    # Set non-interactive frontend for apt-get
    export DEBIAN_FRONTEND=noninteractive
    ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime
    apt-get update && apt-get install -y tzdata
    dpkg-reconfigure --frontend noninteractive tzdata
    
    echo "Update the apt package list and install necessary softwares"
    apt-get install -y \
        wget software-properties-common \
        build-essential \
        cmake \
        wget \
        git \
        flex \
        bison \
        zlib1g-dev \
        libboost-all-dev \
        libopenmpi-dev \
        openmpi-bin \
        gfortran \
        libgmp-dev \
        libmpfr-dev \
        libmpc-dev \
        python3 \
        python3-pip \
        python3-dev \
        curl \
        libatlas-base-dev \
        libblas-dev \
        liblapack-dev \
        
    echo "Install OpenFOAM v9 and ParaView"
    # Create installation directories
    sh -c "wget -O - https://dl.openfoam.org/gpg.key > /etc/apt/trusted.gpg.d/openfoam.asc"
    add-apt-repository http://dl.openfoam.org/ubuntu
    apt-get update
    apt-get install -y openfoam9
  
%runscript    
    exec /bin/bash $@
```

### References
* [OpenFOAM Official Website](https://www.openfoam.com/)
* [The OpenFOAM Foundation](https://openfoam.org/)
* [OpenFOAM User Guide](https://www.openfoam.com/documentation/user-guide)
* [OpenFOAM GitHub Repository](https://github.com/OpenFOAM/OpenFOAM-9)
* [RheoTool OpenFOAM Toolbox GiHub Repository](https://github.com/fppimenta/rheoTool)

