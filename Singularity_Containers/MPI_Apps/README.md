## Singularity & MPI Applications

The goal of these instructions is to help you run [Message Passing Interface (MPI)](https://en.wikipedia.org/wiki/Message_Passing_Interface) programs using [Singularity](https://sylabs.io/guides/3.5/user-guide/introduction.html) containers on the FAS RC cluster. The MPI standard is used to implement distributed parallel applications across compute nodes of a single HPC cluster, such as [Cannon](https://www.rc.fas.harvard.edu/about/cluster-architecture/), or across multiple compute systems. The two major open-source implementations of MPI are [Mpich](https://www.mpich.org/) (and its derivatives, such as [Mvapich](https://mvapich.cse.ohio-state.edu/)), and [OpenMPI](https://www.open-mpi.org/). The most widely used MPI implementation on Cannon is OpenMPI.

For information on using Singularity on the FASRC cluster, refer to [this](https://docs.rc.fas.harvard.edu/kb/singularity-on-the-cluster/) page.

There are several ways of developing and running MPI applications using Singularity containers, where the most popular method relies on the MPI implementation available on the host machine. This approach is named *Host MPI* or the *Hybrid* model since it uses both the MPI implementation on the host and the one in the container.

The key idea behind the *Hybrid* method is that when you execute a Singularity container with a MPI application, you call <code>mpiexec</code>, <code>mpirun</code>, or <code>srun</code>, e.g., when using the *SLURM* scheduler, on the <code>singularity</code> command itself. Then the MPI process outside of the container will work together with MPI inside the container to initialize the parallel job. **Therefore, it is very important that the MPI flavors and versions inside the container and on the host match.**

### Example MPI code

To illustrate how Singularity can be used with MPI applications, we will use a simple MPI code implemented in Fortran 90, <code>mpitest.f90</code>:

```fortran
!=====================================================
! Fortran 90 MPI example: mpitest.f90
!=====================================================
program mpitest
  implicit none
  include 'mpif.h'
  integer(4) :: ierr
  integer(4) :: iproc
  integer(4) :: nproc
  integer(4) :: i
  call MPI_INIT(ierr)
  call MPI_COMM_SIZE(MPI_COMM_WORLD,nproc,ierr)
  call MPI_COMM_RANK(MPI_COMM_WORLD,iproc,ierr)
  do i = 0, nproc-1
     call MPI_BARRIER(MPI_COMM_WORLD,ierr)
     if ( iproc == i ) then
        write (6,*) 'Rank',iproc,'out of',nproc
     end if
  end do
  call MPI_FINALIZE(ierr)
  if ( iproc == 0 ) write(6,*)'End of program.'
  stop
end program mpitest
```

### Singularity Definition File

To build Singularity images you need to write a [Definition File](https://sylabs.io/guides/3.8/user-guide/definition_files.html), where the the exact implementation will depend on the available MPI flavor on the host machine.

#### OpenMPI

If you intend to use OpenMPI, the definition file could look like, e.g., the one below:

```bash
Bootstrap: yum
OSVersion: 7
MirrorURL: http://mirror.centos.org/centos-%{OSVERSION}/%{OSVERSION}/os/$basearch/
Include: yum

%files
  mpitest.f90 /home/

%environment
  export OMPI_DIR=/opt/ompi
  export SINGULARITY_OMPI_DIR=$OMPI_DIR
  export SINGULARITYENV_APPEND_PATH=$OMPI_DIR/bin
  export SINGULAIRTYENV_APPEND_LD_LIBRARY_PATH=$OMPI_DIR/lib

%post
  yum -y install vim-minimal
  yum -y install gcc
  yum -y install gcc-gfortran
  yum -y install gcc-c++
  yum -y install which tar wget gzip bzip2
  yum -y install make
  yum -y install perl

  echo "Installing Open MPI ..."
  export OMPI_DIR=/opt/ompi
  export OMPI_VERSION=4.1.1
  export OMPI_URL="https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-$OMPI_VERSION.tar.bz2"
  mkdir -p /tmp/ompi
  mkdir -p /opt
  # --- Download ---
  cd /tmp/ompi
  wget -O openmpi-$OMPI_VERSION.tar.bz2 $OMPI_URL && tar -xjf openmpi-$OMPI_VERSION.tar.bz2
  # --- Compile and install ---
  cd /tmp/ompi/openmpi-$OMPI_VERSION
  ./configure --prefix=$OMPI_DIR && make -j4 && make install
  # --- Set environmental variables so we can compile our application ---
  export PATH=$OMPI_DIR/bin:$PATH
  export LD_LIBRARY_PATH=$OMPI_DIR/lib:$LD_LIBRARY_PATH
  export MANPATH=$OMPI_DIR/share/man:$MANPATH
  # --- Compile our application ---
  cd /home
  mpif90 -o mpitest.x mpitest.f90 -O2
```

#### Mpich

If you intend to use OpenMPI, the definition file could look like, e.g., the one below:

```bash
Bootstrap: yum
OSVersion: 7
MirrorURL: http://mirror.centos.org/centos-%{OSVERSION}/%{OSVERSION}/os/$basearch/
Include: yum

%files
  /n/home06/pkrastev/holyscratch01/Singularity/MPI/mpitest.f90 /home/

%environment
  export SINGULARITY_MPICH_DIR=/usr

%post
  yum -y install vim-minimal
  yum -y install gcc
  yum -y install gcc-gfortran
  yum -y install gcc-c++
  yum -y install which tar wget gzip
  yum -y install make
  cd /root/
  wget http://www.mpich.org/static/downloads/3.1.4/mpich-3.1.4.tar.gz
  tar xvfz mpich-3.1.4.tar.gz
  cd mpich-3.1.4/
  ./configure --prefix=/usr && make -j2 && make install
  cd /home
  mpif90 -o mpitest.x mpitest.f90 -O2
  cp mpitest.x /usr/bin/
```

### Building Singularity Images

To build Singularity containers, you need *root* access to the build system. Therefore, you cannot build a Singularity container on the FASRC cluster. Please, refer to the available options [here](https://docs.rc.fas.harvard.edu/kb/singularity-on-the-cluster/#Building_Singularity_images).

Provided you have root access to a Linux system, you can use the below commands to build your Singularity images, e.g.:

```bash
# --- Building the OpenMPI based image ---
$ sudo singularity build openmpi_test.simg openmpi_test.def
# --- Building the based Mpich image ---
$ sudo singularity build mpich_test.simg mpich_test.def
```

These will generate the Singularity image files <code>openmpi\_test.simg</code> and <code>mpich\_test.simg</code> respectively.

**Note:** The required definition files (.def) are included in this Github repository. The Singularity images themselves are not included as they are rather large.

### Executing MPI Applications with Singularity

On the FASRC cluster the standard way to execute MPI applications is through a batch-job submission script. Below are two examples, one using OpenMPI, and another Mpich.

#### OpenMPI

```bash
#!/bin/bash
#SBATCH -p test
#SBATCH -n 8
#SBATCH -J mpi_test
#SBATCH -o mpi_test.out
#SBATCH -e mpi_test.err
#SBATCH -t 30
#SBATCH --mem-per-cpu=1000

# --- Set up environment ---
export UCX_TLS=ib
export PMIX_MCA_gds=hash
module load gcc/10.2.0-fasrc01 
module load openmpi/4.1.1-fasrc01

# --- Run the MPI application in the container ---
srun -n 8 --mpi=pmix singularity exec openmpi_test.simg /home/mpitest.x
```
**Note:** Please notice that the version of the OpenMPI implementation used on the host need to match the one in the Singularity container. In this case this is version 4.1.1.

If the above script is named <code>run.sbatch.ompi</code>, the MPI Singularity job is submitted as usual with:

```bash
sbatch run.sbatch.ompi
``` 

#### Mpich

```bash
#!/bin/bash
#SBATCH -p test
#SBATCH -n 8
#SBATCH -J mpi_test
#SBATCH -o mpi_test.out
#SBATCH -e mpi_test.err
#SBATCH -t 30
#SBATCH --mem-per-cpu=1000

# --- Set up environment ---
module load python/3.8.5-fasrc01
source activate python3_env1

# --- Run the MPI application in the container ---
srun -n 8 --mpi=pmi2 singularity exec mpich_test.simg /usr/bin/mpitest.x
```

If the above script is named <code>run.sbatch.mpich</code>, the MPI Singularity job is submitted as usual with:

```bash
$ sbatch run.sbatch.mpich
``` 

**Note:** Please notice that we don't have Mpich installed as a software module on the FASRC cluster and therefore this example assumes that Mpich is installed in your user, or lab, environment. The easiest way to do this is through a [conda](https://docs.conda.io/en/latest/) environment. You can find more information on how to set up conda environments in our computing environment [here](https://docs.rc.fas.harvard.edu/kb/python/).

Provided you have set up and activated a conda environment named, e.g., <code>python3\_env1</code>, Mpich version 3.1.4 can be installed with:

```bash
$ conda install mpich==3.1.4
```

### Example output

```bash
$ cat mpi_test.out
 Rank           0 out of           8
 Rank           1 out of           8
 Rank           2 out of           8
 Rank           3 out of           8
 Rank           4 out of           8
 Rank           5 out of           8
 Rank           6 out of           8
 Rank           7 out of           8
 End of program.
```

### Compiling Code with OpenMPI Inside Singularity Container

To compile inside the Singularity container, we need to request a compute node to run Singularity:

```bash
$ salloc -p test --time=0:30:00 --mem=1000 -n 1
```

Using the file `compile_openmpi.sh`, you can compile `mpitest.f90` by executing `bash compile_openmpi.sh` inside the container `openmpi_test.simg` 

```bash
$ cat compile_openmpi.sh
#!/bin/bash

export PATH=$OMPI_DIR/bin:$PATH
export LD_LIBRARY_PATH=$OMPI_DIR/lib:$LD_LIBRARY_PATH

# compile fortran program
mpif90 -o mpitest.x mpitest.f90 -O2

# compile c program
mpicc -o mpitest.exe mpitest.c

$ singularity exec openmpi_test.simg bash compile_openmpi.sh
```
In `compile_openmpi.sh`, we also included the compilation command for a [c program](https://sylabs.io/guides/3.8/user-guide/mpi.html#test-application).
