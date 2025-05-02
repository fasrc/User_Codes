# KHARMA

KHARMA (Kokkos-based High-order Adaptive Relativistic Magnetohydrodynamics Application) is a state-of-the-art C++ code developed for simulating general relativistic magnetohydrodynamics (GRMHD) in astrophysical environments. Designed for high performance on modern supercomputing architectures, KHARMA leverages the Kokkos library to achieve portability across CPUs and GPUs. It incorporates advanced numerical schemes, adaptive mesh refinement (AMR), and flexible coordinate systems to model highly dynamic, magnetized plasma flows near compact objects such as black holes and neutron stars. KHARMA is widely used in cutting-edge research on accretion disks, jets, and other relativistic phenomena in strong gravitational fields.

## Installing KHARMA on the FASRC Cannon cluster

Below we provide instructions for setting up and running KHARMA on Cannon. We will look at two different ways to set up the code - (i) compiling KHARMA directly, and (ii) installing the code in a Singularity container.

---

### Compiling KHARMA directly

**1. Load Required Modules**

Load the necessary compiler, MPI, HDF5, and Cmake modules:

```bash
module load gcc/12.2.0-fasrc01 openmpi/5.0.5-fasrc02 hdf5/1.14.6-fasrc01 cmake/3.31.6-fasrc01
```

**2. Clone the KHARMA Repository**

```bash
# Create space for the project, e.g.,
mkdir -p KHARMA/
cd KHARMA/

# Clone the code from GitHub
git clone https://github.com/achael/kharma.git
cd kharma/

# Check out dependencies and update submodules
git submodule update --init --recursive
```

**3. Update Machine Configuration**

KHARMA uses machine-specific configuration scripts located in `kharma/machines/`.

- Move or remove the old `cannon_*.sh` if it exists:

```bash
mv machines/cannon_*.sh machines/cannon.sh.bak
```

- Create a new `cannon.sh` file:

```bash
vi machines/cannon.sh
```

- Paste the following into `machines/cannon.sh`, and save the file:

```bash
# Harvard Cannon

if [[ $HOST == *"rc.fas.harvard.edu" ]]; then
    echo CANNON
    HOST_ARCH=HSW
    NPROC=16
    DEVICE_ARCH=AMPERE80
    EXTRA_FLAGS="-DPARTHENON_DISABLE_HDF5_COMPRESSION=ON"

    if [[ "$ARGS" == *"cuda"* ]]; then
        module load gcc/12.2.0-fasrc01
        module load openmpi/5.0.5-fasrc02
        module load hdf5/1.14.6-fasrc01
        module load cmake/3.31.6-fasrc01
        C_NATIVE=gcc
        CXX_NATIVE=g++
    fi
fi
```

> **Note**: Adjust `DEVICE_ARCH` to `VOLTA70` if you are targeting V100 GPUs instead of A100.

**4. Compile KHARMA**

Use KHARMA’s standard make system to clean and build for CUDA:

```bash
./make.sh clean cuda
```

Upon successful compilation, the executable `kharma.cuda` will be located in the root directory `KHARMA/kharma/`.

---

### Building a Singularity container

Use the provided Singularity definition file `kharma_v2.def` to build a Singularity container.

```bash
Bootstrap: docker
From: nvidia/cuda:12.2.0-devel-ubuntu22.04

%labels
    Maintainer Plamen G. Krastev, FASRC
    Version kharma-openmpi-hdf5

%environment
    export PATH=/opt/openmpi/bin:/opt/hdf5/bin:/opt/kharma:$PATH
    export LD_LIBRARY_PATH=/opt/openmpi/lib:/opt/hdf5/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    export OMPI_ALLOW_RUN_AS_ROOT=1
    export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

%post
    echo ">>> Installing base packages"
    apt-get update && apt-get install -y \
        build-essential cmake git wget curl nano \
        libfftw3-dev libpnetcdf-dev \
        libnuma-dev pciutils m4 autoconf automake libtool \
        python3 python3-pip \
        && apt-get clean

    echo ">>> Building UCX from source"
    cd /tmp
    rm -rf ucx
    apt-get install -y librdmacm-dev libibverbs-dev numactl libnuma-dev
    git clone https://github.com/openucx/ucx.git
    cd ucx
    git checkout v1.16.0
    ./autogen.sh
    ./configure --prefix=/opt/ucx \
        --with-cuda=/usr/local/cuda \
        --with-rdmacm --with-verbs \
        --enable-mt
    make -j$(nproc) && make install

    export UCX_PATH=/opt/ucx
    export PATH=${UCX_PATH}/bin:$PATH
    export LD_LIBRARY_PATH=${UCX_PATH}/lib:$LD_LIBRARY_PATH

    echo ">>> Building OpenMPI 5.0.5 with CUDA and UCX"
    cd /tmp
    rm -rf openmpi-5.0.5 openmpi-5.0.5.tar.gz
    apt-get update && apt-get install -y build-essential libnuma-dev
    wget https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.5.tar.gz
    tar -xzf openmpi-5.0.5.tar.gz
    cd openmpi-5.0.5
    ./configure --prefix=/opt/openmpi \
                --with-cuda=/usr/local/cuda \
                --with-ucx=/opt/ucx
    make -j$(nproc) && make install

    export PATH=/opt/openmpi/bin:$PATH
    export LD_LIBRARY_PATH=/opt/openmpi/lib:$LD_LIBRARY_PATH

    echo ">>> Building parallel HDF5 1.14.6 with OpenMPI"
    cd /tmp
    rm -rf hdf5-1.14.6 hdf5-1.14.6.tar.gz
    wget https://support.hdfgroup.org/releases/hdf5/v1_14/v1_14_6/downloads/hdf5-1.14.6.tar.gz
    tar -xzf hdf5-1.14.6.tar.gz
    cd hdf5-1.14.6
    ./configure --prefix=/opt/hdf5 --enable-parallel --enable-shared \
                CC=/opt/openmpi/bin/mpicc
    make -j$(nproc) && make install

    export PATH=/opt/hdf5/bin:$PATH
    export LD_LIBRARY_PATH=/opt/hdf5/lib:$LD_LIBRARY_PATH

    echo ">>> Cloning KHARMA"
    cd /opt
    git clone https://github.com/AFD-Illinois/kharma.git
    cd kharma
    git submodule update --init --recursive

    echo ">>> Updating machine files"
    mv machines/cannon_ramesh.sh /tmp/cannon_ramesh.sh

    cat << 'EOF' > machines/cannon.sh
# Harvard Cannon

if [[ $HOST == *"rc.fas.harvard.edu" ]]; then
    echo CANNON
    HOST_ARCH=HSW
    NPROC=16
    DEVICE_ARCH=AMPERE80  # For V100 nodes (change if needed)
    EXTRA_FLAGS="-DPARTHENON_DISABLE_HDF5_COMPRESSION=ON -DKokkos_ARCH_AMPERE80=ON"

    if [[ "$ARGS" == *"cuda"* ]]; then
        #module load gcc/12.2.0-fasrc01 
        #module load openmpi/5.0.5-fasrc02
        #module load hdf5/1.14.6-fasrc01
        #module load cmake/3.31.6-fasrc01
        C_NATIVE=gcc
        CXX_NATIVE=g++ 
    fi
fi
EOF

    echo ">>> Building KHARMA with make.sh"
    ./make.sh clean cuda

%runscript
    exec /bin/bash "$@"

%test
    echo "KHARMA with OpenMPI, HDF5, CUDA is ready!"
```

Build the container with:

```bash
singularity build --fakeroot kharma_v2.sif kharma_v2.def
```

This will generate the Singularity image `kharma_v2.sif`

## Running KHARMA on the FASRC Cannon cluster

Here we provide example batch-job submission scripts to run KHARMA on the Cannon cluster.

### Running KHARMA using the compiled executable `kharma.cuda`

- 1 node / 4 GPUs (4 GPUs per Node)

```bash
#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=8G
#SBATCH -J mad_gpu4
#SBATCH -t 1:00:00
#SBATCH -p gpu
#SBATCH -o ./out4_test
#SBATCH -e ./err4_test

module load gcc/12.2.0-fasrc01 
module load openmpi/5.0.5-fasrc02
module load hdf5/1.14.6-fasrc01

export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export KOKKOS_NUM_DEVICES=4

srun -n 4 --mpi=pmix ../KHARMA/kharma/kharma.cuda -i ./par/mad.par4
```

- 2 Nodes / 8 GPUs (4 GPUs per Node)

```bash
#!/bin/bash
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=8G
#SBATCH -J mad_gpu4
#SBATCH -t 1:00:00
#SBATCH -p gpu
#SBATCH -o ./out4_test
#SBATCH -e ./err4_test

module load gcc/12.2.0-fasrc01 
module load openmpi/5.0.5-fasrc02
module load hdf5/1.14.6-fasrc01

export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export KOKKOS_NUM_DEVICES=4

srun -n 8 --mpi=pmix ../KHARMA/kharma/kharma.cuda -i ./par/mad.par4
```

### Running KHARMA using Singularity

- 8 Nodes / 8 GPUs (1 GPU per Node) 

```bash
#!/bin/bash
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=8G
#SBATCH -J mad_gpu4
#SBATCH -t 1:00:00
#SBATCH -p gpu
#SBATCH -o ./out8_singularity_test
#SBATCH -e ./err8_singularity_test

module load gcc/12.2.0-fasrc01 
module load openmpi/5.0.5-fasrc02
module load hdf5/1.14.6-fasrc01

export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export KOKKOS_NUM_DEVICES=1

srun -n 8 --mpi=pmix singularity exec --nv ../KHARMA/Image/kharma_v2.sif /opt/kharma/kharma.cuda  -i ./par/mad.par4
```
>**Note:** It is important to notice that you need to a MPI module matching the MPI flavor and version inside the container. This is required for running MPI apps with Singularity.

The above job-submission scripts assume that you have a KHARMA parameter file in the working directory, e.g.,

```
./par/mad.par4
```

## Comments

- `./make.sh` automatically picks up the correct flags based on your host and machine file.
- `VOLTA70` targets V100 GPUs available on Cannon.
- `AMPERE80` targets A100 GPUs available on Cannon.
- `NPROC=16` sets the number of compile threads.
- The OpenMPI module `openmpi/5.0.5-fasrc02` is CUDA-aware, built against `cuda/12.2.0-fasrc01`.

