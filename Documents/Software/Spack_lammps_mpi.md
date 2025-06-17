# Build LAMMPS MPI version using Spack

<img src="Images/spack-logo.svg" alt="spack-logo" width="100"/>

[LAMMPS](https://www.lammps.org/#gsc.tab=0) is a classical molecular dynamics code with a focus on materials modeling. It's an acronym for Large-scale Atomic/Molecular Massively Parallel Simulator.

The below instructions provide a spack recipe for building MPI capable instance of LAMMPS on the [FASRC Cannon cluster](https://www.rc.fas.harvard.edu/about/cluster-architecture/).

## Pre-requisites: Compiler and MPI Library Spack configuration

Here we will use the [GNU/GCC](https://gcc.gnu.org/) compiler suite together with [OpenMPI](https://www.open-mpi.org/). We will also use a module for [FFTW](https://www.fftw.org/).

The below instructions assume that spack is already configured to use the GCC compiler `gcc@14.2.0`, OpenMPI Library `openmpi@5.0.5`, and FFTW with module `fftw/3.3.10-fasrc01`. If you have not configured them yet, see:

1. To add `gcc` compiler: [Spack compiler configuration](Spack.md#compiler-configuration)
2. To add `openmpi`: [Spack MPI Configuration](Spack.md#mpi-configuration)
3. To add `fftw` as an external package:  [Using an Lmod module in Spack](Spack.md#using-an-lmod-module-in-spack)

## Create LAMMPS spack environment and activate it

First, request an interactive job

```
salloc --partition test --time 06:00:00 --mem-per-cpu 4G -c 8
```

Second, download and activate Spack. For performance, we recommend using a Lab share in Holyoke (i.e., path starts with `holy`) instead of using your home directory. Here, we show an example with `/n/holylabs`:

```
cd /n/holylabs/jharvard_lab/Lab/jharvard
git clone -c feature.manyFiles=true https://github.com/spack/spack.git spack_lammps
cd spack_lammps/
source share/spack/setup-env.sh
```

Finally, create a Spack environment and activate it

```bash
spack env create lammps
spack env activate -p lammps
```

## Install the LAMMPS environment

### Note on architecture

If you are planning to run LAMMPS in different partitions, we recommend setting Spack to a [general architecture](spack.md#default-architecture). Otherwise, Spack will detect the architecture of the node that you are building LAMMPS and optimize for this specific architecture and may not run on another hardware. For example, LAMMPS built on Sapphire Rapids may not run on Cascade Lake.

### Install `libbsd`

> **Note:** In this recipe, we first install `libbsd` with the system version of the GCC compiler, `gcc@8.5.0`, as the installation fails, if we try to add it directly to the environment and install it with `gcc@12.2.0`.

```bash
spack install --add libbsd@0.12.2 % gcc@8.5.0
```

### Add the rest of the required packages to the spack environment

First, add `Python<=3.10` because newer versions of Python do not contain the package `distutils` ([reference](https://stackoverflow.com/a/76691103)) and will cause the installation to fail.

```
spack add python@3.10
```
Second, add openmpi

```
spack add openmpi@5.0.5
```

Third, add FFTW

```
spack add fftw@3.3.10
```

Then, add LAMMPS required packages

```bash
spack add lammps +asphere +body +class2 +colloid +compress +coreshell +dipole +granular +kokkos +kspace +manybody +mc +misc +molecule +mpiio +openmp-package +peri +python +qeq +replica +rigid +shock +snap +spin +srd +user-reaxc +user-misc % gcc@14.2.0 ^ openmpi@5.0.5
```

### Install the environment

Once all required packages are added to the environment, it can be installed with (note that the installation can take 1-2 hours):

```bash
spack install
```

For example,

```bash
[lammps] [jharvard@holy8a24101 spack_lammps]$ spack add python@3.10
==> Adding python@3.10 to environment lammps

[lammps] [jharvard@holy8a24101 spack_lammps]$ spack add openmpi@5.0.5
==> Adding openmpi@5.0.5 to environment lammps

[lammps] [jharvard@holy8a24101 spack_lammps]$ spack add fftw@3.3.10
==> Adding fftw@3.3.10 to environment lammps

[lammps] [jharvard@holy8a24101 spack_lammps]$ spack add lammps +asphere +body +class2 +colloid +compress +coreshell +dipole +granular +kokkos +kspace +manybody +mc +misc +molecule +mpiio +openmp-package +peri +python +qeq +replica +rigid +shock +snap +spin +srd +user-reaxc +user-misc % gcc@14.2.0 ^ openmpi@5.0.5
==> Adding lammps+asphere+body+class2+colloid+compress+coreshell+dipole+granular+kokkos+kspace+manybody+mc+misc+molecule+mpiio+openmp-package+peri+python+qeq+replica+rigid+shock+snap+spin+srd+user-misc+user-reaxc %gcc@14.2.0 ^openmpi@5.0.5 to environment lammps

[lammps] [jharvard@holy8a24101 spack_lammps]$ time spack install
==> Concretized 4 specs
[e]  3h3h2fa  fftw@3.3.10+mpi~openmp~pfft_patches+shared build_system=autotools patches:=872cff9 precision:=double,float arch=linux-rocky8-x86_64
 -   mpxtdzi  lammps@20201029+asphere+body+class2+colloid+compress+coreshell~cuda~cuda_mps+dipole~exceptions~ffmpeg+granular~ipo~jpeg~kim+kokkos+kspace~latte+lib+manybody+mc+misc~mliap+molecule+mpi+mpiio~opencl+openmp+openmp-package~opt+peri~png~poems+python+qeq+replica+rigid~rocm+shock+snap+spin+srd~tools~user-adios~user-atc~user-awpmd~user-bocs~user-cgsdk~user-colvars~user-diffraction~user-dpd~user-drude~user-eff~user-fep~user-h5md~user-intel~user-lb~user-manifold~user-meamc~user-mesodpd~user-mesont~user-mgpt+user-misc~user-mofff~user-molfile~user-netcdf~user-omp~user-phonon~user-plumed~user-ptm~user-qtb~user-reaction+user-reaxc~user-sdpd~user-smd~user-smtbq~user-sph~user-tally~user-uef~user-yaff~voronoi build_system=cmake build_type=Release fft=fftw3 fftw_precision=double generator=make lammps_sizes=smallbig arch=linux-rocky8-x86_64
 -   4rflzcx      ^cmake@3.31.6~doc+ncurses+ownlibs~qtgui build_system=generic build_type=Release arch=linux-rocky8-x86_64
 -   7ntqqo3          ^curl@8.11.1~gssapi~ldap~libidn2~librtmp~libssh~libssh2+nghttp2 build_system=autotools libs:=shared,static tls:=openssl arch=linux-rocky8-x86_64
 -   7naninn              ^nghttp2@1.65.0 build_system=autotools arch=linux-rocky8-x86_64
[+]  4jidax3      ^compiler-wrapper@1.0 build_system=generic arch=linux-rocky8-x86_64
[e]  ttcg57n      ^gcc@14.2.0~binutils+bootstrap~graphite~mold~nvptx~piclibs~profiled~strip build_system=autotools build_type=RelWithDebInfo languages:='c,c++,fortran' arch=linux-rocky8-x86_64
 -   xfiurn5      ^gcc-runtime@14.2.0 build_system=generic arch=linux-rocky8-x86_64
[e]  o6n6gob      ^glibc@2.28 build_system=autotools arch=linux-rocky8-x86_64
[+]  bbt3qol      ^gmake@4.4.1~guile build_system=generic arch=linux-rocky8-x86_64
[e]  g32o7e4          ^gcc@8.5.0~binutils+bootstrap~graphite~nvptx~piclibs~profiled~strip build_system=autotools build_type=RelWithDebInfo languages:='c,c++,fortran' patches:=98a9c96,d4919d6 arch=linux-rocky8-x86_64
[+]  ilgbgax          ^gcc-runtime@8.5.0 build_system=generic arch=linux-rocky8-x86_64
 -   osze522      ^kokkos@3.7.02~aggressive_vectorization~cmake_lang~compiler_warnings+complex_align~cuda~debug~debug_bounds_check~debug_dualview_modify_check~deprecated_code~examples~hip_relocatable_device_code~hpx~hpx_async_dispatch~hwloc~ipo~memkind~numactl~openmp~openmptarget~pic~rocm+serial+shared~sycl~tests~threads~tuning~wrapper build_system=cmake build_type=Release cxxstd=17 generator=make intel_gpu_arch=none arch=linux-rocky8-x86_64
 -   nffr23q      ^py-build@1.2.2~virtualenv build_system=python_pip arch=linux-rocky8-x86_64
 -   zvk3yap          ^py-flit-core@3.12.0 build_system=python_pip arch=linux-rocky8-x86_64
 -   qmwbcrl          ^py-packaging@25.0 build_system=python_pip arch=linux-rocky8-x86_64
 -   hzb2z4c          ^py-pyproject-hooks@1.2.0 build_system=python_pip arch=linux-rocky8-x86_64
 -   rvtjcyg          ^py-tomli@2.0.1 build_system=python_pip arch=linux-rocky8-x86_64
 -   fcmf63t      ^py-mpi4py@4.0.1 build_system=python_pip arch=linux-rocky8-x86_64
 -   3lhqnpu          ^py-cython@3.0.11 build_system=python_pip arch=linux-rocky8-x86_64
 -   xrauvfb          ^py-setuptools@79.0.1 build_system=generic arch=linux-rocky8-x86_64
 -   uph3xco      ^py-numpy@2.2.6 build_system=python_pip patches:=1c9cb08,873745d arch=linux-rocky8-x86_64
 -   oldpzfb          ^openblas@0.3.29~bignuma~consistent_fpcsr+dynamic_dispatch+fortran~ilp64+locking+pic+shared build_system=makefile symbol_suffix=none threads=none arch=linux-rocky8-x86_64
 -   4na77sl          ^py-meson-python@0.16.0 build_system=python_pip arch=linux-rocky8-x86_64
 -   phdbgoa              ^meson@1.7.0 build_system=python_pip patches:=0f0b1bd arch=linux-rocky8-x86_64
 -   6antgmb                  ^ninja@1.12.1+re2c build_system=generic patches:=93f4bb3 arch=linux-rocky8-x86_64
 -   bbhlvsn                      ^re2c@3.1 build_system=autotools arch=linux-rocky8-x86_64
 -   6dqk3y7              ^py-pyproject-metadata@0.9.1 build_system=python_pip arch=linux-rocky8-x86_64
 -   aawwbc3      ^py-pip@25.1.1 build_system=generic arch=linux-rocky8-x86_64
 -   n3ekb2x      ^py-wheel@0.45.1 build_system=generic arch=linux-rocky8-x86_64
 -   ktvjhpk      ^python-venv@1.0 build_system=generic arch=linux-rocky8-x86_64
[e]  guivmvv  openmpi@5.0.5+atomics~cuda~debug~gpfs~internal-hwloc~internal-libevent~internal-pmix~ipv6~java~lustre~memchecker~openshmem~romio+rsh~static~two_level_namespace+vt+wrapper-rpath build_system=autotools fabrics:=none romio-filesystem:=none schedulers:=none arch=linux-rocky8-x86_64
 -   e6cslwo  python@3.10.16+bz2+crypt+ctypes+dbm~debug+libxml2+lzma~optimizations+pic+pyexpat+pythoncmd+readline+shared+sqlite3+ssl~tkinter+uuid+zlib build_system=generic patches:=0d98e93,7d40923,ebdca64,f2fd060 arch=linux-rocky8-x86_64
 -   aw2b7q4      ^bzip2@1.0.8~debug~pic+shared build_system=generic arch=linux-rocky8-x86_64
 -   e6p5vwc          ^diffutils@3.10 build_system=autotools arch=linux-rocky8-x86_64
 -   4kuimjy      ^expat@2.7.1+libbsd build_system=autotools arch=linux-rocky8-x86_64
[+]  aonj3db          ^libbsd@0.12.2 build_system=autotools arch=linux-rocky8-x86_64
[+]  nwfft3p              ^libmd@1.1.0 build_system=autotools arch=linux-rocky8-x86_64
 -   tuio76x      ^gdbm@1.23 build_system=autotools arch=linux-rocky8-x86_64
 -   wpkusmx      ^gettext@0.23.1+bzip2+curses+git~libunistring+libxml2+pic+shared+tar+xz build_system=autotools arch=linux-rocky8-x86_64
 -   rbqgqpv          ^libiconv@1.18 build_system=autotools libs:=shared,static arch=linux-rocky8-x86_64
 -   4tegpgq          ^libxml2@2.13.5~http+pic~python+shared build_system=autotools arch=linux-rocky8-x86_64
 -   kzgcnyy          ^tar@1.35 build_system=autotools zip=pigz arch=linux-rocky8-x86_64
 -   6xso4pk              ^pigz@2.8 build_system=makefile arch=linux-rocky8-x86_64
 -   6pbg4je              ^zstd@1.5.7+programs build_system=makefile compression:=none libs:=shared,static arch=linux-rocky8-x86_64
 -   3vj4gpx      ^libffi@3.4.7 build_system=autotools arch=linux-rocky8-x86_64
 -   we4xltk      ^libxcrypt@4.4.38~obsolete_api build_system=autotools arch=linux-rocky8-x86_64
 -   hctt263          ^perl@5.40.0+cpanm+opcode+open+shared+threads build_system=generic arch=linux-rocky8-x86_64
 -   nwmbelg              ^berkeley-db@18.1.40+cxx~docs+stl build_system=autotools patches:=26090f4,b231fcc arch=linux-rocky8-x86_64
 -   xybm2qm      ^ncurses@6.5~symlinks+termlib abi=none build_system=autotools patches:=7a351bc arch=linux-rocky8-x86_64
 -   o6gbeku      ^openssl@3.4.1~docs+shared build_system=generic certs=mozilla arch=linux-rocky8-x86_64
 -   feo5ibm          ^ca-certificates-mozilla@2025-02-25 build_system=generic arch=linux-rocky8-x86_64
 -   z365t2j      ^pkgconf@2.3.0 build_system=autotools arch=linux-rocky8-x86_64
 -   o2rpuwe      ^readline@8.2 build_system=autotools patches:=1ea4349,24f587b,3d9885e,5911a5b,622ba38,6c8adf8,758e2ec,79572ee,a177edc,bbf97f1,c7b45ff,e0013d9,e065038 arch=linux-rocky8-x86_64
 -   emd77vc      ^sqlite@3.46.0+column_metadata+dynamic_extensions+fts~functions+rtree build_system=autotools arch=linux-rocky8-x86_64
 -   vlbruks      ^util-linux-uuid@2.41 build_system=autotools arch=linux-rocky8-x86_64
 -   x3xcxj3      ^xz@5.6.3~pic build_system=autotools libs:=shared,static arch=linux-rocky8-x86_64
 -   yprpmnc      ^zlib-ng@2.2.4+compat+new_strategies+opt+pic+shared build_system=autotools arch=linux-rocky8-x86_64

[+] /n/sw/helmod-rocky8/apps/Comp/gcc/14.2.0-fasrc01/openmpi/5.0.5-fasrc01 (external openmpi-5.0.5-guivmvvfvgx5ztklxiu5ahrhih7dp65j)
[+] /n/sw/helmod-rocky8/apps/MPI/gcc/14.2.0-fasrc01/openmpi/5.0.5-fasrc01/fftw/3.3.10-fasrc01 (external fftw-3.3.10-3h3h2falkeomenwugyxqgvk5u6cx3rh3)
[+] /n/holylabs/jharvard_lab/Lab/software/spack_lammps/opt/spack/linux-x86_64/compiler-wrapper-1.0-4jidax3a3jc3ogpk7a5x5gufb2cxs4gr
==> gcc@14.2.0 : has external module in ['gcc/14.2.0-fasrc01']
[+] /n/sw/helmod-rocky8/apps/Core/gcc/14.2.0-fasrc01 (external gcc-14.2.0-ttcg57nq5t3r5a4aaslllnnypcwscff5)
[+] /usr (external glibc-2.28-o6n6gob4a7744vkxw5jqpioi55tdj633)
==> Installing ca-certificates-mozilla-2025-02-25-feo5ibmyevhdlh75lb4wcn3pqq72gtgi [6/57]
==> No binary for ca-certificates-mozilla-2025-02-25-feo5ibmyevhdlh75lb4wcn3pqq72gtgi found: installing from source
==> Fetching https://mirror.spack.io/_source-cache/archive/50/50a6277ec69113f00c5fd45f09e8b97a4b3e32daa35d3a95ab30137a55386cef
    [100%]  233.26 KB @   22.7 MB/s
==> No patches needed for ca-certificates-mozilla
==> ca-certificates-mozilla: Executing phase: 'install'
==> ca-certificates-mozilla: Successfully installed ca-certificates-mozilla-2025-02-25-feo5ibmyevhdlh75lb4wcn3pqq72gtgi
  Stage: 0.08s.  Install: 0.01s.  Post-install: 0.06s.  Total: 0.20s
[+] /n/holylabs/jharvard_lab/Lab/software/spack_lammps/opt/spack/linux-x86_64/ca-certificates-mozilla-2025-02-25-feo5ibmyevhdlh75lb4wcn3pqq72gtgi
[+] /n/holylabs/jharvard_lab/Lab/software/spack_lammps/opt/spack/linux-x86_64/gcc-runtime-8.5.0-ilgbgaxvhaiblyhnxila5hwescpoucab
==> Installing gcc-runtime-14.2.0-xfiurn5qvdozoj7vdjsnff2gh3fugkr7 [8/57]
==> No binary for gcc-runtime-14.2.0-xfiurn5qvdozoj7vdjsnff2gh3fugkr7 found: installing from source
==> No patches needed for gcc-runtime
==> gcc-runtime: Executing phase: 'install'
==> gcc-runtime: Successfully installed gcc-runtime-14.2.0-xfiurn5qvdozoj7vdjsnff2gh3fugkr7
  Stage: 0.00s.  Install: 1.38s.  Post-install: 0.15s.  Total: 1.59s
[+] /n/holylabs/jharvard_lab/Lab/software/spack_lammps/opt/spack/linux-x86_64/gcc-runtime-14.2.0-xfiurn5qvdozoj7vdjsnff2gh3fugkr7
[+] /n/holylabs/jharvard_lab/Lab/software/spack_lammps/opt/spack/linux-x86_64/libmd-1.1.0-nwfft3pazs5ut2ybvxpnuy7s3wei3twc
[+] /n/holylabs/jharvard_lab/Lab/software/spack_lammps/opt/spack/linux-x86_64/gmake-4.4.1-bbt3qoltmvddzvkp7ntwxlcwvowdw77b
==> Installing zstd-1.5.7-6pbg4jenkueraxcdqpgqn44bj2p6mylh [11/57]
==> No binary for zstd-1.5.7-6pbg4jenkueraxcdqpgqn44bj2p6mylh found: installing from source
==> Fetching https://mirror.spack.io/_source-cache/archive/37/37d7284556b20954e56e1ca85b80226768902e2edabd3b649e9e72c0c9012ee3.tar.gz
    [100%]    2.45 MB @   45.3 MB/s
==> No patches needed for zstd
==> zstd: Executing phase: 'edit'
==> zstd: Executing phase: 'build'
==> zstd: Executing phase: 'install'
==> zstd: Successfully installed zstd-1.5.7-6pbg4jenkueraxcdqpgqn44bj2p6mylh
  Stage: 0.22s.  Edit: 0.00s.  Build: 0.00s.  Install: 29.18s.  Post-install: 0.15s.  Total: 29.65s
[+] /n/holylabs/jharvard_lab/Lab/software/spack_lammps/opt/spack/linux-x86_64/zstd-1.5.7-6pbg4jenkueraxcdqpgqn44bj2p6mylh
==> Installing libffi-3.4.7-3vj4gpxomsn32hwgfpvftdqfrbe3dkrt [12/57]
==> No binary for libffi-3.4.7-3vj4gpxomsn32hwgfpvftdqfrbe3dkrt found: installing from source
==> Fetching https://mirror.spack.io/_source-cache/archive/13/138607dee268bdecf374adf9144c00e839e38541f75f24a1fcf18b78fda48b2d.tar.gz
    [100%]    1.39 MB @   54.6 MB/s
==> No patches needed for libffi
==> libffi: Executing phase: 'autoreconf'
==> libffi: Executing phase: 'configure'
==> libffi: Executing phase: 'build'
==> libffi: Executing phase: 'install'
==> libffi: Successfully installed libffi-3.4.7-3vj4gpxomsn32hwgfpvftdqfrbe3dkrt
  Stage: 0.14s.  Autoreconf: 0.00s.  Configure: 4.95s.  Build: 1.53s.  Install: 0.40s.  Post-install: 0.11s.  Total: 7.28s

... omitted output ...

[+] /n/holylabs/jharvard_lab/Lab/software/spack_lammps/opt/spack/linux-x86_64/py-numpy-2.2.6-uph3xcohv3d6hy5oo7gimunjan4cgrsx
==> Installing lammps-20201029-mpxtdzikg462l7exn4m4igeqs2ymrfkg [57/57]
==> No binary for lammps-20201029-mpxtdzikg462l7exn4m4igeqs2ymrfkg found: installing from source
==> Fetching https://mirror.spack.io/_source-cache/archive/75/759705e16c1fedd6aa6e07d028cc0c78d73c76b76736668420946a74050c3726.tar.gz
    [100%]  127.07 MB @   63.1 MB/s
==> No patches needed for lammps
==> lammps: Executing phase: 'cmake'
==> lammps: Executing phase: 'build'
==> lammps: Executing phase: 'install'
==> lammps: Successfully installed lammps-20201029-mpxtdzikg462l7exn4m4igeqs2ymrfkg
  Stage: 4.38s.  Cmake: 5.00s.  Build: 5m 56.16s.  Install: 4.94s.  Post-install: 0.74s.  Total: 6m 11.57s
[+] /n/holylabs/jharvard_lab/Lab/software/spack_lammps/opt/spack/linux-x86_64/lammps-20201029-mpxtdzikg462l7exn4m4igeqs2ymrfkg
==> Updating view at /n/holylabs/jharvard_lab/Lab/software/spack_lammps/var/spack/environments/lammps/.spack-env/view
```

### List the packages installed in the LAMMPS environment

```bash
[lammps] [jharvard@holy8a24402 spack_lammps]$ spack find
==> In environment lammps
==> 5 root specs
[e] fftw@3.3.10
[+] lammps+asphere+body+class2+colloid+compress+coreshell+dipole+granular+kokkos+kspace+manybody+mc+misc+molecule+mpiio+openmp-package+peri+python+qeq+replica+rigid+shock+snap+spin+srd+user-misc+user-reaxc
[+] libbsd@0.12.2
[e] openmpi@5.0.5
[+] python@3.10

-- linux-rocky8-x86_64 / gcc@8.5.0 ------------------------------
gmake@4.4.1  libbsd@0.12.2  libmd@1.1.0

-- linux-rocky8-x86_64 / gcc@14.2.0 -----------------------------
berkeley-db@18.1.40  expat@2.7.1      libffi@3.4.7      nghttp2@1.65.0   pigz@2.8                py-numpy@2.2.6  tar@1.35
bzip2@1.0.8          gdbm@1.23        libiconv@1.18     ninja@1.12.1     pkgconf@2.3.0           python@3.10.16  util-linux-uuid@2.41
cmake@3.31.6         gettext@0.23.1   libxcrypt@4.4.38  openblas@0.3.29  py-cython@3.0.11        re2c@3.1        xz@5.6.3
curl@8.11.1          kokkos@3.7.02    libxml2@2.13.5    openssl@3.4.1    py-meson-python@0.16.0  readline@8.2    zlib-ng@2.2.4
diffutils@3.10       lammps@20201029  ncurses@6.5       perl@5.40.0      py-mpi4py@4.0.1         sqlite@3.46.0   zstd@1.5.7

-- linux-rocky8-x86_64 / no compiler ----------------------------
ca-certificates-mozilla@2025-02-25  gcc@14.2.0          meson@1.7.0          py-packaging@25.0            py-setuptools@79.0.1
compiler-wrapper@1.0                gcc-runtime@8.5.0   openmpi@5.0.5        py-pip@25.1.1                py-tomli@2.0.1
fftw@3.3.10                         gcc-runtime@14.2.0  py-build@1.2.2       py-pyproject-hooks@1.2.0     py-wheel@0.45.1
gcc@8.5.0                           glibc@2.28          py-flit-core@3.12.0  py-pyproject-metadata@0.9.1  python-venv@1.0
==> 58 installed packages
==> 0 concretized packages to be installed (show with `spack find -c`)
```

## Use LAMMPS

Once the environment is installed, all installed packages in the LAMMPS environment are available on the `PATH`, e.g.,:

```bash
[lammps] [jharvard@holy8a24102 spack_lammps]$ lmp -h

Large-scale Atomic/Molecular Massively Parallel Simulator - 29 Oct 2020

Usage example: lmp -var t 300 -echo screen -in in.alloy

List of command line options supported by this LAMMPS executable:

-echo none/screen/log/both  : echoing of input script (-e)
-help                       : print this help message (-h)
-in filename                : read input from file, not stdin (-i)
-kokkos on/off ...          : turn KOKKOS mode on or off (-k)
-log none/filename          : where to send log output (-l)
-mpicolor color             : which exe in a multi-exe mpirun cmd (-m)
-nocite                     : disable writing log.cite file (-nc)
-package style ...          : invoke package command (-pk)
-partition size1 size2 ...  : assign partition sizes (-p)
-plog basename              : basename for partition logs (-pl)
-pscreen basename           : basename for partition screens (-ps)
-restart2data rfile dfile ... : convert restart to data file (-r2data)
-restart2dump rfile dgroup dstyle dfile ...
                            : convert restart to dump file (-r2dump)
-reorder topology-specs     : processor reordering (-r)
-screen none/filename       : where to send screen output (-sc)
-suffix gpu/intel/opt/omp   : style suffix to apply (-sf)
-var varname value          : set index style variable (-v)

OS: Linux 4.18.0-513.18.1.el8_9.x86_64 on x86_64

Compiler: GNU C++ 14.2.0 with OpenMP 4.5
C++ standard: C++17
MPI v3.1: Open MPI v5.0.5, package: Open MPI pedmon@builds01.rc.fas.harvard.edu Distribution, ident: 5.0.5, repo rev: v5.0.5, Jul 22, 2024

Active compile time flags:

-DLAMMPS_GZIP
-DLAMMPS_SMALLBIG
sizeof(smallint): 32-bit
sizeof(imageint): 32-bit
sizeof(tagint):   32-bit
sizeof(bigint):   64-bit

Installed packages:

ASPHERE BODY CLASS2 COLLOID COMPRESS CORESHELL DIPOLE GRANULAR KOKKOS KSPACE
MANYBODY MC MISC MOLECULE MPIIO PERI PYTHON QEQ REPLICA RIGID SHOCK SNAP SPIN
SRD USER-MISC USER-REAXC

List of individual style options included in this LAMMPS executable

* Atom styles:

angle           angle/kk        angle/kk/device angle/kk/host   atomic
atomic/kk       atomic/kk/device                atomic/kk/host  body
bond            bond/kk         bond/kk/device  bond/kk/host    charge
charge/kk       charge/kk/device                charge/kk/host  dipole
ellipsoid       full            full/kk         full/kk/device  full/kk/host
hybrid          hybrid/kk       line            molecular       molecular/kk
molecular/kk/device             molecular/kk/host               peri
sphere          sphere/kk       sphere/kk/device                sphere/kk/host
spin            template        tri

* Integrate styles:

respa           verlet          verlet/kk       verlet/kk/device
verlet/kk/host  verlet/split

* Minimize styles:

cg              cg/kk           cg/kk/device    cg/kk/host      fire
fire/old        hftn            quickmin        sd              spin
spin/cg         spin/lbfgs

* Pair styles:

adp             agni            airebo          airebo/morse    atm
beck            body/nparticle  body/rounded/polygon
body/rounded/polyhedron         bop             born            born/coul/dsf
born/coul/dsf/cs                born/coul/long  born/coul/long/cs
born/coul/msm   born/coul/wolf  born/coul/wolf/cs               brownian
brownian/poly   buck            buck/coul/cut   buck/coul/cut/kk
buck/coul/cut/kk/device         buck/coul/cut/kk/host           buck/coul/long
buck/coul/long/cs               buck/coul/long/kk
buck/coul/long/kk/device        buck/coul/long/kk/host          buck/coul/msm
buck/kk         buck/kk/device  buck/kk/host    buck/long/coul/long
buck/mdf        colloid         comb            comb3           cosine/squared
coul/cut        coul/cut/kk     coul/cut/kk/device              coul/cut/kk/host
coul/debye      coul/debye/kk   coul/debye/kk/device
coul/debye/kk/host              coul/diel       coul/dsf        coul/dsf/kk
coul/dsf/kk/device              coul/dsf/kk/host                coul/long
coul/long/cs    coul/long/kk    coul/long/kk/device
coul/long/kk/host               coul/msm        coul/shield     coul/slater/cut
coul/slater/long                coul/streitz    coul/wolf       coul/wolf/cs
coul/wolf/kk    coul/wolf/kk/device             coul/wolf/kk/host
reax            dpd             dpd/tstat       drip            dsmc
e3b             eam             eam/alloy       eam/alloy/kk
eam/alloy/kk/device             eam/alloy/kk/host               eam/cd
eam/cd/old      eam/fs          eam/fs/kk       eam/fs/kk/device
eam/fs/kk/host  eam/kk          eam/kk/device   eam/kk/host     edip
edip/multi      eim             extep           gauss           gauss/cut
gayberne        gran/hertz/history              gran/hooke
gran/hooke/history              gran/hooke/history/kk
gran/hooke/history/kk/device    gran/hooke/history/kk/host      granular
gw              gw/zbl          hbond/dreiding/lj
hbond/dreiding/morse            hybrid          hybrid/kk       hybrid/overlay
hybrid/overlay/kk               ilp/graphene/hbn
kolmogorov/crespi/full          kolmogorov/crespi/z             lcbop
lebedeva/z      lennard/mdf     line/lj         list            lj96/cut
lj/charmm/coul/charmm           lj/charmm/coul/charmm/implicit
lj/charmm/coul/charmm/implicit/kk
lj/charmm/coul/charmm/implicit/kk/device
lj/charmm/coul/charmm/implicit/kk/host          lj/charmm/coul/charmm/kk
lj/charmm/coul/charmm/kk/device lj/charmm/coul/charmm/kk/host
lj/charmm/coul/long             lj/charmm/coul/long/kk
lj/charmm/coul/long/kk/device   lj/charmm/coul/long/kk/host
lj/charmm/coul/msm              lj/charmmfsw/coul/charmmfsh
lj/charmmfsw/coul/long          lj/class2       lj/class2/coul/cut
lj/class2/coul/cut/kk           lj/class2/coul/cut/kk/device
lj/class2/coul/cut/kk/host      lj/class2/coul/long
lj/class2/coul/long/cs          lj/class2/coul/long/kk
lj/class2/coul/long/kk/device   lj/class2/coul/long/kk/host     lj/class2/kk
lj/class2/kk/device             lj/class2/kk/host               lj/cubic
lj/cut          lj/cut/coul/cut lj/cut/coul/cut/kk
lj/cut/coul/cut/kk/device       lj/cut/coul/cut/kk/host
lj/cut/coul/debye               lj/cut/coul/debye/kk
lj/cut/coul/debye/kk/device     lj/cut/coul/debye/kk/host       lj/cut/coul/dsf
lj/cut/coul/dsf/kk              lj/cut/coul/dsf/kk/device
lj/cut/coul/dsf/kk/host         lj/cut/coul/long
lj/cut/coul/long/cs             lj/cut/coul/long/kk
lj/cut/coul/long/kk/device      lj/cut/coul/long/kk/host        lj/cut/coul/msm
lj/cut/coul/wolf                lj/cut/dipole/cut
lj/cut/dipole/long              lj/cut/kk       lj/cut/kk/device
lj/cut/kk/host  lj/cut/tip4p/cut                lj/cut/tip4p/long
lj/expand       lj/expand/coul/long             lj/expand/kk
lj/expand/kk/device             lj/expand/kk/host               lj/gromacs
lj/gromacs/coul/gromacs         lj/gromacs/coul/gromacs/kk
lj/gromacs/coul/gromacs/kk/device               lj/gromacs/coul/gromacs/kk/host
lj/gromacs/kk   lj/gromacs/kk/device            lj/gromacs/kk/host
lj/long/coul/long               lj/long/dipole/long
lj/long/tip4p/long              lj/mdf          lj/sf/dipole/sf lj/smooth
lj/smooth/linear                lj/sf           local/density   lubricate
lubricateU      lubricateU/poly lubricate/poly  meam/spline     meam/sw/spline
mie/cut         momb            morse           morse/kk        morse/kk/device
morse/kk/host   morse/smooth/linear             nb3b/harmonic   nm/cut
nm/cut/coul/cut nm/cut/coul/long                peri/eps        peri/lps
peri/pmb        peri/ves        polymorphic     python          reax/c
reax/c/kk       reax/c/kk/device                reax/c/kk/host  rebo
resquared       snap            snap/kk         snap/kk/device  snap/kk/host
soft            spin/dipole/cut spin/dipole/long                spin/dmi
spin/exchange   spin/magelec    spin/neel       srp             sw
sw/kk           sw/kk/device    sw/kk/host      table           table/kk
table/kk/device table/kk/host   tersoff         tersoff/kk
tersoff/kk/device               tersoff/kk/host tersoff/mod     tersoff/mod/c
tersoff/mod/kk  tersoff/mod/kk/device           tersoff/mod/kk/host
tersoff/table   tersoff/zbl     tersoff/zbl/kk  tersoff/zbl/kk/device
tersoff/zbl/kk/host             tip4p/cut       tip4p/long      tri/lj
ufm             vashishta       vashishta/kk    vashishta/kk/device
vashishta/kk/host               vashishta/table yukawa          yukawa/colloid
yukawa/kk       yukawa/kk/device                yukawa/kk/host  zbl
zbl/kk          zbl/kk/device   zbl/kk/host     zero

* Bond styles:

class2          class2/kk       class2/kk/device                class2/kk/host
fene            fene/expand     fene/kk         fene/kk/device  fene/kk/host
gromos          harmonic        harmonic/kk     harmonic/kk/device
harmonic/kk/host                harmonic/shift  harmonic/shift/cut
hybrid          morse           nonlinear       quartic         special
table           zero

* Angle styles:

charmm          charmm/kk       charmm/kk/device                charmm/kk/host
class2          class2/kk       class2/kk/device                class2/kk/host
cosine          cosine/delta    cosine/kk       cosine/kk/device
cosine/kk/host  cosine/periodic cosine/shift    cosine/shift/exp
cosine/squared  dipole          fourier         fourier/simple  harmonic
harmonic/kk     harmonic/kk/device              harmonic/kk/host
hybrid          quartic         table           zero

* Dihedral styles:

charmm          charmm/kk       charmm/kk/device                charmm/kk/host
charmmfsw       class2          class2/kk       class2/kk/device
class2/kk/host  cosine/shift/exp                fourier         harmonic
harmonic/kk     harmonic/kk/device              harmonic/kk/host
helix           hybrid          multi/harmonic  nharmonic       opls
opls/kk         opls/kk/device  opls/kk/host    quadratic       spherical
table           table/cut       zero

* Improper styles:

class2          class2/kk       class2/kk/device                class2/kk/host
cossq           cvff            distance        fourier         harmonic
harmonic/kk     harmonic/kk/device              harmonic/kk/host
hybrid          ring            umbrella        zero

* KSpace styles:

ewald           ewald/dipole    ewald/dipole/spin               ewald/disp
msm             msm/cg          pppm            pppm/cg         pppm/dipole
pppm/dipole/spin                pppm/disp       pppm/disp/tip4p pppm/kk
pppm/kk/device  pppm/kk/host    pppm/stagger    pppm/tip4p

* Fix styles

accelerate/cos  adapt           addforce        addtorque       append/atoms
atom/swap       ave/atom        ave/chunk       ave/correlate
ave/correlate/long              ave/histo       ave/histo/weight
ave/time        aveforce        balance         bond/break      bond/create
bond/create/angle               bond/swap       box/relax       cmap
controller      deform          deform/kk       deform/kk/device
deform/kk/host  deposit         ave/spatial     ave/spatial/sphere
drag            dt/reset        efield          ehex
electron/stopping               enforce2d       enforce2d/kk
enforce2d/kk/device             enforce2d/kk/host               evaporate
external        ffl             filter/corotate flow/gauss      freeze
freeze/kk       freeze/kk/device                freeze/kk/host  gcmc
gld             gle             gravity         gravity/kk
gravity/kk/device               gravity/kk/host grem            halt
heat            hyper/global    hyper/local     imd             indent
ipi             langevin        langevin/kk     langevin/kk/device
langevin/kk/host                langevin/spin   lineforce       momentum
momentum/chunk  momentum/kk     momentum/kk/device              momentum/kk/host
move            msst            neb             neb/spin        nph
nph/asphere     nph/body        nph/kk          nph/kk/device   nph/kk/host
nph/sphere      nphug           npt             npt/asphere     npt/body
npt/cauchy      npt/kk          npt/kk/device   npt/kk/host     npt/sphere
numdiff         nve             nve/asphere     nve/asphere/noforce
nve/body        nve/kk          nve/kk/device   nve/kk/host     nve/limit
nve/line        nve/noforce     nve/sphere      nve/sphere/kk
nve/sphere/kk/device            nve/sphere/kk/host              nve/spin
nve/tri         nvk             nvt             nvt/asphere     nvt/body
nvt/kk          nvt/kk/device   nvt/kk/host     nvt/sllod       nvt/sphere
oneway          orient/bcc      orient/eco      orient/fcc      pafi
pimd            planeforce      pour            precession/spin press/berendsen
print           propel/self     property/atom   property/atom/kk
python/invoke   python          python/move     qeq/comb        qeq/dynamic
qeq/dynamic     qeq/fire        qeq/fire        qeq/point       qeq/point
qeq/reax        qeq/reax/kk     qeq/reax/kk/device              qeq/reax/kk/host
qeq/shielded    qeq/shielded    qeq/slater      qeq/slater      rattle
reax/c/bonds    reax/c/bonds/kk reax/c/bonds/kk/device
reax/c/bonds/kk/host            reax/c/species  reax/c/species/kk
reax/c/species/kk/device        reax/c/species/kk/host          recenter
restrain        rhok            rigid           rigid/nph       rigid/nph/small
rigid/npt       rigid/npt/small rigid/nve       rigid/nve/small rigid/nvt
rigid/nvt/small rigid/small     setforce        setforce/kk
setforce/kk/device              setforce/kk/host                setforce/spin
shake           smd             spring          spring/chunk    spring/rg
spring/self     srd             store/force     store/state     temp/berendsen
temp/csld       temp/csvr       temp/rescale    tfmc
thermal/conductivity            ti/spring       tmd             ttm
ttm/mod         tune/kspace     vector          viscosity       viscous
wall/body/polygon               wall/body/polyhedron            wall/colloid
wall/ees        wall/gran       wall/gran/region                wall/harmonic
wall/lj1043     wall/lj126      wall/lj93       wall/lj93/kk
wall/lj93/kk/device             wall/lj93/kk/host               wall/morse
wall/piston     wall/reflect    wall/reflect/kk wall/reflect/kk/device
wall/reflect/kk/host            wall/reflect/stochastic         wall/region
wall/region/ees wall/srd        widom

* Compute styles:

ackland/atom    adf             aggregate/atom  angle           angle/local
angmom/chunk    basal/atom      body/local      bond            bond/local
centro/atom     centroid/stress/atom            chunk/atom
chunk/spread/atom               cluster/atom    cna/atom        cnp/atom
com             com/chunk       contact/atom    coord/atom      coord/atom/kk
coord/atom/kk/device            coord/atom/kk/host              damage/atom
dihedral        dihedral/local  dilatation/atom dipole/chunk    displace/atom
entropy/atom    erotate/asphere erotate/rigid   erotate/sphere
erotate/sphere/atom             event/displace  fragment/atom   global/atom
group/group     gyration        gyration/chunk  gyration/shape
gyration/shape/chunk            heat/flux       hexorder/atom   hma
improper        improper/local  inertia/chunk   ke              ke/atom
ke/rigid        momentum        msd             msd/chunk       msd/nongauss
omega/chunk     orientorder/atom                orientorder/atom/kk
orientorder/atom/kk/device      orientorder/atom/kk/host        pair
pair/local      pe              pe/atom         plasticity/atom pressure
pressure/cylinder               property/atom   property/chunk  property/local
rdf             reduce          reduce/chunk    reduce/region   rigid/local
slice           sna/atom        snad/atom       snap            snav/atom
spin            stress/atom     stress/mop      stress/mop/profile
temp            temp/asphere    temp/body       temp/chunk      temp/com
temp/cs         temp/deform     temp/kk         temp/kk/device  temp/kk/host
temp/partial    temp/profile    temp/ramp       temp/region     temp/rotate
temp/sphere     ti              torque/chunk    vacf            vcm/chunk
viscosity/cos

* Region styles:

block           block/kk        block/kk/device block/kk/host   cone
cylinder        intersect       plane           prism           sphere
union

* Dump styles:

atom            atom/gz         atom/mpiio      atom/zstd       cfg
cfg/gz          cfg/mpiio       cfg/zstd        custom          custom/gz
custom/mpiio    custom/zstd     dcd             image           local
local/gz        local/zstd      movie           xtc             xyz
xyz/gz          xyz/mpiio       xyz/zstd

* Command styles

balance         change_box      create_atoms    create_bonds    create_box
delete_atoms    delete_bonds    reset_ids       displace_atoms  hyper
info            minimize        neb             neb/spin        prd
read_data       read_dump       read_restart    replicate       rerun
reset_atom_ids  reset_mol_ids   run             set             tad
temper          temper/grem     temper/npt      velocity        write_coeff
write_data      write_dump      write_restart
```

### Interactive runs

You can run LAMMPS interactively in both serial and parallel mode. This assumes you have requested an interactive session first, as explained [here](https://docs.rc.fas.harvard.edu/kb/running-jobs/#articleTOC_14).

**Prerequisite:**

Source Spack and activate environment

```
### NOTE: Replace <PATH TO spack_lammps> with the actual path to your spack installation

[jharvard@holy8a24102 ~]$ cd <PATH TO spack_lammps>
[jharvard@holy8a24102 spack_lammps]$ source share/spack/setup-env.sh
[jharvard@holy8a24102 spack_lammps]$ spack env activate -p lammps
```

**Example: input file `in.demo` (3D Lennard-Jones melt)**

The examples below use [in.demo](https://github.com/lammps/lammps/blob/08d285655821cbaea1b803a21c9ff39b93e164c3/python/examples/in.demo#) from [LAMMPS repository](https://github.com/lammps/lammps).

```
[lammps] [jharvard@holy8a24301 spack_lammps]$ cat in.demo
# 3d Lennard-Jones melt

units		lj
atom_style	atomic
atom_modify	map hash

lattice		fcc 0.8442
region		box block 0 10 0 10 0 10
create_box	1 box
create_atoms	1 box
mass		1 1.0

velocity	all create 1.44 87287 loop geom

pair_style	lj/cut 2.5
pair_coeff	1 1 1.0 1.0 2.5

neighbor	0.3 bin
neigh_modify	delay 0 every 20 check no

fix		1 all nve

variable	eng equal pe
variable	vy atom vy

run		100
```

**Serial**

```
[lammps] [jharvard@holy8a24301 spack_lammps]$ lmp -in in.demo
LAMMPS (29 Oct 2020)
  using 1 OpenMP thread(s) per MPI task
Lattice spacing in x,y,z = 1.6795962 1.6795962 1.6795962
Created orthogonal box = (0.0000000 0.0000000 0.0000000) to (16.795962 16.795962 16.795962)
  1 by 1 by 1 MPI processor grid
Created 4000 atoms
  create_atoms CPU = 0.001 seconds
Neighbor list info ...
  update every 20 steps, delay 0 steps, check no
  max neighbors/atom: 2000, page size: 100000
  master list distance cutoff = 2.8
  ghost atom cutoff = 2.8
  binsize = 1.4, bins = 12 12 12
  1 neighbor lists, perpetual/occasional/extra = 1 0 0
  (1) pair lj/cut, perpetual
      attributes: half, newton on
      pair build: half/bin/atomonly/newton
      stencil: half/bin/3d/newton
      bin: standard
Setting up Verlet run ...
  Unit style    : lj
  Current step  : 0
  Time step     : 0.005
Per MPI rank memory allocation (min/avg/max) = 3.763 | 3.763 | 3.763 Mbytes
Step Temp E_pair E_mol TotEng Press
       0         1.44   -6.7733681            0   -4.6139081   -5.0199732
     100   0.75715334   -5.7581426            0   -4.6226965   0.20850222
Loop time of 0.166196 on 1 procs for 100 steps with 4000 atoms

Performance: 259934.223 tau/day, 601.700 timesteps/s
99.8% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0.1346     | 0.1346     | 0.1346     |   0.0 | 80.99
Neigh   | 0.026011   | 0.026011   | 0.026011   |   0.0 | 15.65
Comm    | 0.0026822  | 0.0026822  | 0.0026822  |   0.0 |  1.61
Output  | 2.4151e-05 | 2.4151e-05 | 2.4151e-05 |   0.0 |  0.01
Modify  | 0.0023231  | 0.0023231  | 0.0023231  |   0.0 |  1.40
Other   |            | 0.0005541  |            |       |  0.33

Nlocal:        4000.00 ave        4000 max        4000 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:        5754.00 ave        5754 max        5754 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:       150362.0 ave      150362 max      150362 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 150362
Ave neighs/atom = 37.590500
Neighbor list builds = 5
Dangerous builds not checked
Total wall time: 0:00:00
```

**Parallel (e.g., 4 MPI tasks)**

```
[lammps] [jharvard@holy8a24301 spack_lammps]$ mpirun -np 4 lmp -in in.demo
LAMMPS (29 Oct 2020)
  using 1 OpenMP thread(s) per MPI task
Lattice spacing in x,y,z = 1.6795962 1.6795962 1.6795962
Created orthogonal box = (0.0000000 0.0000000 0.0000000) to (16.795962 16.795962 16.795962)
  1 by 2 by 2 MPI processor grid
Created 4000 atoms
  create_atoms CPU = 0.001 seconds
Neighbor list info ...
  update every 20 steps, delay 0 steps, check no
  max neighbors/atom: 2000, page size: 100000
  master list distance cutoff = 2.8
  ghost atom cutoff = 2.8
  binsize = 1.4, bins = 12 12 12
  1 neighbor lists, perpetual/occasional/extra = 1 0 0
  (1) pair lj/cut, perpetual
      attributes: half, newton on
      pair build: half/bin/atomonly/newton
      stencil: half/bin/3d/newton
      bin: standard
Setting up Verlet run ...
  Unit style    : lj
  Current step  : 0
  Time step     : 0.005
Per MPI rank memory allocation (min/avg/max) = 3.224 | 3.224 | 3.224 Mbytes
Step Temp E_pair E_mol TotEng Press
       0         1.44   -6.7733681            0   -4.6139081   -5.0199732
     100   0.75715334   -5.7581426            0   -4.6226965   0.20850222
Loop time of 0.0487977 on 4 procs for 100 steps with 4000 atoms

Performance: 885288.044 tau/day, 2049.278 timesteps/s
96.1% CPU use with 4 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0.034078   | 0.034385   | 0.034719   |   0.1 | 70.46
Neigh   | 0.0066096  | 0.0066407  | 0.0066986  |   0.0 | 13.61
Comm    | 0.0065501  | 0.0069472  | 0.0072876  |   0.3 | 14.24
Output  | 2.2972e-05 | 2.5369e-05 | 3.2187e-05 |   0.0 |  0.05
Modify  | 0.0005837  | 0.00059166 | 0.00060061 |   0.0 |  1.21
Other   |            | 0.0002079  |            |       |  0.43

Nlocal:        1000.00 ave        1019 max         980 min
Histogram: 1 0 0 0 0 2 0 0 0 1
Nghost:        2858.75 ave        2881 max        2844 min
Histogram: 1 1 0 0 1 0 0 0 0 1
Neighs:        37590.5 ave       38013 max       37269 min
Histogram: 2 0 0 0 0 0 0 1 0 1

Total # of neighbors = 150362
Ave neighs/atom = 37.590500
Neighbor list builds = 5
Dangerous builds not checked
Total wall time: 0:00:00
```

### Batch jobs

**Example batch job submission script**

Below is an example batch-job submission script `run_lammps.sh` using the LAMMPS spack environment.

```bash
#!/bin/bash
#SBATCH -J lammps_test        # job name
#SBATCH -o lammps_test.out    # standard output file
#SBATCH -e lammps_test.err    # standard error file
#SBATCH -p shared             # partition
#SBATCH -n 4                  # ntasks
#SBATCH -t 00:30:00           # time in HH:MM:SS
#SBATCH --mem-per-cpu=500     # memory in megabytes

# --- Activate the LAMMPS Spack environment., e.g., ---
### NOTE: Replace <PATH TO> with the actual path to your spack installation
. <PATH TO>/spack_lammps/share/spack/setup-env.sh
spack env activate lammps

# --- Run the executable ---
srun -n $SLURM_NTASKS --mpi=pmix lmp -in in.demo
```

Submit the job

```
sbatch run_lammps.sh
```


## References:

* [LAMMPS official website](https://www.lammps.org/index.html#gsc.tab=0)
* [LAMMPS Documentation](https://docs.lammps.org/Manual.html)
