# Build LAMMPS MPI version using Spack

<img src="Images/spack-logo.svg" alt="spack-logo" width="100"/>

[LAMMPS](https://www.lammps.org/#gsc.tab=0) is a classical molecular dynamics code with a focus on materials modeling. It's an acronym for Large-scale Atomic/Molecular Massively Parallel Simulator.

The below instructions provide a spack recipe for building MPI capable instance of LAMMPS on the [FASRC Cannon cluster](https://www.rc.fas.harvard.edu/about/cluster-architecture/).

## Compiler and MPI Library Spack configuration

Here we will use the [GNU/GCC](https://gcc.gnu.org/) compiler suite together with [OpenMPI](https://www.open-mpi.org/).

 The below instructions assume that spack is already configured to use the GCC compiler `gcc@12.2.0` and OpenMPI Library `openmpi@4.1.5`, as explained [here](https://github.com/fasrc/User_Codes/blob/master/Documents/Software/Spack.md).

## Create LAMMPS spack environment and activate it

```bash
spack env create lammps
spack env activate -p lammps
```

## Install the LAMMPS environment

### Install `libbsd`

> **Note:** In this recipe, we first install `libbsd` with the system version of the GCC compiler, `gcc@8.5.0`, as the installation fails, if we try to add it directly to the environment and install it with `gcc@12.2.0`.

```bash
spack install --add libbsd@0.11.6 % gcc@8.5.0
```

### Add the rest of the required packages to the spack environment

```bash
spack add openmpi@4.1.5
spack add lammps +asphere +body +class2 +colloid +compress +coreshell +dipole +granular +kokkos +kspace +manybody +mc +misc +molecule +mpiio +openmp-package +peri +python +qeq +replica +rigid +shock +snap +spin +srd +user-reaxc +user-misc % gcc@12.2.0 ^ openmpi@4.1.5
```

### Install the environment

Once all required packages are added to the environment, it can be installed with:

```bash
spack install
```
For example,

```bash
[lammps] [pkrastev@builds01 spack]$ spack add lammps +asphere +body +class2 +colloid +compress +coreshell +dipole +granular +kokkos +kspace +manybody +mc +misc +molecule +mpiio +openmp-package +peri +python +qeq +replica +rigid +shock +snap +spin +srd +user-reaxc +user-misc % gcc@12.2.0 ^ openmpi@4.1.5
==> Adding lammps%gcc@12.2.0+asphere+body+class2+colloid+compress+coreshell+dipole+granular+kokkos+kspace+manybody+mc+misc+molecule+mpiio+openmp-package+peri+python+qeq+replica+rigid+shock+snap+spin+srd+user-misc+user-reaxc ^openmpi@4.1.5 to environment lammps
[lammps] [pkrastev@builds01 spack]$ spack install
==> Concretized lammps%gcc@12.2.0+asphere+body+class2+colloid+compress+coreshell+dipole+granular+kokkos+kspace+manybody+mc+misc+molecule+mpiio+openmp-package+peri+python+qeq+replica+rigid+shock+snap+spin+srd+user-misc+user-reaxc ^openmpi@4.1.5
 -   brvuawz  lammps@20201029%gcc@12.2.0+asphere+body+class2+colloid+compress+coreshell~cuda~cuda_mps+dipole~exceptions~ffmpeg+granular~ipo~jpeg~kim+kokkos+kspace~latte+lib+manybody+mc+misc~mliap+molecule+mpi+mpiio~opencl+openmp+openmp-package~opt+peri~png~poems+python+qeq+replica+rigid~rocm+shock+snap+spin+srd~user-adios~user-atc~user-awpmd~user-bocs~user-cgsdk~user-colvars~user-diffraction~user-dpd~user-drude~user-eff~user-fep~user-h5md~user-intel~user-lb~user-manifold~user-meamc~user-mesodpd~user-mesont~user-mgpt+user-misc~user-mofff~user-molfile~user-netcdf~user-omp~user-phonon~user-plumed~user-ptm~user-qtb~user-reaction+user-reaxc~user-sdpd~user-smd~user-smtbq~user-sph~user-tally~user-uef~user-yaff~voronoi build_system=cmake build_type=Release fftw_precision=double generator=make lammps_sizes=smallbig arch=linux-rocky8-x86_64
[+]  gfpb5mz      ^cmake@3.27.7%gcc@12.2.0~doc+ncurses+ownlibs build_system=generic build_type=Release arch=linux-rocky8-x86_64
[+]  yvt4hk5          ^curl@8.4.0%gcc@12.2.0~gssapi~ldap~libidn2~librtmp~libssh~libssh2+nghttp2 build_system=autotools libs=shared,static tls=openssl arch=linux-rocky8-x86_64
[+]  lvshkiv              ^nghttp2@1.57.0%gcc@12.2.0 build_system=autotools arch=linux-rocky8-x86_64
[+]  enjvgjw          ^ncurses@6.4%gcc@12.2.0~symlinks+termlib abi=none build_system=autotools arch=linux-rocky8-x86_64
[+]  vydxoz7          ^zlib-ng@2.1.4%gcc@12.2.0+compat+opt build_system=autotools arch=linux-rocky8-x86_64
[+]  o3gyqz4      ^fftw@3.3.10%gcc@12.2.0+mpi~openmp~pfft_patches build_system=autotools precision=double,float arch=linux-rocky8-x86_64
[+]  svtpsvv      ^gmake@4.4.1%gcc@12.2.0~guile build_system=generic arch=linux-rocky8-x86_64
[+]  jiw5kkj      ^kokkos@3.7.02%gcc@12.2.0~aggressive_vectorization~compiler_warnings~cuda~debug~debug_bounds_check~debug_dualview_modify_check~deprecated_code~examples~hpx~hpx_async_dispatch~hwloc~ipo~memkind~numactl~openmp~openmptarget~pic~rocm+serial+shared~sycl~tests~threads~tuning~wrapper build_system=cmake build_type=Release cxxstd=17 generator=make intel_gpu_arch=none arch=linux-rocky8-x86_64
[e]  rzl24ya      ^openmpi@4.1.5%gcc@12.2.0~atomics~cuda~cxx~cxx_exceptions~gpfs~internal-hwloc~internal-pmix~java~legacylaunchers~lustre~memchecker~openshmem~orterunprefix+romio+rsh~singularity+static+vt+wrapper-rpath build_system=autotools fabrics=none schedulers=none arch=linux-rocky8-x86_64
 -   eo7i4v4      ^py-mpi4py@3.1.4%gcc@12.2.0 build_system=python_pip arch=linux-rocky8-x86_64
 -   p66ce3e          ^py-cython@0.29.36%gcc@12.2.0 build_system=python_pip patches=c4369ad arch=linux-rocky8-x86_64
 -   uwtiq6m          ^py-pip@23.1.2%gcc@12.2.0 build_system=generic arch=linux-rocky8-x86_64
 -   beehlpu          ^py-setuptools@68.0.0%gcc@12.2.0 build_system=generic arch=linux-rocky8-x86_64
 -   dwr4bwe          ^py-wheel@0.41.2%gcc@12.2.0 build_system=generic arch=linux-rocky8-x86_64
 -   5hw7nhs      ^py-numpy@1.26.1%gcc@12.2.0 build_system=python_pip patches=873745d arch=linux-rocky8-x86_64
 -   kqjh467          ^ninja@1.11.1%gcc@12.2.0+re2c build_system=generic arch=linux-rocky8-x86_64
[+]  7kut2tz              ^re2c@2.2%gcc@12.2.0 build_system=generic arch=linux-rocky8-x86_64
[+]  5oqjy7z          ^openblas@0.3.24%gcc@12.2.0~bignuma~consistent_fpcsr+fortran~ilp64+locking+pic+shared build_system=makefile symbol_suffix=none threads=none arch=linux-rocky8-x86_64
[+]  k7dhgbu              ^perl@5.38.0%gcc@12.2.0+cpanm+opcode+open+shared+threads build_system=generic patches=714e4d1 arch=linux-rocky8-x86_64
[+]  gyedpff                  ^berkeley-db@18.1.40%gcc@12.2.0+cxx~docs+stl build_system=autotools patches=26090f4,b231fcc arch=linux-rocky8-x86_64
[+]  bd3npcq          ^pkgconf@1.9.5%gcc@12.2.0 build_system=autotools arch=linux-rocky8-x86_64
 -   u2hdpyq          ^py-pyproject-metadata@0.7.1%gcc@12.2.0 build_system=python_pip arch=linux-rocky8-x86_64
 -   qkyvv7u              ^py-packaging@23.1%gcc@12.2.0 build_system=python_pip arch=linux-rocky8-x86_64
 -   zo3mqxg                  ^py-flit-core@3.9.0%gcc@12.2.0 build_system=python_pip arch=linux-rocky8-x86_64
 -   ayhpfzo      ^python@3.11.6%gcc@12.2.0+bz2+crypt+ctypes+dbm~debug+libxml2+lzma~nis~optimizations+pic+pyexpat+pythoncmd+readline+shared+sqlite3+ssl~tkinter+uuid+zlib build_system=generic patches=13fa8bf,b0615b2,ebdca64,f2fd060 arch=linux-rocky8-x86_64
[+]  xxvougl          ^bzip2@1.0.8%gcc@12.2.0~debug~pic+shared build_system=generic arch=linux-rocky8-x86_64
[+]  6delrmq              ^diffutils@3.9%gcc@12.2.0 build_system=autotools arch=linux-rocky8-x86_64
 -   utxeqjk          ^expat@2.5.0%gcc@12.2.0+libbsd build_system=autotools arch=linux-rocky8-x86_64
[+]  p4j7tjd              ^libbsd@0.11.6%gcc@8.5.0 build_system=autotools arch=linux-rocky8-x86_64
[+]  dhtpyny                  ^gmake@4.4.1%gcc@8.5.0~guile build_system=generic arch=linux-rocky8-x86_64
[+]  zgocxqu                  ^libmd@1.0.4%gcc@8.5.0 build_system=autotools arch=linux-rocky8-x86_64
[+]  r4ncx2h          ^gdbm@1.23%gcc@12.2.0 build_system=autotools arch=linux-rocky8-x86_64
[+]  nbmdkeh          ^gettext@0.22.3%gcc@12.2.0+bzip2+curses+git~libunistring+libxml2+pic+shared+tar+xz build_system=autotools arch=linux-rocky8-x86_64
[+]  4ic3bei              ^libiconv@1.17%gcc@12.2.0 build_system=autotools libs=shared,static arch=linux-rocky8-x86_64
[+]  vib3s6c              ^libxml2@2.10.3%gcc@12.2.0+pic~python+shared build_system=autotools arch=linux-rocky8-x86_64
[+]  bmtzwo2              ^tar@1.34%gcc@12.2.0 build_system=autotools zip=pigz arch=linux-rocky8-x86_64
[+]  g5ndpd7                  ^pigz@2.7%gcc@12.2.0 build_system=makefile arch=linux-rocky8-x86_64
[+]  kjif3oe                  ^zstd@1.5.5%gcc@12.2.0+programs build_system=makefile compression=none libs=shared,static arch=linux-rocky8-x86_64
[+]  vxfrf2m          ^libffi@3.4.4%gcc@12.2.0 build_system=autotools arch=linux-rocky8-x86_64
[+]  pevl53e          ^libxcrypt@4.4.35%gcc@12.2.0~obsolete_api build_system=autotools patches=4885da3 arch=linux-rocky8-x86_64
[+]  t6g7cgp          ^openssl@3.1.3%gcc@12.2.0~docs+shared build_system=generic certs=mozilla arch=linux-rocky8-x86_64
[+]  y4pl22s              ^ca-certificates-mozilla@2023-05-30%gcc@12.2.0 build_system=generic arch=linux-rocky8-x86_64
[+]  gqnfuu6          ^readline@8.2%gcc@12.2.0 build_system=autotools patches=bbf97f1 arch=linux-rocky8-x86_64
[+]  ir4yzhv          ^sqlite@3.43.2%gcc@12.2.0+column_metadata+dynamic_extensions+fts~functions+rtree build_system=autotools arch=linux-rocky8-x86_64
[+]  2lrosds          ^util-linux-uuid@2.38.1%gcc@12.2.0 build_system=autotools arch=linux-rocky8-x86_64
[+]  o2frdmd          ^xz@5.4.1%gcc@12.2.0~pic build_system=autotools libs=shared,static arch=linux-rocky8-x86_64

[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/nghttp2-1.57.0-lvshkivi7b4qjhtdvew667hctxypbszb
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/zlib-ng-2.1.4-vydxoz7fvgvsutjrmly6cayjaedjyrmg
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/ncurses-6.4-enjvgjww5qpxthriw5sywb7isei43bww
[+] /n/sw/helmod-rocky8/apps/Comp/gcc/12.2.0-fasrc01/openmpi/4.1.5-fasrc03 (external openmpi-4.1.5-rzl24yaivlkwvg4g5qznqm3vv32l2fh3)
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/gmake-4.4.1-svtpsvvjlmcqaoikxzgv7tn65gm5qkd2
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/bzip2-1.0.8-xxvougl2pg4l2ovlb763elztdhdigg4c
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-8.5.0/libmd-1.0.4-zgocxqucujk4t6egi7imbdmsmk6kf7hm
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/re2c-2.2-7kut2tz4mzisyoxxpdzw5fsoserrhg5b
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/openssl-3.1.3-t6g7cgpe6j762uy7k6tywuvoehrpoxlr
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/util-linux-uuid-2.38.1-2lrosds6dyqiwq2dqezt4rfxpilyw7mx
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/pkgconf-1.9.5-bd3npcqaivwm7xw6ryfdx4ngkhsqvw5y
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/fftw-3.3.10-o3gyqz4xr6icuwhvkij36gelql4w7zhe
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/xz-5.4.1-o2frdmd6pdpc5shh4b7ceuh6m3r24tky
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/libxcrypt-4.4.35-pevl53etfet57jj6otonswfahl4lpgln
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/libiconv-1.17-4ic3bei7fcc3od6alzc4xhoag6tovpc6
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/pigz-2.7-g5ndpd7evf6ikjvebjge3twqrifi3pio
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/openblas-0.3.24-5oqjy7z5brnhphx4cutohw242bfxnk7m
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/kokkos-3.7.02-jiw5kkjl63zeyy45klhoojvcbc2yndbq
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/readline-8.2-gqnfuu6cimjmva7lm3226ajtlbbf33wb
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/zstd-1.5.5-kjif3oebskvl7e2ctnhmfjeyllpfdb44
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/libffi-3.4.4-vxfrf2modrtt52i565y3vtph64q7kdgi
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-8.5.0/libbsd-0.11.6-p4j7tjdjyqh53iyoeeui2npoompdw33g
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/curl-8.4.0-yvt4hk57jtntsqm3n3jund52c5b6j3yf
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/libxml2-2.10.3-vib3s6cnyl2yvtxwpneekgtl2wrbbhmr
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/gdbm-1.23-r4ncx2hr7cny6evxb65xredi36qykaee
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/sqlite-3.43.2-ir4yzhvm3qpqqtznwpyzb6i34emgbtzw
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/tar-1.34-bmtzwo27cebonnqmsq7bmgff334itq4a
==> Installing expat-2.5.0-utxeqjkqdtzrd4hj6thql2awuuyn6nu2 [28/42]
==> No binary for expat-2.5.0-utxeqjkqdtzrd4hj6thql2awuuyn6nu2 found: installing from source
==> Fetching https://mirror.spack.io/_source-cache/archive/6f/6f0e6e01f7b30025fa05c85fdad1e5d0ec7fd35d9f61b22f34998de11969ff67.tar.bz2
==> No patches needed for expat
==> expat: Executing phase: 'autoreconf'
==> expat: Executing phase: 'configure'
==> expat: Executing phase: 'build'
==> expat: Executing phase: 'install'
==> expat: Successfully installed expat-2.5.0-utxeqjkqdtzrd4hj6thql2awuuyn6nu2
  Stage: 0.32s.  Autoreconf: 0.00s.  Configure: 11.32s.  Build: 5.61s.  Install: 0.60s.  Post-install: 0.06s.  Total: 18.07s
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/expat-2.5.0-utxeqjkqdtzrd4hj6thql2awuuyn6nu2
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/cmake-3.27.7-gfpb5mzafl4j7znv6r3z7v3n65y4lmhd
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/gettext-0.22.3-nbmdkehmpboc6ujcs7i4h7nkzchrtmlz
==> Installing python-3.11.6-ayhpfzopafuhgnbm76kgzhcwzzj3bgk6 [31/42]
==> No binary for python-3.11.6-ayhpfzopafuhgnbm76kgzhcwzzj3bgk6 found: installing from source
==> Fetching https://www.python.org/ftp/python/3.11.6/Python-3.11.6.tgz
==> Applied patch /builds/pkrastev/Spack/spack/var/spack/repos/builtin/packages/python/python-3.7.4+-distutils-C++-testsuite.patch
==> Applied patch /builds/pkrastev/Spack/spack/var/spack/repos/builtin/packages/python/python-3.11-distutils-C++.patch
==> Applied patch /builds/pkrastev/Spack/spack/var/spack/repos/builtin/packages/python/tkinter-3.11.patch
==> Applied patch /builds/pkrastev/Spack/spack/var/spack/repos/builtin/packages/python/rpath-non-gcc.patch
==> Ran patch() for python
==> python: Executing phase: 'configure'
==> python: Executing phase: 'build'
==> python: Executing phase: 'install'
==> python: Successfully installed python-3.11.6-ayhpfzopafuhgnbm76kgzhcwzzj3bgk6
  Stage: 2.25s.  Configure: 1m 13.81s.  Build: 21.41s.  Install: 21.77s.  Post-install: 1.42s.  Total: 2m 0.87s
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/python-3.11.6-ayhpfzopafuhgnbm76kgzhcwzzj3bgk6
==> Installing ninja-1.11.1-kqjh467ra53zkcgsfw2f2seztreu6cn7 [32/42]
==> No binary for ninja-1.11.1-kqjh467ra53zkcgsfw2f2seztreu6cn7 found: installing from source
==> Fetching https://mirror.spack.io/_source-cache/archive/31/31747ae633213f1eda3842686f83c2aa1412e0f5691d1c14dbbcc67fe7400cea.tar.gz
==> No patches needed for ninja
==> ninja: Executing phase: 'configure'
==> ninja: Executing phase: 'install'
==> ninja: Successfully installed ninja-1.11.1-kqjh467ra53zkcgsfw2f2seztreu6cn7
  Stage: 0.24s.  Configure: 28.56s.  Install: 0.01s.  Post-install: 0.08s.  Total: 29.06s
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/ninja-1.11.1-kqjh467ra53zkcgsfw2f2seztreu6cn7
==> Installing py-pip-23.1.2-uwtiq6mvi75v6eztiwl35ehuwvdjlzvq [33/42]
==> No binary for py-pip-23.1.2-uwtiq6mvi75v6eztiwl35ehuwvdjlzvq found: installing from source
==> Fetching https://mirror.spack.io/_source-cache/archive/3e/3ef6ac33239e4027d9a5598a381b9d30880a1477e50039db2eac6e8a8f6d1b18
==> No patches needed for py-pip
==> py-pip: Executing phase: 'install'
==> py-pip: Successfully installed py-pip-23.1.2-uwtiq6mvi75v6eztiwl35ehuwvdjlzvq
  Stage: 0.27s.  Install: 2.75s.  Post-install: 0.23s.  Total: 3.38s
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/py-pip-23.1.2-uwtiq6mvi75v6eztiwl35ehuwvdjlzvq
==> Installing py-wheel-0.41.2-dwr4bwera5xqx3tcjni2xwle35yni2xv [34/42]
==> No binary for py-wheel-0.41.2-dwr4bwera5xqx3tcjni2xwle35yni2xv found: installing from source
==> Fetching https://files.pythonhosted.org/packages/py3/w/wheel/wheel-0.41.2-py3-none-any.whl
==> No patches needed for py-wheel
==> py-wheel: Executing phase: 'install'
==> py-wheel: Successfully installed py-wheel-0.41.2-dwr4bwera5xqx3tcjni2xwle35yni2xv
  Stage: 0.96s.  Install: 0.57s.  Post-install: 0.07s.  Total: 1.73s
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/py-wheel-0.41.2-dwr4bwera5xqx3tcjni2xwle35yni2xv
==> Installing py-setuptools-68.0.0-beehlpujcxge7rergkdu2sqtd3juztlx [35/42]
==> No binary for py-setuptools-68.0.0-beehlpujcxge7rergkdu2sqtd3juztlx found: installing from source
==> Fetching https://mirror.spack.io/_source-cache/archive/11/11e52c67415a381d10d6b462ced9cfb97066179f0e871399e006c4ab101fc85f
==> No patches needed for py-setuptools
==> py-setuptools: Executing phase: 'install'
==> py-setuptools: Successfully installed py-setuptools-68.0.0-beehlpujcxge7rergkdu2sqtd3juztlx
  Stage: 0.10s.  Install: 0.97s.  Post-install: 0.13s.  Total: 1.33s
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/py-setuptools-68.0.0-beehlpujcxge7rergkdu2sqtd3juztlx
==> Installing py-flit-core-3.9.0-zo3mqxgcedttrjdp3s3e44ws3og4zn53 [36/42]
==> No binary for py-flit-core-3.9.0-zo3mqxgcedttrjdp3s3e44ws3og4zn53 found: installing from source
==> Fetching https://files.pythonhosted.org/packages/source/f/flit-core/flit_core-3.9.0.tar.gz
==> No patches needed for py-flit-core
==> py-flit-core: Executing phase: 'install'
==> py-flit-core: Successfully installed py-flit-core-3.9.0-zo3mqxgcedttrjdp3s3e44ws3og4zn53
  Stage: 0.92s.  Install: 0.81s.  Post-install: 0.09s.  Total: 1.94s
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/py-flit-core-3.9.0-zo3mqxgcedttrjdp3s3e44ws3og4zn53
==> Installing py-cython-0.29.36-p66ce3exy7gg5fta3sfefwwwh6lwc2w2 [37/42]
==> No binary for py-cython-0.29.36-p66ce3exy7gg5fta3sfefwwwh6lwc2w2 found: installing from source
==> Fetching https://mirror.spack.io/_source-cache/archive/41/41c0cfd2d754e383c9eeb95effc9aa4ab847d0c9747077ddd7c0dcb68c3bc01f.tar.gz
==> Applied patch /builds/pkrastev/Spack/spack/var/spack/repos/builtin/packages/py-cython/5712.patch
==> py-cython: Executing phase: 'install'
==> py-cython: Successfully installed py-cython-0.29.36-p66ce3exy7gg5fta3sfefwwwh6lwc2w2
  Stage: 0.46s.  Install: 58.86s.  Post-install: 0.15s.  Total: 59.59s
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/py-cython-0.29.36-p66ce3exy7gg5fta3sfefwwwh6lwc2w2
==> Installing py-packaging-23.1-qkyvv7uu42txcfawduemltgl7ddjy4jn [38/42]
==> No binary for py-packaging-23.1-qkyvv7uu42txcfawduemltgl7ddjy4jn found: installing from source
==> Fetching https://mirror.spack.io/_source-cache/archive/a3/a392980d2b6cffa644431898be54b0045151319d1e7ec34f0cfed48767dd334f.tar.gz
==> No patches needed for py-packaging
==> py-packaging: Executing phase: 'install'
==> py-packaging: Successfully installed py-packaging-23.1-qkyvv7uu42txcfawduemltgl7ddjy4jn
  Stage: 0.26s.  Install: 0.76s.  Post-install: 0.07s.  Total: 1.23s
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/py-packaging-23.1-qkyvv7uu42txcfawduemltgl7ddjy4jn
==> Installing py-mpi4py-3.1.4-eo7i4v4wgy35fuvmf76aygklxnxzkbs2 [39/42]
==> No binary for py-mpi4py-3.1.4-eo7i4v4wgy35fuvmf76aygklxnxzkbs2 found: installing from source
==> Fetching https://mirror.spack.io/_source-cache/archive/17/17858f2ebc623220d0120d1fa8d428d033dde749c4bc35b33d81a66ad7f93480.tar.gz
==> No patches needed for py-mpi4py
==> py-mpi4py: Executing phase: 'install'
==> py-mpi4py: Successfully installed py-mpi4py-3.1.4-eo7i4v4wgy35fuvmf76aygklxnxzkbs2
  Stage: 1.40s.  Install: 49.94s.  Post-install: 0.08s.  Total: 51.58s
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/py-mpi4py-3.1.4-eo7i4v4wgy35fuvmf76aygklxnxzkbs2
==> Installing py-pyproject-metadata-0.7.1-u2hdpyqmxrrs54rhmddn6rwbawbqvn4z [40/42]
==> No binary for py-pyproject-metadata-0.7.1-u2hdpyqmxrrs54rhmddn6rwbawbqvn4z found: installing from source
==> Fetching https://mirror.spack.io/_source-cache/archive/0a/0a94f18b108b9b21f3a26a3d541f056c34edcb17dc872a144a15618fed7aef67.tar.gz
==> No patches needed for py-pyproject-metadata
==> py-pyproject-metadata: Executing phase: 'install'
==> py-pyproject-metadata: Successfully installed py-pyproject-metadata-0.7.1-u2hdpyqmxrrs54rhmddn6rwbawbqvn4z
  Stage: 0.23s.  Install: 0.96s.  Post-install: 0.07s.  Total: 1.40s
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/py-pyproject-metadata-0.7.1-u2hdpyqmxrrs54rhmddn6rwbawbqvn4z
==> Installing py-numpy-1.26.1-5hw7nhsp23hc6m35emvtzzuficgf5745 [41/42]
==> No binary for py-numpy-1.26.1-5hw7nhsp23hc6m35emvtzzuficgf5745 found: installing from source
==> Fetching https://files.pythonhosted.org/packages/source/n/numpy/numpy-1.26.1.tar.gz
==> Applied patch /builds/pkrastev/Spack/spack/var/spack/repos/builtin/packages/py-numpy/check_executables.patch
==> py-numpy: Executing phase: 'install'
==> py-numpy: Successfully installed py-numpy-1.26.1-5hw7nhsp23hc6m35emvtzzuficgf5745
  Stage: 1.56s.  Install: 1m 10.30s.  Post-install: 0.34s.  Total: 1m 12.37s
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/py-numpy-1.26.1-5hw7nhsp23hc6m35emvtzzuficgf5745
==> Installing lammps-20201029-brvuawzoq5ge6ak27a7474ziesqmbbq4 [42/42]
==> No binary for lammps-20201029-brvuawzoq5ge6ak27a7474ziesqmbbq4 found: installing from source
==> Fetching https://mirror.spack.io/_source-cache/archive/75/759705e16c1fedd6aa6e07d028cc0c78d73c76b76736668420946a74050c3726.tar.gz
==> No patches needed for lammps
==> lammps: Executing phase: 'cmake'
==> lammps: Executing phase: 'build'
==> lammps: Executing phase: 'install'
==> lammps: Successfully installed lammps-20201029-brvuawzoq5ge6ak27a7474ziesqmbbq4
  Stage: 8.79s.  Cmake: 8.14s.  Build: 2m 12.32s.  Install: 1.78s.  Post-install: 0.21s.  Total: 2m 31.51s
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/lammps-20201029-brvuawzoq5ge6ak27a7474ziesqmbbq4
==> Updating view at /builds/pkrastev/Spack/spack/var/spack/environments/lammps/.spack-env/view
==> Warning: Skipping external package: openmpi@4.1.5%gcc@12.2.0~atomics~cuda~cxx~cxx_exceptions~gpfs~internal-hwloc~internal-pmix~java~legacylaunchers~lustre~memchecker~openshmem~orterunprefix+romio+rsh~singularity+static+vt+wrapper-rpath build_system=autotools fabrics=none schedulers=none arch=linux-rocky8-x86_64/rzl24ya
```

### List the packages installed in the LAMMPS environment

```bash
[lammps] [pkrastev@builds01 spack]$ spack find
==> In environment lammps
==> Root specs
openmpi@4.1.5 

-- no arch / gcc@8.5.0 ------------------------------------------
libbsd@0.11.6%gcc@8.5.0 

-- no arch / gcc@12.2.0 -----------------------------------------
lammps%gcc@12.2.0 +asphere+body+class2+colloid+compress+coreshell+dipole+granular+kokkos+kspace+manybody+mc+misc+molecule+mpiio+openmp-package+peri+python+qeq+replica+rigid+shock+snap+spin+srd+user-misc+user-reaxc

==> Installed packages
-- linux-rocky8-x86_64 / gcc@8.5.0 ------------------------------
gmake@4.4.1  libbsd@0.11.6  libmd@1.0.4

-- linux-rocky8-x86_64 / gcc@12.2.0 -----------------------------
berkeley-db@18.1.40                 diffutils@3.9   gmake@4.4.1      libxcrypt@4.4.35  openblas@0.3.24  pkgconf@1.9.5       py-packaging@23.1            python@3.11.6  util-linux-uuid@2.38.1
bzip2@1.0.8                         expat@2.5.0     kokkos@3.7.02    libxml2@2.10.3    openmpi@4.1.5    py-cython@0.29.36   py-pip@23.1.2                re2c@2.2       xz@5.4.1
ca-certificates-mozilla@2023-05-30  fftw@3.3.10     lammps@20201029  ncurses@6.4       openssl@3.1.3    py-flit-core@3.9.0  py-pyproject-metadata@0.7.1  readline@8.2   zlib-ng@2.1.4
cmake@3.27.7                        gdbm@1.23       libffi@3.4.4     nghttp2@1.57.0    perl@5.38.0      py-mpi4py@3.1.4     py-setuptools@68.0.0         sqlite@3.43.2  zstd@1.5.5
curl@8.4.0                          gettext@0.22.3  libiconv@1.17    ninja@1.11.1      pigz@2.7         py-numpy@1.26.1     py-wheel@0.41.2              tar@1.34
==> 47 installed packages
```

## Use LAMMPS

Once the environment is installed, all installed packages in the LAMMPS environment are available on the `PATH`, e.g.,:

```bash
[lammps] [pkrastev@builds01 ~]$ lmp -h

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

OS: Linux 4.18.0-425.10.1.el8_7.x86_64 on x86_64

Compiler: GNU C++ 12.2.0 with OpenMP 4.5
C++ standard: C++17
MPI v3.1: Open MPI v4.1.5, package: Open MPI pedmon@builds01.rc.fas.harvard.edu Distribution, ident: 4.1.5, repo rev: v4.1.5, Feb 23, 2023

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

**Serial**

```
[lammps] [pkrastev@builds01 Fort]$ lmp -in in.demo 

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
Loop time of 0.168767 on 1 procs for 100 steps with 4000 atoms

Performance: 255973.752 tau/day, 592.532 timesteps/s
99.8% CPU use with 1 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0.13874    | 0.13874    | 0.13874    |   0.0 | 82.21
Neigh   | 0.023646   | 0.023646   | 0.023646   |   0.0 | 14.01
Comm    | 0.003051   | 0.003051   | 0.003051   |   0.0 |  1.81
Output  | 2.8434e-05 | 2.8434e-05 | 2.8434e-05 |   0.0 |  0.02
Modify  | 0.0027839  | 0.0027839  | 0.0027839  |   0.0 |  1.65
Other   |            | 0.0005168  |            |       |  0.31

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
[lammps] [pkrastev@builds01 Fort]$ mpirun -np 4 lmp -in in.demo 
LAMMPS (29 Oct 2020)
  using 1 OpenMP thread(s) per MPI task
Lattice spacing in x,y,z = 1.6795962 1.6795962 1.6795962
Created orthogonal box = (0.0000000 0.0000000 0.0000000) to (16.795962 16.795962 16.795962)
  1 by 2 by 2 MPI processor grid
Created 4000 atoms
  create_atoms CPU = 0.002 seconds
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
Loop time of 0.0526894 on 4 procs for 100 steps with 4000 atoms

Performance: 819898.804 tau/day, 1897.914 timesteps/s
99.7% CPU use with 4 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0.034334   | 0.035334   | 0.036511   |   0.4 | 67.06
Neigh   | 0.006092   | 0.0061221  | 0.0061656  |   0.0 | 11.62
Comm    | 0.0090592  | 0.01032    | 0.011391   |   0.8 | 19.59
Output  | 2.6026e-05 | 2.9345e-05 | 3.4108e-05 |   0.0 |  0.06
Modify  | 0.00065987 | 0.00070476 | 0.0007444  |   0.0 |  1.34
Other   |            | 0.0001791  |            |       |  0.34

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

**Example input file `in.demo` (3d Lennard-Jones melt)**

```
[lammps] [pkrastev@builds01 Fort]$ cat in.demo 
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
### Batch jobs

**Example batch job submission script**

Below is an example batch-job submission script using the LAMMPS spack environment.

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
. <PATH TO>/spack/share/spack/setup-env.sh
spack env activate lammps

# --- Run the executable ---
srun -n $SLURM_NTASKS --mpi=pmix lmp -in in.demo
```

## References:

* [LAMMPS official website](https://www.lammps.org/index.html#gsc.tab=0)
* [LAMMPS Documentation](https://docs.lammps.org/Manual.html)