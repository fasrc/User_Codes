# Build GROMACS with MPI using Spack

<img src="Images/spack-logo.svg" alt="spack-logo" width="100"/>

[GROMACS](https://www.gromacs.org/) is a free and open-source software suite for high-performance molecular dynamics and output analysis.

The below instructions provide a spack recipe for building MPI capable instance of LAMMPS on the [FASRC Cannon cluster](https://www.rc.fas.harvard.edu/about/cluster-architecture/).

## Compiler and MPI Library Spack configuration

Here we will use the [GNU/GCC](https://gcc.gnu.org/) compiler suite together with [OpenMPI](https://www.open-mpi.org/).

 The below instructions assume that spack is already configured to use the GCC compiler `gcc@12.2.0` and OpenMPI Library `openmpi@4.1.5`, as explained [here](https://github.com/fasrc/User_Codes/blob/master/Documents/Software/Spack.md).

## Create GROMACS spack environment and activate it

```bash
spack env create gromacs
spack env activate -p gromacs
```

## Install the GROMACS environment

### Add the required packages to the spack environment

```bash
spack add openmpi@4.1.5
spack add gromacs@2023.3 + mpi + openmp % gcc@12.2.0 ^ openmpi@4.1.5
```

### Install the environment

Once all required packages are added to the environment, it can be installed with:

```bash
spack install
```
For example,

```bash
[gromacs] [pkrastev@builds01 Spack]$ spack install
==> Concretized gromacs@2023.3%gcc@12.2.0+mpi+openmp ^openmpi@4.1.5
 -   42ku4gz  gromacs@2023.3%gcc@12.2.0~cp2k~cuda~cycle_subcounters~double+hwloc~intel_provided_gcc~ipo~mdrun_only+mpi~nosuffix~opencl+openmp~plumed~relaxed_double_precision+shared~sycl build_system=cmake build_type=Release generator=make openmp_max_threads=none arch=linux-rocky8-x86_64
[+]  gfpb5mz      ^cmake@3.27.7%gcc@12.2.0~doc+ncurses+ownlibs build_system=generic build_type=Release arch=linux-rocky8-x86_64
[+]  yvt4hk5          ^curl@8.4.0%gcc@12.2.0~gssapi~ldap~libidn2~librtmp~libssh~libssh2+nghttp2 build_system=autotools libs=shared,static tls=openssl arch=linux-rocky8-x86_64
[+]  lvshkiv              ^nghttp2@1.57.0%gcc@12.2.0 build_system=autotools arch=linux-rocky8-x86_64
[+]  t6g7cgp              ^openssl@3.1.3%gcc@12.2.0~docs+shared build_system=generic certs=mozilla arch=linux-rocky8-x86_64
[+]  y4pl22s                  ^ca-certificates-mozilla@2023-05-30%gcc@12.2.0 build_system=generic arch=linux-rocky8-x86_64
[+]  enjvgjw          ^ncurses@6.4%gcc@12.2.0~symlinks+termlib abi=none build_system=autotools arch=linux-rocky8-x86_64
[+]  vydxoz7          ^zlib-ng@2.1.4%gcc@12.2.0+compat+opt build_system=autotools arch=linux-rocky8-x86_64
[+]  o3gyqz4      ^fftw@3.3.10%gcc@12.2.0+mpi~openmp~pfft_patches build_system=autotools precision=double,float arch=linux-rocky8-x86_64
[+]  svtpsvv      ^gmake@4.4.1%gcc@12.2.0~guile build_system=generic arch=linux-rocky8-x86_64
 -   tuxaidn      ^hwloc@2.9.1%gcc@12.2.0~cairo~cuda~gl~libudev+libxml2~netloc~nvml~oneapi-level-zero~opencl+pci~rocm build_system=autotools libs=shared,static arch=linux-rocky8-x86_64
 -   o2opvjh          ^libpciaccess@0.17%gcc@12.2.0 build_system=autotools arch=linux-rocky8-x86_64
 -   vjgd4jr              ^libtool@2.4.7%gcc@12.2.0 build_system=autotools arch=linux-rocky8-x86_64
 -   nliiavo                  ^m4@1.4.19%gcc@12.2.0+sigsegv build_system=autotools patches=9dc5fbd,bfdffa7 arch=linux-rocky8-x86_64
 -   tzc6rts                      ^libsigsegv@2.14%gcc@12.2.0 build_system=autotools arch=linux-rocky8-x86_64
 -   nizhp7h              ^util-macros@1.19.3%gcc@12.2.0 build_system=autotools arch=linux-rocky8-x86_64
[+]  vib3s6c          ^libxml2@2.10.3%gcc@12.2.0+pic~python+shared build_system=autotools arch=linux-rocky8-x86_64
[+]  4ic3bei              ^libiconv@1.17%gcc@12.2.0 build_system=autotools libs=shared,static arch=linux-rocky8-x86_64
[+]  o2frdmd              ^xz@5.4.1%gcc@12.2.0~pic build_system=autotools libs=shared,static arch=linux-rocky8-x86_64
[+]  bd3npcq          ^pkgconf@1.9.5%gcc@12.2.0 build_system=autotools arch=linux-rocky8-x86_64
[+]  5oqjy7z      ^openblas@0.3.24%gcc@12.2.0~bignuma~consistent_fpcsr+fortran~ilp64+locking+pic+shared build_system=makefile symbol_suffix=none threads=none arch=linux-rocky8-x86_64
[+]  k7dhgbu          ^perl@5.38.0%gcc@12.2.0+cpanm+opcode+open+shared+threads build_system=generic patches=714e4d1 arch=linux-rocky8-x86_64
[+]  gyedpff              ^berkeley-db@18.1.40%gcc@12.2.0+cxx~docs+stl build_system=autotools patches=26090f4,b231fcc arch=linux-rocky8-x86_64
[+]  xxvougl              ^bzip2@1.0.8%gcc@12.2.0~debug~pic+shared build_system=generic arch=linux-rocky8-x86_64
[+]  6delrmq                  ^diffutils@3.9%gcc@12.2.0 build_system=autotools arch=linux-rocky8-x86_64
[+]  r4ncx2h              ^gdbm@1.23%gcc@12.2.0 build_system=autotools arch=linux-rocky8-x86_64
[+]  gqnfuu6                  ^readline@8.2%gcc@12.2.0 build_system=autotools patches=bbf97f1 arch=linux-rocky8-x86_64
[e]  rzl24ya      ^openmpi@4.1.5%gcc@12.2.0~atomics~cuda~cxx~cxx_exceptions~gpfs~internal-hwloc~internal-pmix~java~legacylaunchers~lustre~memchecker~openshmem~orterunprefix+romio+rsh~singularity+static+vt+wrapper-rpath build_system=autotools fabrics=none schedulers=none arch=linux-rocky8-x86_64

==> Concretized openmpi@4.1.5
[e]  rzl24ya  openmpi@4.1.5%gcc@12.2.0~atomics~cuda~cxx~cxx_exceptions~gpfs~internal-hwloc~internal-pmix~java~legacylaunchers~lustre~memchecker~openshmem~orterunprefix+romio+rsh~singularity+static+vt+wrapper-rpath build_system=autotools fabrics=none schedulers=none arch=linux-rocky8-x86_64

[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/nghttp2-1.57.0-lvshkivi7b4qjhtdvew667hctxypbszb
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/zlib-ng-2.1.4-vydxoz7fvgvsutjrmly6cayjaedjyrmg
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/ncurses-6.4-enjvgjww5qpxthriw5sywb7isei43bww
[+] /n/sw/helmod-rocky8/apps/Comp/gcc/12.2.0-fasrc01/openmpi/4.1.5-fasrc03 (external openmpi-4.1.5-rzl24yaivlkwvg4g5qznqm3vv32l2fh3)
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/gmake-4.4.1-svtpsvvjlmcqaoikxzgv7tn65gm5qkd2
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/openssl-3.1.3-t6g7cgpe6j762uy7k6tywuvoehrpoxlr
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/fftw-3.3.10-o3gyqz4xr6icuwhvkij36gelql4w7zhe
==> Installing util-macros-1.19.3-nizhp7hfpxaa6b4z2ys35s4ns55ei5ek [8/22]
==> No binary for util-macros-1.19.3-nizhp7hfpxaa6b4z2ys35s4ns55ei5ek found: installing from source
==> Fetching https://mirror.spack.io/_source-cache/archive/0f/0f812e6e9d2786ba8f54b960ee563c0663ddbe2434bf24ff193f5feab1f31971.tar.bz2
==> No patches needed for util-macros
==> util-macros: Executing phase: 'autoreconf'
==> util-macros: Executing phase: 'configure'
==> util-macros: Executing phase: 'build'
==> util-macros: Executing phase: 'install'
==> util-macros: Successfully installed util-macros-1.19.3-nizhp7hfpxaa6b4z2ys35s4ns55ei5ek
  Stage: 0.28s.  Autoreconf: 0.00s.  Configure: 1.74s.  Build: 0.02s.  Install: 0.07s.  Post-install: 0.05s.  Total: 2.30s
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/util-macros-1.19.3-nizhp7hfpxaa6b4z2ys35s4ns55ei5ek
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/pkgconf-1.9.5-bd3npcqaivwm7xw6ryfdx4ngkhsqvw5y
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/libiconv-1.17-4ic3bei7fcc3od6alzc4xhoag6tovpc6
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/xz-5.4.1-o2frdmd6pdpc5shh4b7ceuh6m3r24tky
==> Installing libsigsegv-2.14-tzc6rtsvmr5l6h4gvvzbq7srr66aqzrq [12/22]
==> No binary for libsigsegv-2.14-tzc6rtsvmr5l6h4gvvzbq7srr66aqzrq found: installing from source
==> Fetching https://mirror.spack.io/_source-cache/archive/cd/cdac3941803364cf81a908499beb79c200ead60b6b5b40cad124fd1e06caa295.tar.gz
==> No patches needed for libsigsegv
==> libsigsegv: Executing phase: 'autoreconf'
==> libsigsegv: Executing phase: 'configure'
==> libsigsegv: Executing phase: 'build'
==> libsigsegv: Executing phase: 'install'
==> libsigsegv: Successfully installed libsigsegv-2.14-tzc6rtsvmr5l6h4gvvzbq7srr66aqzrq
  Stage: 0.28s.  Autoreconf: 0.00s.  Configure: 9.94s.  Build: 1.07s.  Install: 0.29s.  Post-install: 0.05s.  Total: 11.79s
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/libsigsegv-2.14-tzc6rtsvmr5l6h4gvvzbq7srr66aqzrq
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/openblas-0.3.24-5oqjy7z5brnhphx4cutohw242bfxnk7m
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/curl-8.4.0-yvt4hk57jtntsqm3n3jund52c5b6j3yf
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/diffutils-3.9-6delrmqopdzic26vrdoi6p6lbuc34upl
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/libxml2-2.10.3-vib3s6cnyl2yvtxwpneekgtl2wrbbhmr
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/cmake-3.27.7-gfpb5mzafl4j7znv6r3z7v3n65y4lmhd
==> Installing m4-1.4.19-nliiavoqywyldqe343hf4gos3qjelsuz [18/22]
==> No binary for m4-1.4.19-nliiavoqywyldqe343hf4gos3qjelsuz found: installing from source
==> Fetching https://mirror.spack.io/_source-cache/archive/3b/3be4a26d825ffdfda52a56fc43246456989a3630093cced3fbddf4771ee58a70.tar.gz
==> Applied patch /builds/pkrastev/Spack/spack/var/spack/repos/builtin/packages/m4/checks-198.sysval.1.patch
==> Applied patch /builds/pkrastev/Spack/spack/var/spack/repos/builtin/packages/m4/checks-198.sysval.2.patch
==> Ran patch() for m4
==> m4: Executing phase: 'autoreconf'
==> m4: Executing phase: 'configure'
==> m4: Executing phase: 'build'
==> m4: Executing phase: 'install'
==> m4: Successfully installed m4-1.4.19-nliiavoqywyldqe343hf4gos3qjelsuz
  Stage: 0.41s.  Autoreconf: 0.00s.  Configure: 1m 26.82s.  Build: 4.67s.  Install: 1.52s.  Post-install: 0.07s.  Total: 1m 33.64s
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/m4-1.4.19-nliiavoqywyldqe343hf4gos3qjelsuz
==> Installing libtool-2.4.7-vjgd4jric73ratfriyaikqpty34ouzaj [19/22]
==> No binary for libtool-2.4.7-vjgd4jric73ratfriyaikqpty34ouzaj found: installing from source
==> Fetching https://mirror.spack.io/_source-cache/archive/04/04e96c2404ea70c590c546eba4202a4e12722c640016c12b9b2f1ce3d481e9a8.tar.gz
==> Ran patch() for libtool
==> libtool: Executing phase: 'autoreconf'
==> libtool: Executing phase: 'configure'
==> libtool: Executing phase: 'build'
==> libtool: Executing phase: 'install'
==> libtool: Successfully installed libtool-2.4.7-vjgd4jric73ratfriyaikqpty34ouzaj
  Stage: 0.26s.  Autoreconf: 0.00s.  Configure: 16.30s.  Build: 2.06s.  Install: 1.72s.  Post-install: 0.06s.  Total: 20.57s
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/libtool-2.4.7-vjgd4jric73ratfriyaikqpty34ouzaj
==> Installing libpciaccess-0.17-o2opvjho6vpqgvxrmvulczijo4u2lnll [20/22]
==> No binary for libpciaccess-0.17-o2opvjho6vpqgvxrmvulczijo4u2lnll found: installing from source
==> Fetching https://mirror.spack.io/_source-cache/archive/bf/bf6985a77d2ecb00e2c79da3edfb26b909178ffca3f2e9d14ed0620259ab733b.tar.gz
==> No patches needed for libpciaccess
==> libpciaccess: Executing phase: 'autoreconf'
==> libpciaccess: Executing phase: 'configure'
==> libpciaccess: Executing phase: 'build'
==> libpciaccess: Executing phase: 'install'
==> libpciaccess: Successfully installed libpciaccess-0.17-o2opvjho6vpqgvxrmvulczijo4u2lnll
  Stage: 0.16s.  Autoreconf: 0.00s.  Configure: 10.65s.  Build: 1.39s.  Install: 0.44s.  Post-install: 0.05s.  Total: 12.86s
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/libpciaccess-0.17-o2opvjho6vpqgvxrmvulczijo4u2lnll
==> Installing hwloc-2.9.1-tuxaidnvxmy3rbetxfo5saf7da226y6n [21/22]
==> No binary for hwloc-2.9.1-tuxaidnvxmy3rbetxfo5saf7da226y6n found: installing from source
==> Fetching https://mirror.spack.io/_source-cache/archive/a4/a440e2299f7451dc10a57ddbfa3f116c2a6c4be1bb97c663edd3b9c7b3b3b4cf.tar.gz
==> No patches needed for hwloc
==> hwloc: Executing phase: 'autoreconf'
==> hwloc: Executing phase: 'configure'
==> hwloc: Executing phase: 'build'
==> hwloc: Executing phase: 'install'
==> hwloc: Successfully installed hwloc-2.9.1-tuxaidnvxmy3rbetxfo5saf7da226y6n
  Stage: 0.45s.  Autoreconf: 0.00s.  Configure: 31.37s.  Build: 7.53s.  Install: 1.82s.  Post-install: 0.14s.  Total: 41.50s
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/hwloc-2.9.1-tuxaidnvxmy3rbetxfo5saf7da226y6n
==> Installing gromacs-2023.3-42ku4gzzitbmzoy4zq43o3ozwr5el3tx [22/22]
==> No binary for gromacs-2023.3-42ku4gzzitbmzoy4zq43o3ozwr5el3tx found: installing from source
==> Fetching https://mirror.spack.io/_source-cache/archive/4e/4ec8f8d0c7af76b13f8fd16db8e2c120e749de439ae9554d9f653f812d78d1cb.tar.gz
==> Ran patch() for gromacs
==> gromacs: Executing phase: 'cmake'
==> gromacs: Executing phase: 'build'
==> gromacs: Executing phase: 'install'
==> gromacs: Successfully installed gromacs-2023.3-42ku4gzzitbmzoy4zq43o3ozwr5el3tx
  Stage: 1.40s.  Cmake: 27.57s.  Build: 1m 34.31s.  Install: 1.54s.  Post-install: 0.20s.  Total: 2m 5.23s
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/gromacs-2023.3-42ku4gzzitbmzoy4zq43o3ozwr5el3tx
==> Updating view at /builds/pkrastev/Spack/spack/var/spack/environments/gromacs/.spack-env/view
==> Warning: Skipping external package: openmpi@4.1.5%gcc@12.2.0~atomics~cuda~cxx~cxx_exceptions~gpfs~internal-hwloc~internal-pmix~java~legacylaunchers~lustre~memchecker~openshmem~orterunprefix+romio+rsh~singularity+static+vt+wrapper-rpath build_system=autotools fabrics=none schedulers=none arch=linux-rocky8-x86_64/rzl24ya
```

### List the packages installed in the GROMACS environment

```bash
[gromacs] [pkrastev@builds01 Spack]$ spack find
==> In environment gromacs
==> Root specs
openmpi@4.1.5 

-- no arch / gcc@12.2.0 -----------------------------------------
gromacs@2023.3%gcc@12.2.0 +mpi+openmp

==> Installed packages
-- linux-rocky8-x86_64 / gcc@12.2.0 -----------------------------
berkeley-db@18.1.40                 cmake@3.27.7   fftw@3.3.10  gromacs@2023.3  libpciaccess@0.17  libxml2@2.10.3  nghttp2@1.57.0   openssl@3.1.3  readline@8.2        zlib-ng@2.1.4
bzip2@1.0.8                         curl@8.4.0     gdbm@1.23    hwloc@2.9.1     libsigsegv@2.14    m4@1.4.19       openblas@0.3.24  perl@5.38.0    util-macros@1.19.3
ca-certificates-mozilla@2023-05-30  diffutils@3.9  gmake@4.4.1  libiconv@1.17   libtool@2.4.7      ncurses@6.4     openmpi@4.1.5    pkgconf@1.9.5  xz@5.4.1
==> 28 installed packages
```

## Use GROMACS

Once the environment is installed, all installed packages in the GROMACS environment are available on the `PATH`, e.g.,:

```bash
[gromacs] [pkrastev@builds01 Spack]$ gmx_mpi -h
                    :-) GROMACS - gmx_mpi, 2023.3-spack (-:

Executable:   /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/gromacs-2023.3-42ku4gzzitbmzoy4zq43o3ozwr5el3tx/bin/gmx_mpi
Data prefix:  /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-x86_64/gcc-12.2.0/gromacs-2023.3-42ku4gzzitbmzoy4zq43o3ozwr5el3tx
Working dir:  /builds/pkrastev/Spack
Command line:
  gmx_mpi -h

SYNOPSIS

gmx [-[no]h] [-[no]quiet] [-[no]version] [-[no]copyright] [-nice <int>]
    [-[no]backup]

OPTIONS

Other options:

 -[no]h                     (no)
           Print help and quit
 -[no]quiet                 (no)
           Do not print common startup info or quotes
 -[no]version               (no)
           Print extended version information and quit
 -[no]copyright             (no)
           Print copyright information on startup
 -nice   <int>              (19)
           Set the nicelevel (default depends on command)
 -[no]backup                (yes)
           Write backups if output files exist

Additional help is available on the following topics:
    commands    List of available commands
    selections  Selection syntax and usage
To access the help, use 'gmx help <topic>'.
For help on a command, use 'gmx help <command>'.

GROMACS reminds you: "Predictions can be very difficult - especially about the future." (Niels Bohr)
```
### Interactive runs

You can run GROMACS interactively. This assumes you have requested an interactive session first, as explained [here](https://docs.rc.fas.harvard.edu/kb/running-jobs/#articleTOC_14).

In order to set up your GROMACS environment, you need to run the commands:

```bash
### Replace <PATH TO> with the actual path to your spack installation
. <PATH TO>/spack/share/spack/setup-env.sh
spack env activate gromacs
```

### Batch jobs

When submitting batch-jobs, you will need to add the below lines to your submission script:

```bash
# --- Activate the GROMACS Spack environment., e.g., ---
### NOTE: Replace <PATH TO> with the actual path to your spack installation
. <PATH TO>/spack/share/spack/setup-env.sh
spack env activate gromacs
```

## References:

* [GROMACS official website](https://www.gromacs.org/)
* [GROMACS Documentation](https://manual.gromacs.org/current/index.html)