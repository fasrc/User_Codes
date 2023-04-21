# Spack

<img src="Images/spack-logo.svg" alt="spack-logo" width="200"/>

## What is Spack?

[Spack](https://spack.io) is a package management tool designed to support multiple versions and configurations of software on a wide variety of platforms and environments. It was designed for large supercomputer centers, where many users and application teams share common installations of software on clusters with exotic architectures, using non-standard libraries. Spack is non-destructive: installing a new version does not break existing installations. In this way several configurations can coexist on the same system.

Most importantly, Spack is simple. It offers a simple spec syntax so that users can specify versions and configuration options concisely. Spack is also simple for package authors: package files are written in pure Python, and specs allow package authors to maintain a single file for many different builds of the same package.

These instructions are intended to guide you on how to use Spack on the FAS RC Cannon cluster.

## Install and Setup

Spack works out of the box. Simply clone Spack to get going. In this example, we will clone Spack and check out the one of the releases, v0.19. 
> **Note:** <code>Spack</code> can be installed in your home or lab space. For best performance and efficiency, we recommend to install Spack in your lab directory, e.g., <code>/n/holylabs/LABS/<PI_LAB>/Lab/software</code>/

```bash
$ git clone -c feature.manyFiles=true https://github.com/spack/spack.git
Cloning into 'spack'...
remote: Enumerating objects: 439346, done.
remote: Counting objects: 100% (144/144), done.
remote: Compressing objects: 100% (99/99), done.
remote: Total 439346 (delta 46), reused 113 (delta 31), pack-reused 439202
Receiving objects: 100% (439346/439346), 226.46 MiB | 16.95 MiB/s, done.
Resolving deltas: 100% (179280/179280), done.
```

This will create the <code>spack</code> folder in the current directory. Next, we go to this directory and check out the v0.19 release.

```bash
$ cd spack/
$ git checkout releases/v0.19
Branch 'releases/v0.19' set up to track remote branch 'releases/v0.19' from 'origin'.
Switched to a new branch 'releases/v0.19'
```

Next, add Spack to your path. Spack has some nice command-line integration tools, so instead of simply appending to your <code>PATH</code> variable, source the Spack setup script.

```bash
$ . share/spack/setup-env.sh
$ spack --version
0.19.1 (5e0d2107348eed6cbe6deca43a30f5b06c5e40af)
```

## Available Spack Packages

A complete list of all available Spack packages can be found also [here](https://spack.readthedocs.io/en/latest/package_list.html).
The <code>spack list</code> displays the available packages, e.g.,

```bash
$ spack list
==> 6752 packages
<omitted output>
```
The <code>spack list</code> command can also take a query string. Spack automatically adds wildcards to both ends of the string, or you can add your own wildcards. For example, we can view all available <code>Python</code> packages.

```bash
# with wildcard at both ends of the strings
$ spack list py
==> 1979 packages
<omitted outout>

# add your own wilcard: here, list packages that start with py
$ spack list 'py-*'
==> 1960 packages.
<omitted output>
```

You can also look for specific packages, e.g.,

```bash
$ spack list lammps
==> 1 packages.
lammps
```

You can display available software versions, e.g.,

```bash
$ spack versions lammps
==> Safe versions (already checksummed):
  master    20211214  20210929.2  20210929  20210831  20210728  20210514  20210310  20200721  20200505  20200227  20200204  20200109  20191030  20190807  20181212  20181127  20181109  20181010  20180905  20180822  20180316  20170922
  20220107  20211027  20210929.1  20210920  20210730  20210702  20210408  20201029  20200630  20200303  20200218  20200124  20191120  20190919  20190605  20181207  20181115  20181024  20180918  20180831  20180629  20180222  20170901
==> Remote versions (not yet checksummed):
  1Sep2017
```

**Note**: for the `spack versions` command, the package name needs to match exactly. For example, `spack versions lamm` will not be found:

```bash
$ spack versions lamm
==> Error: Package 'lamm' not found.
You may need to run 'spack clean -m'.
```

## Installing Packages

Installing packages with Spack is very straightforward. To install a package simply type <code>spack install PACKAGE_NAME</code>. Large packages with multiple dependencies can take significant time to install, thus we recommend doing this in a screen/tmux session or a Open Ondemand Remote Desktop session.

To install the latest version of a package, type:

```bash
$ spack install bzip2
```

To install a specific version (1.0.8) of <code>bzip2</code>, add `@` and the version number you need:

```bash
$ spack install bzip2@1.0.8
==> Bootstrapping clingo from pre-built binaries
==> Fetching https://mirror.spack.io/bootstrap/github-actions/v0.4/build_cache/linux-centos7-x86_64-gcc-10.2.1-clingo-bootstrap-spack-prqkzynv2nwko5mktitebgkeumuxkveu.spec.json
==> Fetching https://mirror.spack.io/bootstrap/github-actions/v0.4/build_cache/linux-centos7-x86_64/gcc-10.2.1/clingo-bootstrap-spack/linux-centos7-x86_64-gcc-10.2.1-clingo-bootstrap-spack-prqkzynv2nwko5mktitebgkeumuxkveu.spack
==> Installing "clingo-bootstrap@spack%gcc@10.2.1~docs~ipo+python+static_libstdcpp build_type=Release arch=linux-centos7-x86_64" from a buildcache
==> Installing libiconv-1.16-rc3o6ckaij6pgxu5444faznhssp4gcia
==> No binary for libiconv-1.16-rc3o6ckaij6pgxu5444faznhssp4gcia found: installing from source
==> Fetching https://mirror.spack.io/_source-cache/archive/e6/e6a1b1b589654277ee790cce3734f07876ac4ccfaecbee8afa0b649cf529cc04.tar.gz
==> No patches needed for libiconv
==> libiconv: Executing phase: 'autoreconf'
==> libiconv: Executing phase: 'configure'
==> libiconv: Executing phase: 'build'
==> libiconv: Executing phase: 'install'
==> libiconv: Successfully installed libiconv-1.16-rc3o6ckaij6pgxu5444faznhssp4gcia
  Fetch: 0.20s.  Build: 32.76s.  Total: 32.96s.
[+] /home/spack/opt/spack/linux-rocky8-icelake/gcc-8.5.0/libiconv-1.16-rc3o6ckaij6pgxu5444faznhssp4gcia
==> Installing diffutils-3.8-ejut7cm752b57stai5g6f7nsmte4jvps
==> No binary for diffutils-3.8-ejut7cm752b57stai5g6f7nsmte4jvps found: installing from source
==> Fetching https://mirror.spack.io/_source-cache/archive/a6/a6bdd7d1b31266d11c4f4de6c1b748d4607ab0231af5188fc2533d0ae2438fec.tar.xz
==> No patches needed for diffutils
==> diffutils: Executing phase: 'autoreconf'
==> diffutils: Executing phase: 'configure'
==> diffutils: Executing phase: 'build'
==> diffutils: Executing phase: 'install'
==> diffutils: Successfully installed diffutils-3.8-ejut7cm752b57stai5g6f7nsmte4jvps
  Fetch: 0.17s.  Build: 46.04s.  Total: 46.21s.
[+] /home/spack/opt/spack/linux-rocky8-icelake/gcc-8.5.0/diffutils-3.8-ejut7cm752b57stai5g6f7nsmte4jvps
==> Installing bzip2-1.0.8-aohgpu7zn62kzpanpohuevbkufypbnff
==> No binary for bzip2-1.0.8-aohgpu7zn62kzpanpohuevbkufypbnff found: installing from source
==> Fetching https://mirror.spack.io/_source-cache/archive/ab/ab5a03176ee106d3f0fa90e381da478ddae405918153cca248e682cd0c4a2269.tar.gz
==> Ran patch() for bzip2
==> bzip2: Executing phase: 'install'
==> bzip2: Successfully installed bzip2-1.0.8-aohgpu7zn62kzpanpohuevbkufypbnff
  Fetch: 0.23s.  Build: 2.07s.  Total: 2.30s.
[+] /home/spack/opt/spack/linux-rocky8-icelake/gcc-8.5.0/bzip2-1.0.8-aohgpu7zn62kzpanpohuevbkufypbnff
```
Here we installed a specific version (1.0.8)  of <code>bzip2</code>. The installed packages can be displayed by the command <code>spack find</code>:

```bash
$ spack find
-- linux-rocky8-icelake / gcc@8.5.0 -----------------------------
bzip2@1.0.8  diffutils@3.8  libiconv@1.16
==> 3 installed packages
```

One can also request that Spack uses a specific compiler flavor / version to install packages, e.g.,

```bash
$ spack install zlib@1.2.13%gcc@8.5.0
==> Installing zlib-1.2.13-xlt7jpku4zv2d4jhrr3azbz2vnktzfeb
==> No binary for zlib-1.2.13-xlt7jpku4zv2d4jhrr3azbz2vnktzfeb found: installing from source
==> Fetching https://mirror.spack.io/_source-cache/archive/b3/b3a24de97a8fdbc835b9833169501030b8977031bcb54b3b3ac13740f846ab30.tar.gz
==> No patches needed for zlib
==> zlib: Executing phase: 'edit'
==> zlib: Executing phase: 'build'
==> zlib: Executing phase: 'install'
==> zlib: Successfully installed zlib-1.2.13-xlt7jpku4zv2d4jhrr3azbz2vnktzfeb
  Fetch: 0.46s.  Build: 1.87s.  Total: 2.33s.
[+] /home/spack/opt/spack/linux-rocky8-icelake/gcc-8.5.0/zlib-1.2.13-xlt7jpku4zv2d4jhrr3azbz2vnktzfeb
```

To specify the desired compiler, one uses the <code>%</code> sigil.

The <code>@</code> sigil is used to specify versions, both of packages and of compilers, e.g.,

```bash
$ spack install zlib@1.2.8
$ spack install zlib@1.2.8%gcc@8.5.0
```

### Finding External Packages
Spack will normally built its own package stack, even if there are libaries available as part of the operating system. If you want Spack to build against system libraries instead of building its own you will need to have it [discover](https://www.amd.com/en/developer/spack/build-customization.html) what libraries available natively on the system. You can do this using the <code>spack external find</code>.

```bash
$ spack external find
==> The following specs have been detected on this system and added to /n/home/jharvard/.spack/packages.yaml
autoconf@2.69    binutils@2.30.117  curl@7.61.1    findutils@4.6.0  git@2.31.1   groff@1.22.3   m4@1.4.18      openssl@1.1.1k  tar@1.30
automake@1.16.1  coreutils@8.30     diffutils@3.6  gawk@4.2.1       gmake@4.2.1  libtool@2.4.6  openssh@8.0p1  pkgconf@1.4.2   texinfo@6.5
```

This even works with modules loaded from other package managers.  You simply have to have those loaded prior to running the [find](https://spack.readthedocs.io/en/latest/build_settings.html#automatically-find-external-packages) command. After these have been added to Spack, Spack will try to use them if it can in future builds rather than installing its own versions.

## Uninstalling Packages

Spack provides an easy way to uninstall packages with the <code>spack uninstall PACKAGE_NAME</code>, e.g.,

```bash
$ spack uninstall zlib@1.2.13%gcc@8.5.0
==> The following packages will be uninstalled:

    -- linux-rocky8-icelake / gcc@8.5.0 -----------------------------
    xlt7jpk zlib@1.2.13

==> Do you want to proceed? [y/N] y
==> Successfully uninstalled zlib@1.2.13%gcc@8.5.0+optimize+pic+shared build_system=makefile arch=linux-rocky8-icelake/xlt7jpk
```
> **Note:** The recommended way of uninstalling packages is by specifying the full package name, including the package version and compiler flavor and version used to install the package on the first place.

## Using Installed Packages

There are several different ways to use Spack packages once you have installed them. The easiest way is to use <code>spack load PACKAGE_NAME</code> to load and <code>spack unload PACKAGE_NAME</code> to unload packages, e.g.,

```bash
$ spack load bzip2
$ which bzip2
/home/spack/opt/spack/linux-rocky8-icelake/gcc-8.5.0/bzip2-1.0.8-aohgpu7zn62kzpanpohuevbkufypbnff/bin/bzip2
```

The loaded packages can be listed  with <code>spack find --loaded</code>, e.g.,

```bash
$ spack find --loaded
-- linux-rocky8-icelake / gcc@8.5.0 -----------------------------
bzip2@1.0.8  diffutils@3.8  libiconv@1.16
==> 3 loaded packages
```

If you no longer need the loaded packages, you can unload them with:

```bash
$ spack unload 
[pkrastev@builds01 spack]$ spack find --loaded
==> 0 loaded packages
```

## Compiler Configuration

Spack has the ability to build packages with multiple compilers and compiler versions. This can be particularly useful, if a package needs to be built with specific compilers and compiler versions. You can display the available compilers by the <code>spack compilers</code> command, e.g.,

```bash
$ spack compilers
==> Available compilers
-- gcc rocky8-x86_64 --------------------------------------------
gcc@8.5.0
```

The listed compilers are system level compilers provided by the OS itself. On the cluster, we support a set of core compilers, such as GNU (GCC) compiler suit, Intel, and PGI provided on the cluster through [software modules](https://docs.rc.fas.harvard.edu/kb/modules-intro).

You can easily add additional compilers to spack by loading the appropriate software modules, running the <code>spack compiler find</code> command, and edit the <code>compilers.yaml</code> configuration file. For instance, if you need GCC version 9.3.0 you need to do the following:

* ### Load the required software module

```bash
$ module load gcc/12.2.0-fasrc01
$ which gcc
/n/sw/helmod-rocky8/apps/Core/gcc/12.2.0-fasrc01/bin/gcc
```
* ### Add this GCC compiler version to the spack compilers

```bash
$ spack compiler find
==> Added 1 new compiler to ~/.spack/linux/compilers.yaml
    gcc@12.2.0
==> Compilers are defined in the following files:
    ~/.spack/linux/compilers.yaml
```
If you run <code>spack compilers</code> again, you will see that the new compiler has been added to the compiler list and made a default (listed first), e.g.,

```bash
$ spack compilers
==> Available compilers
-- gcc rocky8-x86_64 --------------------------------------------
gcc@12.2.0  gcc@8.5.0
```

> **Note:** By default, spack does not fill in the <code>modules:</code> field in the <code>compilers.yaml</code> file. If you are using a compiler from a module, then you should add this field manually.

* ### Edit manually the compiler configuration file

Use your favorite text editor, e.g., <code>Vim</code>, <code>Emacs</code>,<code>VSCode</code>, etc., to edit the compiler configuration YAML file <code>~/.spack/linux/compilers.yaml</code>, e.g.,

```bash
vi ~/.spack/linux/compilers.yaml
```
Each <code>-compiler:</code> section in this file is similar to the below:

```bash
- compiler:
    spec: gcc@12.2.0
    paths:
      cc: /n/sw/helmod-rocky8/apps/Core/gcc/12.2.0-fasrc01/bin/gcc
      cxx: /n/sw/helmod-rocky8/apps/Core/gcc/12.2.0-fasrc01/bin/g++
      f77: /n/sw/helmod-rocky8/apps/Core/gcc/12.2.0-fasrc01/bin/gfortran
      fc: /n/sw/helmod-rocky8/apps/Core/gcc/12.2.0-fasrc01/bin/gfortran
    flags: {}
    operating_system: rocky8
    target: x86_64
    modules: []
    environment: {}
    extra_rpaths: []
```
We have to edit the <code>modules: []</code> line to read

```bash
    modules: [gcc/12.2.0-fasrc01]
```
and save the compiler config. file. If more than one modules are required by the compiler, these need to be separated by semicolon (;).

We can display the configuration of a specific compiler by the <code>spack compiler info</code> command, e.g.,

```bash
]$ spack compiler info gcc@12.2.0
gcc@12.2.0:
	paths:
		cc = /n/sw/helmod-rocky8/apps/Core/gcc/12.2.0-fasrc01/bin/gcc
		cxx = /n/sw/helmod-rocky8/apps/Core/gcc/12.2.0-fasrc01/bin/g++
		f77 = /n/sw/helmod-rocky8/apps/Core/gcc/12.2.0-fasrc01/bin/gfortran
		fc = /n/sw/helmod-rocky8/apps/Core/gcc/12.2.0-fasrc01/bin/gfortran
	modules  = ['gcc/12.2.0-fasrc01']
	operating system  = rocky8
```

Once the new compiler is configured, it can be used to build packages. The below example shows how to install the GNU Scientific Library (GSL) with <code>gcc@9.3.0</code>.

```bash
# Check available GSL versions
$ spack versions gsl
==> Safe versions (already checksummed):
  2.7.1  2.7  2.6  2.5  2.4  2.3  2.2.1  2.1  2.0  1.16
==> Remote versions (not yet checksummed):
  2.2  1.15  1.14  1.13  1.12  1.11  1.10  1.9  1.8  1.7  1.6  1.5  1.4  1.3  1.2  1.1.1  1.1  1.0

# Install GSL version 2.7.1 with GCC version 12.2.0
$ spack install gsl@2.7.1%gcc@12.2.0
==> Installing gsl-2.7.1-uj6i6eqdsymvgupsqulhgewhb7nkr2vc
==> No binary for gsl-2.7.1-uj6i6eqdsymvgupsqulhgewhb7nkr2vc found: installing from source
==> Fetching https://mirror.spack.io/_source-cache/archive/dc/dcb0fbd43048832b757ff9942691a8dd70026d5da0ff85601e52687f6deeb34b.tar.gz
==> No patches needed for gsl
==> gsl: Executing phase: 'autoreconf'
==> gsl: Executing phase: 'configure'
==> gsl: Executing phase: 'build'
==> gsl: Executing phase: 'install'
==> gsl: Successfully installed gsl-2.7.1-uj6i6eqdsymvgupsqulhgewhb7nkr2vc
  Fetch: 0.93s.  Build: 1m 38.29s.  Total: 1m 39.22s.
[+] /home/spack/opt/spack/linux-rocky8-icelake/gcc-12.2.0/gsl-2.7.1-uj6i6eqdsymvgupsqulhgewhb7nkr2vc

# Load the installed package
$ spack load gsl@2.7.1%gcc@12.2.0

# List the loaded package
$ spack find --loaded
-- linux-rocky8-icelake / gcc@12.2.0 ----------------------------
gsl@2.7.1
==> 1 loaded package
```

## MPI Configuration

Many HPC software packages work in parallel using MPI. Although <code>spack</code> has the ability to install MPI libraries from scratch, the recommended way is to configure <code>spack</code> to use  MPI already available on the cluster as software modules, instead of building its own MPI libraries. 

MPI is configured through the <code>packages.yaml</code> file. For instance, if we need <code>OpenMPI</code> version 4.1.3 compiled with <code>GCC</code> version 12, we could follow the below steps to add this MPI configuration:

### Determine the MPI location / prefix

```bash
$ module load gcc/12.2.0-fasrc01 openmpi/4.1.4-fasrc01
$ echo $MPI_HOME
/n/sw/helmod-rocky8/apps/Comp/gcc/12.2.0-fasrc01/openmpi/4.1.4-fasrc01
```

### Edit manually the packages configuration file

Use your favorite text editor, e.g., <code>Vim</code>, <code>Emacs</code>,<code>VSCode</code>, etc., to edit the packages configuration YAML file <code>~/.spack/packages.yaml</code>, e.g.,

```bash
$ vi ~/.spack/packages.yaml
```
> **Note:** If the file <code>~/.spack/packages.yaml</code> does not exist, you will need to create it.

Include the following contents:

```yaml
packages:
  openmpi:
    externals:
    - spec: openmpi@4.1.4%gcc@12.2.0
      prefix: /n/sw/helmod-rocky8/apps/Comp/gcc/12.2.0-fasrc01/openmpi/4.1.4-fasrc01
    buildable: False
```
The option <code>buildable: False</code> reassures that MPI won't be built from source. Instead, <code>spack</code> will use the MPI provided as a software module in the corresponding prefix.

Once the MPI is configured, it can be used to build packages. The below example shows how to install <code>HDF5</code> version 1.12.2  with <code>openmpi@4.1.4</code> and <code>gcc@12.2.0</code>.

```bash
$ module purge
$ spack install hdf5@1.12.2 % gcc@12.2.0 ^ openmpi@4.1.4
...
==> Installing hdf5-1.12.2-lfmo7dvzrgmu35mt74zqjz2mfcwa2urb
==> No binary for hdf5-1.12.2-lfmo7dvzrgmu35mt74zqjz2mfcwa2urb found: installing from source
==> Fetching https://mirror.spack.io/_source-cache/archive/2a/2a89af03d56ce7502dcae18232c241281ad1773561ec00c0f0e8ee2463910f14.tar.gz
==> Ran patch() for hdf5
==> hdf5: Executing phase: 'cmake'
==> hdf5: Executing phase: 'build'
==> hdf5: Executing phase: 'install'
==> hdf5: Successfully installed hdf5-1.12.2-lfmo7dvzrgmu35mt74zqjz2mfcwa2urb
  Fetch: 0.58s.  Build: 1m 21.39s.  Total: 1m 21.98s.
[+] /home/spack/opt/spack/linux-rocky8-icelake/gcc-12.2.0/hdf5-1.12.2-lfmo7dvzrgmu35mt74zqjz2mfcwa2urb

# Load the installed package
$ spack load hdf5@1.12.2%gcc@12.2.0

# List the loaded package
$ spack find --loaded
-- linux-rocky8-icelake / gcc@12.2.0 ----------------------------
berkeley-db@18.1.40  ca-certificates-mozilla@2022-10-11  diffutils@3.8  hdf5@1.12.2    ncurses@6.3    openssl@1.1.1s  pkgconf@1.8.0   zlib@1.2.13
bzip2@1.0.8          cmake@3.24.3                        gdbm@1.23      libiconv@1.16  openmpi@4.1.4  perl@5.36.0     readline@8.1.2
==> 15 loaded packages
```

> **Note:** Please note the command <code>module purge</code>. This is required as otherwise the build fails.

## Troubleshooting
When spack builds it uses a <code>stage</code> directory located in <code>/tmp</code>. Spack also cleans up this space once it is done building, regardless of if the build succeeds or fails. This can make troubleshooting failed builds difficult as the logs from those builds are stored in <code>stage</code>. To preserve these files for debugging you will first want to set the <code>$TMP</code> environmental variable to a location that you want to dump files in <code>stage</code> to. Then you will want to add the <code>--keep-stage</code> flag to spack (ex. <code>spack install --keep-stage <package></code>), which tells spack to keep the staging files rather than remove them.

### Cannot open shared object file: No such file or directory
```bash
2 errors found in build log:
     10    Configured with: ./configure --prefix=/n/helmod/apps/centos7/Core/gcc/10.2.0-fasrc01 --program-prefix= --exec-prefix=/n/helmod/apps/centos7/Core/gcc/10.2.0-fasrc01 --bindir=/n/helmod/apps/centos7/Core/gcc/10.2.0-fasrc01/bin 
           --sbindir=/n/helmod/apps/centos7/Core/gcc/10.2.0-fasrc01/sbin --sysconfdir=/n/helmod/apps/centos7/Core/gcc/10.2.0-fasrc01/etc --datadir=/n/helmod/apps/centos7/Core/gcc/10.2.0-fasrc01/share --includedir=/n/helmod/apps/centos7
           /Core/gcc/10.2.0-fasrc01/include --libdir=/n/helmod/apps/centos7/Core/gcc/10.2.0-fasrc01/lib64 --libexecdir=/n/helmod/apps/centos7/Core/gcc/10.2.0-fasrc01/libexec --localstatedir=/n/helmod/apps/centos7/Core/gcc/10.2.0-fasrc0
           1/var --sharedstatedir=/n/helmod/apps/centos7/Core/gcc/10.2.0-fasrc01/var/lib --mandir=/n/helmod/apps/centos7/Core/gcc/10.2.0-fasrc01/share/man --infodir=/n/helmod/apps/centos7/Core/gcc/10.2.0-fasrc01/share/info
     11    Thread model: posix
     12    Supported LTO compression algorithms: zlib
     13    gcc version 10.2.0 (GCC)
     14    COLLECT_GCC_OPTIONS='-o' '/tmp/tmp.LkhfoOt8fH/a.out' '-v' '-mtune=generic' '-march=x86-64'
     15     /n/sw/helmod/apps/centos7/Core/gcc/10.2.0-fasrc01/bin/../libexec/gcc/x86_64-pc-linux-gnu/10.2.0/cc1 -quiet -v -iprefix /n/sw/helmod/apps/centos7/Core/gcc/10.2.0-fasrc01/bin/../lib64/gcc/x86_64-pc-linux-gnu/10.2.0/ /tmp/tmp.
           LkhfoOt8fH/hello-49015.c -quiet -dumpbase hello-49015.c -mtune=generic -march=x86-64 -auxbase hello-49015 -version -o /tmp/ccVvRxDx.s
  >> 16    /n/sw/helmod/apps/centos7/Core/gcc/10.2.0-fasrc01/bin/../libexec/gcc/x86_64-pc-linux-gnu/10.2.0/cc1: error while loading shared libraries: libmpfr.so.6: cannot open shared object file: No such file or directory
     17    
     18    ERROR: Linker : not found
  >> 19     ** makelocalrc step has FAILED.  Linker not found **
     20     ** See gcc output above **
     21    Command used:
     22    /n/helmod/apps/centos7/Core/gcc/10.2.0-fasrc01/bin/gcc -o /tmp/tmp.LkhfoOt8fH/a.out -v /tmp/tmp.LkhfoOt8fH/hello-49015.c
     23    cat /tmp/tmp.LkhfoOt8fH/hello-49015.c:
     24    #include <stdio.h>
     25    int main()

See build log for details:
  /tmp/jharvard
	/spack-stage/spack-stage-nvhpc-22.7-iepk6vgndc7hmzs3evxqz6qw2vf6qt7s/spack-build-out.txt
```
In this error the compiler cannot find a library it is dependent on <code>mpfr</code>.  To fix this we will need to add the relevant library to the compiler definition in <code>~/.spack/linux/compilers.yaml</code>. In this case we are using <code>gcc/10.2.0-fasrc01</code> which when loaded also loads:

```bash
[jharvard@holy7c22501 ~]# module list

Currently Loaded Modules:
  1) gmp/6.2.1-fasrc01   2) mpfr/4.1.0-fasrc01   3) mpc/1.2.1-fasrc01   4) gcc/10.2.0-fasrc01
```

So we will need to grab the location of these libraries to add them. To find that you can do:
```bash
[jharvard@holy7c22501 ~]# module display mpfr/4.1.0-fasrc01
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
   /n/helmod/modulefiles/centos7/Core/mpfr/4.1.0-fasrc01.lua:
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
help([[mpfr-4.1.0-fasrc01
The MPFR library is a C library for multiple-precision floating-point computations with correct rounding.

]], [[
]])
whatis("Name: mpfr")
whatis("Version: 4.1.0-fasrc01")
whatis("Description: The MPFR library is a C library for multiple-precision floating-point computations with correct rounding.")
setenv("MPFR_HOME","/n/helmod/apps/centos7/Core/mpfr/4.1.0-fasrc01")
setenv("MPFR_INCLUDE","/n/helmod/apps/centos7/Core/mpfr/4.1.0-fasrc01/include")
setenv("MPFR_LIB","/n/helmod/apps/centos7/Core/mpfr/4.1.0-fasrc01/lib64")
prepend_path("CPATH","/n/helmod/apps/centos7/Core/mpfr/4.1.0-fasrc01/include")
prepend_path("FPATH","/n/helmod/apps/centos7/Core/mpfr/4.1.0-fasrc01/include")
prepend_path("INFOPATH","/n/helmod/apps/centos7/Core/mpfr/4.1.0-fasrc01/share/info")
prepend_path("LD_LIBRARY_PATH","/n/helmod/apps/centos7/Core/mpfr/4.1.0-fasrc01/lib64")
prepend_path("LIBRARY_PATH","/n/helmod/apps/centos7/Core/mpfr/4.1.0-fasrc01/lib64")
prepend_path("PKG_CONFIG_PATH","/n/helmod/apps/centos7/Core/mpfr/4.1.0-fasrc01/lib64/pkgconfig")
```
And then pull out the <code>LIBRARY_PATH</code>. Once we have the paths for all three of these dependencies we can add them to the <code>~/.spack/linux/compilers.yaml</code> as follows

```bash
- compiler:
    spec: gcc@10.2.0
    paths:
      cc: /n/helmod/apps/centos7/Core/gcc/10.2.0-fasrc01/bin/gcc
      cxx: /n/helmod/apps/centos7/Core/gcc/10.2.0-fasrc01/bin/g++
      f77: /n/helmod/apps/centos7/Core/gcc/10.2.0-fasrc01/bin/gfortran
      fc: /n/helmod/apps/centos7/Core/gcc/10.2.0-fasrc01/bin/gfortran
    flags: {}
    operating_system: centos7
    target: x86_64
    modules: []
    environment:
      prepend_path:
        LIBRARY_PATH: /n/helmod/apps/centos7/Core/mpc/1.2.1-fasrc01/lib64:/n/helmod/apps/centos7/Core/mpfr/4.1.0-fasrc01/lib64:/n/helmod/apps/centos7/Core/gmp/6.2.1-fasrc01/lib64
        LD_LIBRARY_PATH: /n/helmod/apps/centos7/Core/mpc/1.2.1-fasrc01/lib64:/n/helmod/apps/centos7/Core/mpfr/4.1.0-fasrc01/lib64:/n/helmod/apps/centos7/Core/gmp/6.2.1-fasrc01/lib64
    extra_rpaths: []
```
Namely we needed to add the <code>prepend_path</code> to the <code>environment</code>.  With those additional paths defined the compiler will now work because it can find its dependencies.

### C compiler cannont create executables
This is the same type of error as the <code>Cannot open shared object file: No such file or directory</code>. Namely the compiler cannot find the libraries it is dependent on. See the troubleshooting section for the [shared objects error](#cannot-open-shared-object-file-no-such-file-or-directory) for how to resolve.

## References

* [Spack official documentation](https://spack.readthedocs.io/en/latest/index.html)
* [Spack official tutorial](https://spack-tutorial.readthedocs.io/en/latest/)
