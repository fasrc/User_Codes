# Spack

<img src="Images/spack-logo.svg" alt="MXNet-logo" width="200"/>

## What is Spack?

[Spack](https://spack.io) is a package management tool designed to support multiple versions and configurations of software on a wide variety of platforms and environments. It was designed for large supercomputer centers, where many users and application teams share common installations of software on clusters with exotic architectures, using non-standard libraries. Spack is non-destructive: installing a new version does not break existing installations. In this way several configurations can coexist on the same system.

Most importantly, Spack is simple. It offers a simple spec syntax so that users can specify versions and configuration options concisely. Spack is also simple for package authors: package files are written in pure Python, and specs allow package authors to maintain a single file for many different builds of the same package.

These instructions are intended to guide you on how to use Spack on the FAS RC Cannon cluster.

## Install and Setup

Spack works out of the box. Simply clone Spack to get going. In this example, we will clone Spack and check out the most recent release, v0.18. 
> **Note:** Spack can be installed in your home or lab space. For best performance and efficiency, we recommend to install Spack in your lab directory, e.g., <code>/n/holystore01/LABS/<PI_LAB>/Lab/spack</code>

```bash
$ git clone -c feature.manyFiles=true https://github.com/spack/spack.git
Cloning into 'spack'...
remote: Enumerating objects: 400786, done.
remote: Counting objects: 100% (30/30), done.
remote: Compressing objects: 100% (25/25), done.
remote: Total 400786 (delta 9), reused 3 (delta 0), pack-reused 400756
Receiving objects: 100% (400786/400786), 201.93 MiB | 42.75 MiB/s, done.
Resolving deltas: 100% (160994/160994), done.
Checking out files: 100% (9971/9971), done.
```

This will create the <code>spack</code> folder in the current directory. Next, we go to this directory and check out the most recent release.

```bash
$ cd spack/
$ git checkout releases/v0.18
Checking out files: 100% (7823/7823), done.
Branch releases/v0.18 set up to track remote branch releases/v0.18 from origin.
Switched to a new branch 'releases/v0.18'
```

Next, add Spack to your path. Spack has some nice command-line integration tools, so instead of simply appending to your <code>PATH</code> variable, source the Spack setup script.

```bash
$ . share/spack/setup-env.sh
$ spack --version
0.18.1
```

## Available Spack Packages

The <code>spack list</code> displays the available packages, e.g.,

```bash
$ spack list
==> 6416 packages.
<omitted output>
```
The <code>spack list</code> command can also take a query string. Spack automatically adds wildcards to both ends of the string, or you can add your own wildcards. For example, we can view all available <code>Python</code> packages.

```bash
$ spack list 'py-*'
==> 1821 packages.
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
A complete list of all available Spack packages can be found also [here](https://spack.readthedocs.io/en/latest/package_list.html).

## Installing Packages

Installing packages with Spack is very straightforward. To install a package simply type <code>spack install PACKAGE_NAME</code>, e.g.,

```bash
$ spack install bzip2@1.0.8
[+] /home/spack/opt/spack/linux-centos7-haswell/gcc-4.8.5/libiconv-1.16-r5qnpa6ilohigvm4tpqlagaq5hlsyqoh
[+] /home/spack/opt/spack/linux-centos7-haswell/gcc-4.8.5/diffutils-3.8-z33r5zkzw4hwz4ew6imoiv5zj3d542bn
==> Installing bzip2-1.0.8-uge7nkh65buipgcflh3x5lezlj64viy4
==> No binary for bzip2-1.0.8-uge7nkh65buipgcflh3x5lezlj64viy4 found: installing from source
==> Fetching https://mirror.spack.io/_source-cache/archive/ab/ab5a03176ee106d3f0fa90e381da478ddae405918153cca248e682cd0c4a2269.tar.gz
==> Ran patch() for bzip2
==> bzip2: Executing phase: 'install'
==> bzip2: Successfully installed bzip2-1.0.8-uge7nkh65buipgcflh3x5lezlj64viy4
  Fetch: 0.22s.  Build: 2.64s.  Total: 2.85s.
[+] /home/spack/opt/spack/linux-centos7-haswell/gcc-4.8.5/bzip2-1.0.8-uge7nkh65buipgcflh3x5lezlj64viy4
```
Here we installed a specific version (1.0.8)  of <code>bzip2</code>. The installed packages can be displayed by the command <code>spack find</code>:

```bash
$ spack find
==> 3 installed packages
-- linux-centos7-haswell / gcc@4.8.5 ----------------------------
bzip2@1.0.8  diffutils@3.8  libiconv@1.16
```

One can also request that Spack uses a specific compiler flavor / version to install packages, e.g.,

```bash
$ spack install bzip2@1.0.8%gcc@4.4.7
==> Installing libiconv-1.16-v34omwnfau3x6hkgcn3sswtagb64myrp
==> No binary for libiconv-1.16-v34omwnfau3x6hkgcn3sswtagb64myrp found: installing from source
==> Using cached archive: /home/spack/var/spack/cache/_source-cache/archive/e6/e6a1b1b589654277ee790cce3734f07876ac4ccfaecbee8afa0b649cf529cc04.tar.gz
==> No patches needed for libiconv
==> libiconv: Executing phase: 'autoreconf'
==> libiconv: Executing phase: 'configure'
==> libiconv: Executing phase: 'build'
==> libiconv: Executing phase: 'install'
==> libiconv: Successfully installed libiconv-1.16-v34omwnfau3x6hkgcn3sswtagb64myrp
  Fetch: 0.04s.  Build: 33.75s.  Total: 33.78s.
[+] /home/spack/opt/spack/linux-centos7-core2/gcc-4.4.7/libiconv-1.16-v34omwnfau3x6hkgcn3sswtagb64myrp
==> Installing diffutils-3.8-aw44y2twgwictbq32lcslokntw3tpjg2
==> No binary for diffutils-3.8-aw44y2twgwictbq32lcslokntw3tpjg2 found: installing from source
==> Using cached archive: /home/spack/var/spack/cache/_source-cache/archive/a6/a6bdd7d1b31266d11c4f4de6c1b748d4607ab0231af5188fc2533d0ae2438fec.tar.xz
==> No patches needed for diffutils
==> diffutils: Executing phase: 'autoreconf'
==> diffutils: Executing phase: 'configure'
==> diffutils: Executing phase: 'build'
==> diffutils: Executing phase: 'install'
==> diffutils: Successfully installed diffutils-3.8-aw44y2twgwictbq32lcslokntw3tpjg2
  Fetch: 0.02s.  Build: 49.43s.  Total: 49.45s.
[+] /home/spack/opt/spack/linux-centos7-core2/gcc-4.4.7/diffutils-3.8-aw44y2twgwictbq32lcslokntw3tpjg2
==> Installing bzip2-1.0.8-33mscorpzegrhdoj7mexrayyh2gnwjcl
==> No binary for bzip2-1.0.8-33mscorpzegrhdoj7mexrayyh2gnwjcl found: installing from source
==> Using cached archive: /home/spack/var/spack/cache/_source-cache/archive/ab/ab5a03176ee106d3f0fa90e381da478ddae405918153cca248e682cd0c4a2269.tar.gz
==> Ran patch() for bzip2
==> bzip2: Executing phase: 'install'
==> bzip2: Successfully installed bzip2-1.0.8-33mscorpzegrhdoj7mexrayyh2gnwjcl
  Fetch: 0.01s.  Build: 2.35s.  Total: 2.36s.
[+] /home/spack/opt/spack/linux-centos7-core2/gcc-4.4.7/bzip2-1.0.8-33mscorpzegrhdoj7mexrayyh2gnwjcl
```

Note that this installation is located separately from the previous one. We will discuss this in more detail later, but this is part of what allows Spack to support arbitrarily versioned software.

To specify the desired compiler, one uses the <code>%</code> sigil.

The <code>@</code> sigil is used to specify versions, both of packages and of compilers, e.g.,

```bash
$ spack install zlib@1.2.8
$ spack install zlib@1.2.8%gcc@4.4.7
```

## Uninstalling Packages

Spack provides an easy way to uninstall packages with the <code>spack uninstall PACKAGE_NAME</code>, e.g.,

```bash
$ spack uninstall bzip2@1.0.8%gcc@4.4.7
==> The following packages will be uninstalled:

    -- linux-centos7-core2 / gcc@4.4.7 ------------------------------
    33mscor bzip2@1.0.8

==> Do you want to proceed? [y/N] y
==> Successfully uninstalled bzip2@1.0.8%gcc@4.4.7~debug~pic+shared arch=linux-centos7-core2/33mscor
```
> **Note:** The recommended way of uninstalling packages is by specifying the full package name, including the package version and compiler flavor and version used to install the package on the first place.

## Using Installed Packages

There are several different ways to use Spack packages once you have installed them. The easiest way is to use <code>spack load PACKAGE_NAME</code> to load and <code>spack unload PACKAGE_NAME</code> to unload packages, e.g.,

```bash
$ spack load bzip2
$ which bzip2
/home/spack/opt/spack/linux-centos7-haswell/gcc-4.8.5/bzip2-1.0.8-uge7nkh65buipgcflh3x5lezlj64viy4/bin/bzip2
```

The loaded packages can be listed  with <code>spack find --loaded</code>, e.g.,

```bash
$ spack find --loaded
==> 3 loaded packages
-- linux-centos7-haswell / gcc@4.8.5 ----------------------------
bzip2@1.0.8  diffutils@3.8  libiconv@1.16
```

## Compiler Configuration

Spack has the ability to build packages with multiple compilers and compiler versions. This can be particularly useful, if a package needs to be built with specific compilers and compiler versions. You can display the available compilers by the <code>spack compilers</code> command, e.g.,

```bash
$ spack compilers
==> Available compilers
-- gcc centos7-x86_64 -------------------------------------------
gcc@4.8.5  gcc@4.4.7
```

The listed compilers are system level compilers provided by the OS itself. On the cluster, we support a set of core compilers, such as GNU (GCC) compiler suit, Intel, and PGI provided on the cluster through [software modules](https://docs.rc.fas.harvard.edu/kb/modules-intro).

You can easily add additional compilers to spack by loading the appropriate software modules, running the <code>spack compiler find</code> command, and edit the <code>compilers.yaml</code> configuration file. For instance, if you need GCC version 9.3.0 you need to do the following:

* ### Load the required software module

```bash
$ module load gcc/9.3.0-fasrc01
$ which gcc
/n/helmod/apps/centos7/Core/gcc/9.3.0-fasrc01/bin/gcc
```
* ### Add this GCC compiler version to the spack compilers

```bash
$ spack compiler find
==> Added 1 new compiler to /home/.spack/linux/compilers.yaml
    gcc@9.3.0
==> Compilers are defined in the following files:
    /home/.spack/linux/compilers.yaml
```
If you run <code>spack compilers</code> again, you will see that the new compiler has been added to the compiler list and made a default (listed first), e.g.,

```bash
$ spack compilers
==> Available compilers
-- gcc centos7-x86_64 -------------------------------------------
gcc@9.3.0  gcc@4.8.5  gcc@4.4.7
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
    spec: gcc@9.3.0
    paths:
      cc: /n/helmod/apps/centos7/Core/gcc/9.3.0-fasrc01/bin/gcc
      cxx: /n/helmod/apps/centos7/Core/gcc/9.3.0-fasrc01/bin/g++
      f77: /n/helmod/apps/centos7/Core/gcc/9.3.0-fasrc01/bin/gfortran
      fc: /n/helmod/apps/centos7/Core/gcc/9.3.0-fasrc01/bin/gfortran
    flags: {}
    operating_system: centos7
    target: x86_64
    modules: []
    environment: {}
    extra_rpaths: []
```
We have to edit the <code>modules: []</code> line to read

```bash
    modules: [gcc/9.3.0-fasrc01]
```
and save the compiler config. file. If more than one modules are required by the compiler, these need to be separated by semicolon (;).

We can display the configuration of a specific compiler by the <code>spack compiler info</code> command, e.g.,

```bash
$ spack compiler info gcc@9.3.0
gcc@9.3.0:
        paths:
                cc = /n/helmod/apps/centos7/Core/gcc/9.3.0-fasrc01/bin/gcc
                cxx = /n/helmod/apps/centos7/Core/gcc/9.3.0-fasrc01/bin/g++
                f77 = /n/helmod/apps/centos7/Core/gcc/9.3.0-fasrc01/bin/gfortran
                fc = /n/helmod/apps/centos7/Core/gcc/9.3.0-fasrc01/bin/gfortran
        modules  = ['gcc/9.3.0-fasrc01']
        operating system  = centos7
```

Once the new compiler is configured, it can be used to build packages. The below example shows how to install the GNU Scientific Library (GSL) with <code>gcc@9.3.0</code>.

```bash
# Check available GSL versions
$ spack versions gsl
==> Safe versions (already checksummed):
  2.7.1  2.7  2.6  2.5  2.4  2.3  2.2.1  2.1  2.0  1.16
==> Remote versions (not yet checksummed):
  2.2  1.15  1.14  1.13  1.12  1.11  1.10  1.9  1.8  1.7  1.6  1.5  1.4  1.3  1.2  1.1.1  1.1  1.0

# Install GSL version 2.7.1 with GCC version 9.3.0
$ spack install gsl@2.7.1%gcc@9.3.0
==> Installing gsl-2.7.1-7gfqgajfeedn72qbup6rospcweeh7zln
==> No binary for gsl-2.7.1-7gfqgajfeedn72qbup6rospcweeh7zln found: installing from source
==> Fetching https://mirror.spack.io/_source-cache/archive/dc/dcb0fbd43048832b757ff9942691a8dd70026d5da0ff85601e52687f6deeb34b.tar.gz
==> No patches needed for gsl
==> gsl: Executing phase: 'autoreconf'
==> gsl: Executing phase: 'configure'
==> gsl: Executing phase: 'build'
==> gsl: Executing phase: 'install'
==> gsl: Successfully installed gsl-2.7.1-7gfqgajfeedn72qbup6rospcweeh7zln
  Fetch: 1.45s.  Build: 1m 51.07s.  Total: 1m 52.52s.
[+] /home/spack/opt/spack/linux-centos7-haswell/gcc-9.3.0/gsl-2.7.1-7gfqgajfeedn72qbup6rospcweeh7zln

# Load the installed package
$ spack load gsl@2.7.1%gcc@9.3.0

# List the loaded package
$ spack find --loaded
==> 1 loaded package
-- linux-centos7-haswell / gcc@9.3.0 ----------------------------
gsl@2.7.1
```

## MPI Configuration

Many HPC software packages work in parallel using MPI. Although <code>spack</code> has the ability to install MPI libraries from scratch, the recommended way is to configure <code>spack</code> to use  MPI already available on the cluster as software modules, instead of building its own MPI libraries. 

MPI packages are configure through the <code>packages.yaml</code> file. For instance, if we need <code>OpenMPI</code> version 4.1.3 compiled with <code>GCC</code> version 12, we could follow the below steps to add this MPI configuration:

### Determine the MPI location / prefix

```bash
$ module load gcc/12.1.0-fasrc01 openmpi/4.1.3-fasrc02
$ echo $MPI_HOME
/n/helmod/apps/centos7/Comp/gcc/12.1.0-fasrc01/openmpi/4.1.3-fasrc02
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
    - spec: openmpi@4.1.3%gcc@12.1.0
      prefix: /n/helmod/apps/centos7/Comp/gcc/12.1.0-fasrc01/openmpi/4.1.3-fasrc02
    buildable: False
```
The option <code>buildable: False</code> reassures that MPI won't be built from source. Instead, <code>spack</code> will use the MPI provided as a software module in the corresponding prefix.

Once the MPI is configured, it can be used to build packages. The below example shows how to install <code>HDF5</code> version 1.12.2  with <code>openmpi@4.1.3</code> and <code>gcc@12.1.0</code>.

```bash
$ module purge
$ spack install hdf5@1.12.2 % gcc@12.1.0 ^ openmpi@4.1.3
...
==> Installing hdf5-1.12.2-r6cyc22lecakgm6dwonskqawd5rfcvgu
==> No binary for hdf5-1.12.2-r6cyc22lecakgm6dwonskqawd5rfcvgu found: installing from source
==> Using cached archive: /home/spack/var/spack/cache/_source-cache/archive/2a/2a89af03d56ce7502dcae18232c241281ad1773561ec00c0f0e8ee2463910f14.tar.gz
==> Ran patch() for hdf5
==> hdf5: Executing phase: 'cmake'
==> hdf5: Executing phase: 'build'
==> hdf5: Executing phase: 'install'
==> hdf5: Successfully installed hdf5-1.12.2-r6cyc22lecakgm6dwonskqawd5rfcvgu
  Fetch: 0.08s.  Build: 1m 45.05s.  Total: 1m 45.13s.
[+] /home/spack/opt/spack/linux-centos7-haswell/gcc-12.1.0/hdf5-1.12.2-r6cyc22lecakgm6dwonskqawd5rfcvgu

# Load the installed package
$ spack load hdf5@1.12.2%gcc@12.1.0

# List the loaded package
$ spack find --loaded
==> 14 loaded packages
-- linux-centos7-haswell / gcc@12.1.0 ---------------------------
berkeley-db@18.1.40  bzip2@1.0.8  cmake@3.23.1  diffutils@3.8  gdbm@1.19  hdf5@1.12.2  libiconv@1.16  ncurses@6.2  openmpi@4.1.3  openssl@1.1.1o  perl@5.34.1  pkgconf@1.8.0  readline@8.1  zlib@1.2.12
```

> **Note:** Please note the command <code>module purge</code>. This is required as otherwise the build fails.

## References

* [Spack official documentation](https://spack.readthedocs.io/en/latest/index.html)
* [Spack official tutorial](https://spack-tutorial.readthedocs.io/en/latest/)