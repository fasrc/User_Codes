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

## Spack Packages

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

## Installing and Uninstalling Packages

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

NOTE: Document in progress ...






