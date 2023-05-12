# Intel-MPI Spack Configuration

<img src="Images/spack-logo.svg" alt="spack-logo" width="100"/>

Here we provide instructions to set up spack to build applications with [Intel-MPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/mpi-library.html) on the FASRC Cannon cluster. Intel MPI Library is now included in the [Intel oneAPI HPC Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html#hpc-kit). 

## Intel Compiler Configuration

The first step involves setting spack to use the Intel compiler, which is provided as a software module. This follows similar procedure to that of [adding the GCC compiler](https://github.com/fasrc/User_Codes/blob/master/Documents/Software/Spack.md#compiler-configuration). 

### * Load the required software module

```bash
$ module load intel/23.0.0-fasrc01
$ which icc
/n/sw/intel-oneapi-2023/compiler/2023.0.0/linux/bin/intel64/icc
```
### * Add this Intel compiler version to the spack compilers

```bash
$ spack compiler add
```

If you run the command `spack compilers`, you will see that the following 3 compilers have been added:

```bash
$ spack compilers
...
-- dpcpp rocky8-x86_64 ------------------------------------------
dpcpp@2023.0.0

-- intel rocky8-x86_64 ------------------------------------------
intel@2021.8.0

-- oneapi rocky8-x86_64 -----------------------------------------
oneapi@2023.0.0
```
### * Edit manually the compiler configuration file

Use your favorite text editor, e.g., `Vim`, `Emacs`, `VSCode`, etc., to edit the compiler configuration YAML file `~/.spack/linux/compilers.yaml`, e.g.,

```bash
$ vi ~/.spack/linux/compilers.yaml
```
Each `-compiler:` section in this file is similar to the below:

```yaml
- compiler:
    spec: intel@2021.8.0
    paths:
      cc: /n/sw/intel-oneapi-2023/compiler/2023.0.0/linux/bin/intel64/icc
      cxx: /n/sw/intel-oneapi-2023/compiler/2023.0.0/linux/bin/intel64/icpc
      f77: /n/sw/intel-oneapi-2023/compiler/2023.0.0/linux/bin/intel64/ifort
      fc: /n/sw/intel-oneapi-2023/compiler/2023.0.0/linux/bin/intel64/ifort
    flags: {}
    operating_system: rocky8
    target: x86_64
    modules: []
    environment: {}
    extra_rpaths: []
```
> **Note:** Here we focus specifically on the `intel@2021.8.0` compiler as it is required by the Intel MPI Library.

We have to edit the `modules: []` line to read

```yaml
    modules: [gcc/10.2.0-fasrc01; intel/23.0.0-fasrc01]
```

and save the `compilers.yaml` file. Please, notice that we also added a gcc module to the list, as the Intel compiler requires gcc for certain functionality. 

We can display the configuration of a specific compiler by the spack compiler info command, e.g.,

```bash
$ spack compiler info intel@2021.8.0
intel@2021.8.0:
        paths:
                cc = /n/sw/intel-oneapi-2023/compiler/2023.0.0/linux/bin/intel64/icc
                cxx = /n/sw/intel-oneapi-2023/compiler/2023.0.0/linux/bin/intel64/icpc
                f77 = /n/sw/intel-oneapi-2023/compiler/2023.0.0/linux/bin/intel64/ifort
                fc = /n/sw/intel-oneapi-2023/compiler/2023.0.0/linux/bin/intel64/ifort
        modules  = ['gcc/10.2.0-fasrc01', 'intel/23.0.0-fasrc01']
        operating system  = rocky8
```

## Setting up the Intel MPI Library

Use your favorite text editor, e.g., `Vim`, `Emacs`, `VSCode`, etc., to edit the packages configuration YAML file `~/.spack/packages.yaml`, e.g.,

```bash
$ vi ~/.spack/packages.yaml
```

> **Note:** If the file `~/.spack/packages.yaml` does not exist, you will need to create it.

Include the following contents:

```yaml
packages:
  intel-oneapi-mpi:
    externals:
    - spec: intel-oneapi-mpi@2021.8.0%intel@2021.8.0
      prefix: /n/sw/intel-oneapi-2023
    buildable: false
```
## Example

Once `spack` is configured to use Intel MPI, it can be used to build packages with it. The below example shows how to install HDF5 version 1.12.2 with `intel@2021.8.0` and `intel-oneapi-mpi@2021.8.0`.

You can first test this using the `spack spec` command to show how the spec is concretized:

```bash
$ spack spec hdf5@1.13.2%intel@2021.8.0+mpi+fortran+cxx+hl+threadsafe ^ intel-oneapi-mpi@2021.8.0%intel@2021.8.0
Input spec
--------------------------------
hdf5@1.13.2%intel@2021.8.0+cxx+fortran+hl+mpi+threadsafe
    ^intel-oneapi-mpi@2021.8.0%intel@2021.8.0

Concretized
--------------------------------
hdf5@1.13.2%intel@2021.8.0+cxx+fortran+hl~ipo~java+mpi+shared~szip+threadsafe+tools api=default build_system=cmake build_type=RelWithDebInfo arch=linux-rocky8-icelake
    ^cmake@3.24.3%intel@2021.8.0~doc+ncurses+ownlibs~qt build_system=generic build_type=Release arch=linux-rocky8-icelake
        ^ncurses@6.3%intel@2021.8.0~symlinks+termlib abi=none build_system=autotools arch=linux-rocky8-icelake
        ^openssl@1.1.1s%intel@2021.8.0~docs~shared build_system=generic certs=mozilla arch=linux-rocky8-icelake
            ^ca-certificates-mozilla@2022-10-11%intel@2021.8.0 build_system=generic arch=linux-rocky8-icelake
            ^perl@5.36.0%intel@2021.8.0+cpanm+shared+threads build_system=generic arch=linux-rocky8-icelake
                ^berkeley-db@18.1.40%intel@2021.8.0+cxx~docs+stl build_system=autotools patches=26090f4,b231fcc arch=linux-rocky8-icelake
                ^bzip2@1.0.8%intel@2021.8.0~debug~pic+shared build_system=generic arch=linux-rocky8-icelake
                    ^diffutils@3.8%intel@2021.8.0 build_system=autotools arch=linux-rocky8-icelake
                        ^libiconv@1.16%intel@2021.8.0 build_system=autotools libs=shared,static arch=linux-rocky8-icelake
                ^gdbm@1.23%intel@2021.8.0 build_system=autotools arch=linux-rocky8-icelake
                    ^readline@8.1.2%intel@2021.8.0 build_system=autotools arch=linux-rocky8-icelake
    ^intel-oneapi-mpi@2021.8.0%intel@2021.8.0 cflags="-gcc-name=/usr/bin/gcc" ~external-libfabric~generic-names~ilp64 build_system=generic arch=linux-rocky8-icelake
    ^pkgconf@1.8.0%intel@2021.8.0 build_system=autotools arch=linux-rocky8-icelake
    ^zlib@1.2.13%intel@2021.8.0+optimize+pic+shared build_system=makefile arch=linux-rocky8-icelak
```

Next, you can build it:

```bash
$ spack install hdf5@1.13.2%intel@2021.8.0+mpi+fortran+cxx+hl+threadsafe ^ intel-oneapi-mpi@2021.8.0%intel@2021.8.0
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/pkgconf-1.8.0-p4zr5nbn2qfnm2ezj3yuna7z7ldxugfo
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/ca-certificates-mozilla-2022-10-11-rlli5fmtpjaatq4lseetdudmhhcilc43
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/berkeley-db-18.1.40-m6r6onkubysvl3gxtw7t7tpghoq6x6g6
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/libiconv-1.16-iidmiswunjmyfgqv5im7dbq3vxmfugvn
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/zlib-1.2.13-admgboe6z4bq56lnw3owqrwqydoh2qkv
[+] /n/sw/intel-oneapi-2023 (external intel-oneapi-mpi-2021.8.0-xfvjrn2fuyum7xtnyz7g3gjv3c5d27tf)
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/ncurses-6.3-wq3qyyg5nbr3r4rbpxtm2w6sz2asqwmf
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/diffutils-3.8-idtn6tigrq7qhdqlofjawd5jax6ugen6
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/readline-8.1.2-lu2zjeb2y7y7k6trks5bkgswzc5mhz6h
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/bzip2-1.0.8-q4kqkohouj2iwivc2kav2imqmkkime3k
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/gdbm-1.23-h5r23zyhy5plttfjopj6hfn5nxg3526q
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/perl-5.36.0-moqx3wcrd3zgpgm4q27phfuxoymdhnkg
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/openssl-1.1.1s-rjscbgskagds5vtgqfis7ynma42o2zn5
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/cmake-3.24.3-zj2opbmjgzlp6x3ww5t2m2s6a5fko6sh
==> Installing hdf5-1.13.2-2yyrm7ssqifdsgwjthxbmp3az3pitkxj
==> No binary for hdf5-1.13.2-2yyrm7ssqifdsgwjthxbmp3az3pitkxj found: installing from source
==> Fetching https://mirror.spack.io/_source-cache/archive/01/01643fa5b37dba7be7c4db6bbf3c5d07adf5c1fa17dbfaaa632a279b1b2f06da.tar.gz
==> Ran patch() for hdf5
==> hdf5: Executing phase: 'cmake'
==> hdf5: Executing phase: 'build'
==> hdf5: Executing phase: 'install'
==> hdf5: Successfully installed hdf5-1.13.2-2yyrm7ssqifdsgwjthxbmp3az3pitkxj
  Fetch: 1.49s.  Build: 1m 58.82s.  Total: 2m 0.32s.
[+] /builds/pkrastev/Spack/spack/opt/spack/linux-rocky8-icelake/intel-2021.8.0/hdf5-1.13.2-2yyrm7ssqifdsgwjthxbmp3az3pitkxj
```