<!-- 
TODO: Add training materials relevant to Rocky 8 and SingulartyCE 3.11
      The content below is too old for Rocky 8
# Online training materials

- Slides from our Aug 2022 SingularityCE training: Using [Containers on the Cannon
  Cluster -- SingularityCE
  (PDF)](https://docs.rc.fas.harvard.edu/wp-content/uploads/2022/08/Containers_on_Cannon_08_22.pdf)
- Video from this session: [Using Containers on the Cannon Cluster: SingularityCE
  (Video)](https://harvard.zoom.us/rec/share/jaYAuOMVpBelUonfjXNndHAqxhjFjprQoWZtuyfWbMPLFdEQVyV_g9AEAdU7Uo1b.xEl9EUnD_pF-fiNP)
-->

# Introduction

This page provides information about
[SingularityCE](https://sylabs.io/singularity/) containers and how to use
SingularityCE on the FASRC clusters. SingularityCE is available on the FASRC
clusters Cannon and FASSE. 

SingularityCE enables users to have full control of their operating system
environment (OS). This allows a non-privileged user (e.g. non- root, sudo,
administrator, etc.) to “swap out” the Linux operating system and environment on
the host machine (i.e., the cluster's OS) for another Linux OS and computing
environment that they can control (i.e., the container's OS). For instance, the
host system runs Rocky Linux but your application requires CentOS or Ubuntu
Linux with a specific software stack. You can create a CentOS or Ubuntu image,
install your software into that image, copy the created image to the cluster,
and run your application on that host in its native CentOS or Ubuntu
environment.

SingularityCE leverages the resources of the host system, such as high-speed
interconnect (e.g., InfiniBand), high-performance parallel file systems (e.g.,
Lustre /n/holyscratch01 and /n/holylfs filesystems), GPUs, and other resources
(e.g., licensed Intel compilers).

**Note for Windows and MacOS**: SingularityCE only
supports Linux containers. You cannot create images that use Windows or MacOS
(this is a restriction of the containerization model rather than SingularityCE).

## Docker security concerns

Containerization of workloads has become popular, particularly using
[Docker](https://www.docker.com/). However, Docker is not suitable for HPC
applications as one can gain root (i.e. admin) access to the system with Docker
containers, which poses a security risk. There are a couple of alternatives
for HPC containers, with [SingularityCE](https://sylabs.io/singularity/) being
the one that covers a large set of cases and is avaiable on the FASRC clusters.

## Docker vs. SingularityCE

There are some important differences between Docker and SingularityCE:

- Docker and SingularityCE have their own container formats.
- Docker containers may be imported to run via SingularityCE.
- Docker containers need root privileges for full functionality which is not
  suitable for a shared HPC environment.
- SingularityCE allows working with containers as a regular user.

## Singularity, SingularityCE, Apptainer

SingularityCE (Singularity Community Edition) and Apptainer are
branches/children of the deprecated Singularity. SingularityCE is maintained
by Sylabs while Apptainer is maintained by the Linux Foundation.

## SingularityCE vocabulary

- **SingularityCE** or **Appteiner** or **Docker**: the containerization software
  - as in "SingularityCE 3.11" or "Apptainer 1.0"
- **Image**: a compressed, usually read-only file that contains an OS and specific
  software stack
- **Container**
  - The technology, e.g. "containers vs. virtual machines"
  - An instance of an image, e.g. "I will run my simulation a SingularityCE
    container of PyTorch."
- **Host**: computer/supercomputer where the image is run 

# SingularityCE on the clusters

SingularityCE is available on both Cannon and FASSE clusters. SingularityCE is
only on the compute nodes on the cluster, i.e. it is not available on login
nodes. Therefore, to use it you need to:

- On Cannon
  - Start an interactive job with `salloc` command or
  - start an [Open OnDemand](https://vdi.rc.fas.harvard.edu/) job with Remote 
    Desktop app and launch a terminal or,
  - submit a batch-job
- On FASSE
  - start an [Open OnDemand](https://fasseood.rc.fas.harvard.edu/) job with 
    Remote Desktop app and launch a terminal or,
  - submit a batch-job
  - **note**: interactive jobs with `salloc` on FASSE are not allowed

Check SingularityCE version:

On Cannon:

```bash
[jharvard@boslogin01 ~]$ salloc -p rc-testing --mem 4g -t 0-01:00 -c 1
salloc: Granted job allocation 1451
salloc: Waiting for resource configuration
salloc: Nodes holy2c04309 are ready for job
[jharvard@holy2c04309 ~]$ singularity --version
singularity-ce version 3.11.0-1.el8
```

On FASSE:

Go to [FASSE Open OnDemand](https://fasseood.rc.fas.harvard.edu/) and start a
Remote Desktop job. Launch the Remote Desktop Session. When the Remote Desktop
opens, click on the terminal icon (or go to Applications -> Terminal Emulator).
In the terminal, type the command:

```bash
[jharvard@holy2c04309 ~]$ singularity --version
singularity-ce version 3.11.0-1.el8
```

## SingularityCE documentation

The [SingularityCE User
Guide](https://docs.sylabs.io/guides/latest/user-guide/index.html#) has the
latest documentation.

You can also see the most up-to-date help on SingularityCE from
the command line:

```bash
[jharvard@holy2c04309 ~]$ singularity --help

Linux container platform optimized for High Performance Computing (HPC) and
Enterprise Performance Computing (EPC)

Usage:
  singularity [global options...]

Description:
  Singularity containers provide an application virtualization layer enabling
  mobility of compute via both application and environment portability. With
  Singularity one is capable of building a root file system that runs on any
  other Linux system where Singularity is installed.

Options:
  -c, --config string   specify a configuration file (for root or
                        unprivileged installation only) (default
                        "/etc/singularity/singularity.conf")
  -d, --debug           print debugging information (highest verbosity)
  -h, --help            help for singularity
      --nocolor         print without color output (default False)
  -q, --quiet           suppress normal output
  -s, --silent          only print errors
  -v, --verbose         print additional information
      --version         version for singularity

Available Commands:
  build       Build a Singularity image
  cache       Manage the local cache
  capability  Manage Linux capabilities for users and groups
  completion  Generate the autocompletion script for the specified shell
  config      Manage various singularity configuration (root user only)
  delete      Deletes requested image from the library
  exec        Run a command within a container
  help        Help about any command
  inspect     Show metadata for an image
  instance    Manage containers running as services
  key         Manage OpenPGP keys
  oci         Manage OCI containers
  overlay     Manage an EXT3 writable overlay image
  plugin      Manage Singularity plugins
  pull        Pull an image from a URI
  push        Upload image to the provided URI
  remote      Manage singularity remote endpoints, keyservers and OCI/Docker registry credentials
  run         Run the user-defined default command within a container
  run-help    Show the user-defined help for an image
  search      Search a Container Library for images
  shell       Run a shell within a container
  sif         Manipulate Singularity Image Format (SIF) images
  sign        Add digital signature(s) to an image
  test        Run the user-defined tests within a container
  verify      Verify digital signature(s) within an image
  version     Show the version for Singularity

Examples:
  $ singularity help <command> [<subcommand>]
  $ singularity help build
  $ singularity help instance start


For additional help or support, please visit https://www.sylabs.io/docs/
```

# Build your own SingularityCE container

You can build or import a SingularityCE container in different ways. You can:

1. Download an existing container from the SingularityCE [Container
   Library](https://cloud.sylabs.io/library) or another image repository. This 
   will download an existing SingularityCE image to the FASRC cluster.
2. Download an existing container from
   [DockerHub](https://hub.docker.com/search). This will convert the Docker
   container into a SingularityCE container and the SingularityCE container will
   be downloaded to the FASRC cluster.
<!-- 
3. Create a `--sandbox` directory in the FASRC cluster that functions as a
   SingularityCE container.
-->
3. Build a SingularityCE container from a Singularity definition file directly
   on the FASRC clusters. This is an unpriledged build with `proot`, which means
   that it may have some limitations during the build. It will create a 
   SingularityCE container on the FASRC cluster.
4. Build a SingularityCE container remotely from a local Singularity definition
   file using option `--remote`. This will build an image in Sylabs cloud that
   is automatically downloaded to the FASRC cluster.

**NOTE**: for all options above, you need to be in a compute node. [Singularity
on the clusters](#SingularityCE-on-the-clusters) shows how to request an
interactive job on Cannon and FASSE.

## Download existing container from a library/repo

Build the laughing cow (`lolcow`) image from SingularityCE library:

```bash
[jharvard@holy2c02302 ~]$ singularity build lolcow.sif library://lolcow
INFO:    Starting build...
INFO:    Using cached image
INFO:    Verifying bootstrap image /n/home05/jharvard/.singularity/cache/library/sha256.cef378b9a9274c20e03989909930e87b411d0c08cf4d40ae3b674070b899cb5b
INFO:    Creating SIF file...
INFO:    Build complete: lolcow.sif
```

Build latest Ubuntu image from SingularityCE library:

```bash
[jharvard@holy2c02302 ~]$ singularity pull library://library/default/ubuntu
INFO:    Downloading library image
28.4MiB / 28.4MiB [=======================================] 100 % 7.0 MiB/s 0s
```

## Download an existing container from Docker Hub

Build the [laughing cow](https://hub.docker.com/r/sylabsio/lolcow) (`lolcow`)
image from Docker Hub:

```bash
[jharvard@holy2c02302 ~]$ singularity build lolcow.sif docker://sylabsio/lolcow
INFO:    Starting build...
Getting image source signatures
Copying blob 5ca731fc36c2 done
Copying blob 16ec32c2132b done
Copying config fd0daa4d89 done
Writing manifest to image destination
Storing signatures
2023/03/01 10:29:37  info unpack layer: sha256:16ec32c2132b43494832a05f2b02f7a822479f8250c173d0ab27b3de78b2f058
2023/03/01 10:29:38  info unpack layer: sha256:5ca731fc36c28789c5ddc3216563e8bfca2ab3ea10347e07554ebba1c953242e
INFO:    Creating SIF file...
INFO:    Build complete: lolcow.sif
```

Build [hello world](https://hub.docker.com/_/hello-world) image from Docker
Hub:

```bash
[jharvard@holy2c02302 ~]$ singularity build hello_world.sif docker://hello-world
INFO:    Starting build...
Getting image source signatures
Copying blob 2db29710123e done
Copying config 811f3caa88 done
Writing manifest to image destination
Storing signatures
2023/03/01 10:34:16  info unpack layer: sha256:2db29710123e3e53a794f2694094b9b4338aa9ee5c40b930cb8063a1be392c54
INFO:    Creating SIF file...
INFO:    Build complete: hello_world.sif
```

### Docker Rate Limiting

Docker Hub limits the number of pulls anonymous accounts can make. If you hit
either an error of:

```bash
ERROR: toomanyrequests: Too Many Requests.
```

or

```bash
You have reached your pull rate limit. You may increase the limit by authenticating and upgrading: https://www.docker.com/increase-rate-limits.
```

you will need to create a Docker account to increase your limit.  See the
[Docker documentation](https://www.docker.com/increase-rate-limits/) for more
details.

Once you have a Docker account, you can authenticate with Docker Hub with your
Docker Hub account (not FASRC account) and then run a Docker container

```bash
# use this to login to Docker Hub
[jharvard@holy2c02302 ~]$ singularity remote login --username <dockerhub_username> docker://docker.io

# run the usual command
[jharvard@holy2c02302 ~]$ singularity build lolcow.sif docker://sylabsio/lolcow
```

<!-- 
## Create a writable `--sandbox` directory

The `--sandbox` directory operates just like a container in a SIF file (i.e. a
singularity image). To make changes within the container/directory, you need to
launch it with the `--writable` option. This is a good option to test building
an image that you do not know exactly how to install packages and are still
testing the proper commands, packages, and dependencies that need to be
included.

To create a `--sandbox` directory for the laughing cow image from SingularityCE
library:

```bash
[jharvard@holy2c02302 ~]$ singularity build --sandbox lolcow/ library://lolcow
```
-->

## Build a SingularityCE container from a Singularity definition file

Until SingularityCE 3.10, users were not allowed to build Singularity images
from definition files in the FASRC clusters. Starting in version 3.11,
SingularityCE offers limited unprivileged builds with `proot`.

Builds using `proot` have limitations, as the emulation of the root user is not
complete. See [SingularityCE
docs](https://docs.sylabs.io/guides/latest/user-guide/build_a_container.html#unprivilged-proot-builds)
for more details on the limitations.

**Step 1**: Download `proot` in the directory `~/bin` (`~` or `$HOME` represent
`/n/homeNN/user_name`, where `NN` are numbers that may be different for
different users): 

```bash
# make ~/bin directory
[jharvard@holy2c02302 ~]$ mkdir -p ~/bin

# change to ~/bin directory, download proot, and change permissions to make it executable
[jharvard@holy2c02302 ~]$ cd ~/bin
[jharvard@holy2c02302 bin]$ curl -LO https://proot.gitlab.io/proot/bin/proot
[jharvard@holy2c02302 bin]$ chmod +x ./proot
```

**Step 2**: Ensure `~/bin`  is included in
your `PATH`.  If not, add it:

```bash
# print PATH
[jharvard@holy2c02302 ~]$ echo $PATH
/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/opt/puppetlabs/bin:/n/home01/jharvard/.local/bin

# since /n/home01/jharvard/bin is not part of PATH, add it
[jharvard@holy2c02302 ~]$ export PATH=$PATH:~/bin
```

**Step 3**: Write/obtain a definition file. You will need a definition file
specifying environmental variables, packages, etc. Your SingularityCE image will
be based on this file. See [SingularityCE definition file
docs](https://docs.sylabs.io/guides/latest/user-guide/definition_files.html) for
more details.

This is an example of the laughing cow definition file:

```bash
Bootstrap: docker
From: ubuntu:22.04

%post
    apt-get -y update
    apt-get -y install cowsay lolcat

%environment
    export LC_ALL=C
    export PATH=/usr/games:$PATH

%runscript
    date | cowsay | lolcat
```

**Step 4**: Build SingularityCE image

Note that if `proot` works, you get an output message saying

```bash
INFO:    Using proot to build unprivileged.
```

Build laughing cow image

```bash
[jharvard@holy2c02302 ~]$ singularity build lolcow.sif lolcow.def
INFO:    Using proot to build unprivileged. Not all builds are supported. If build fails, use --remote or --fakeroot.
INFO:    Starting build...
Getting image source signatures
Copying blob 76769433fd8a done

... omitted output ...

Running hooks in /etc/ca-certificates/update.d...
done.
INFO:    Adding environment to container
INFO:    Adding runscript
INFO:    Creating SIF file...
INFO:    Build complete: lolcow.sif
```

### Build a custom SingularityCE image with `proot` based on the FASRC CentOS 7 compute node

If you need to run your codes in the former operating system (pre June 2023)
CentOS 7, you can use the SingularityCE image of a CentOS7 compute node. In
addition, you can add your own software/library/packages under the `%post`
header

**Step 1**: Copy the CentOS 7 compute image to your holylabs (or home
directory). The base image file needs to be copied to a directory that you have
read/write access, otherwise it will fail to build your custom image

```bash
[jharvard@holy2c02302 ~]$ cd /n/holylabs/LABS/jharvard_lab/Users/jharvard
[jharvard@holy2c02302 jharvard]$ cp/n/holystore01/SINGULARITY/FAS/centos7/compute-el7-noslurm-2023-02-15.sif . 
```

**Step 2**: In definition file `centos7_custom.def`, set `Bootstrap: localimage`
and put the path of the existing image that you copied for the `From:` field.
Then, add your packages/software/libraries that you need.  Here, we add
`cowsay`:

```bash
Bootstrap: localimage
From: compute-el7-noslurm-2023-02-15.sif

%help
    This is CentOS 7 Singularity container based on the Cannon compute node with my added programs.

%post
    yum -y update
    yum -y install cowsay
```

**Step 3**: Build the SingularityCE image

```bash
[jharvard@holy2c02302 jharvard]$ singularity build centos7_custom.sif centos7_custom.def
INFO:    Using proot to build unprivileged. Not all builds are supported. If build fails, use --remote or --fakeroot.
INFO:    Starting build...
INFO:    Verifying bootstrap image compute-el7-noslurm-2023-02-15.sif
WARNING: integrity: signature not found for object group 1
WARNING: Bootstrap image could not be verified, but build will continue.
INFO:    Running post scriptlet
+ yum -y update

... omitted output ...

Running transaction
  Installing : cowsay-3.04-4.el7.noarch                   1/1
  Verifying  : cowsay-3.04-4.el7.noarch                   1/1

Installed:
  cowsay.noarch 0:3.04-4.el7

Complete!
INFO:    Adding help info
INFO:    Creating SIF file...
INFO:    Build complete: centos7_custom.sif
```

## Build a SingularityCE container remotely from Singularity definition file using option `--remote`. 

If the unpriveleged `proot` build does not work for you, you can use Sylabs
cloud free service to build Singularity images.

**Step 1**: Create a Sylabs account

1. Go to https://cloud.sylabs.io/library
2. Click “Sign in” on the top right corner
3. Select your method to sign in, with Google, GitLab, HitHub, or Microsoft

**Step 2**: Create a Sylabs access token

1. Go to: https://cloud.sylabs.io/
2. Click “Sign In” and follow the sign in steps.
3. Click on your login icon the top right corner
4. Select “Access Tokens” from the drop down menu.
5. Enter a name for your new access token, such as “Cannon token”.
6. Click the “Create a New Access Token” button.
7. Copy the newly created open (don't close the browser window yet in case you
   need to copy it again)

**Step 3**: Login to Sylabs cloud by adding your Sylabs token on the FASRC
cluster

```bash
[jharvard@holy2c02302 ~]$ singularity remote login
Generate an access token at https://cloud.sylabs.io/auth/tokens, and paste it here.
Token entered will be hidden for security.
Access Token:
```

Paste your token. If successful, you should see a message similar to this:

```bash
INFO:    Access Token Verified!
INFO:    Token stored in /n/home01/jharvard/.singularity/remote.yaml
```

**Step 4**: Singularity definition file

In order to build the Singularity container, you will need to have a definition
file. In the example below, the definition file `centos7.def` may have various
headers that are indicated by the `%` sign (e.g., `%help`, `%post`). To add your
own software installs, add the install commands under the `%post` header. For
more details, refer to the [Singularity definition file
documentation](https://docs.sylabs.io/guides/latest/user-guide/definition_files.html).

```bash
Bootstrap: yum
OSVersion: 7
MirrorURL: http://mirror.centos.org/centos-%{OSVERSION}/%{OSVERSION}/os/$basearch/
Include: yum

%help
    This is Centos 7 Singularity container for my own programs to run in the Cannon cluster.

%post
    yum -y install vim-minimal
    yum -y install gcc
    yum -y install gcc-gfortran
    yum -y install gcc-c++
    yum -y install which tar wget gzip bzip2
    yum -y install make
    yum -y install perl
```

**Step 5**: Build the SingularityCE container

Note that, depending on the libraries and packages added to the container, the
build can take 30+ minutes.

```bash
[jharvard@holy2c02302 ~]$ singularity build --remote centos7.sif centos7.def
INFO:    Starting build...
INFO:    Setting maximum build duration to 1h0m0s
INFO:    Remote "cloud.sylabs.io" added.
INFO:    Access Token Verified!
INFO:    Token stored in /root/.singularity/remote.yaml
INFO:    Remote "cloud.sylabs.io" now in use.
INFO:    Starting build...
INFO:    Skipping GPG Key Import
INFO:    Adding owner write permission to build path: /tmp/build-temp-3368736037/rootfs
INFO:    Running post scriptlet
+ yum -y install vim-minimal
Loaded plugins: fastestmirror
Determining fastest mirrors

... omitted output ...

Complete!
INFO:    Adding help info
INFO:    Creating SIF file...
INFO:    Build complete: /tmp/image-262939644
INFO:    Performing post-build operations
INFO:    Generating SBOM for /tmp/image-262939644
INFO:    Adding SBOM to SIF
INFO:    Calculating SIF image checksum
INFO:    Uploading image to library...
WARNING: Skipping container verification
INFO:    Uploading 226950986 bytes
INFO:    Image uploaded successfully.
INFO:    Build complete: centos7.sif
```

