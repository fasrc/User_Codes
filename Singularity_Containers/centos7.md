# Running Singularity image with CentOS 7 on Rocky 8

If you absolutely need to still use CentOS 7 after the OS upgrade to Rocky 8,
you can use it with SingularityCE. For more information on SingularityCE, see
our [Singularity documentation](README.md).

We have a Singularity image with CentOS7 and the same environment of compute
nodes (as of March 15th, 2023). In addition, you can access CentOS 7 modules
from within the Singularity container. The image is stored at:

```bash
/n/singularity_images/FAS/centos7/compute-el7-noslurm-2023-03-29.sif
```

You can execute this image and/or copy it, but you cannot modify it in its
original location. See below how you can modify this image.

## Run Singularity with CentOS 7

To get a bash shell on CentOS 7 environment, you can run:

```bash
[jharvard@holy7c12102 ~]$ singularity run /n/singularity_images/FAS/centos7/compute-el7-noslurm-2023-03-29.sif
Singularity>
```

or

```bash
[jharvard@holy7c12102 ~]$ singularity exec /n/singularity_images/FAS/centos7/compute-el7-noslurm-2023-03-29.sif /bin/bash
Singularity>
```

**NOTE**: The command `singularity shell` is also an option. However it does not
allow you to access modules as explained in [Load modules](#load-modules)

## Load modules

You can still load modules that were available on CentOS 7 from inside the
Singularity container:

```bash
[jharvard@holy7c12102 ~]$ singularity exec /n/singularity_images/FAS/centos7/compute-el7-noslurm-2023-03-29.sif /bin/bash
Singularity> module load gcc
Singularity> module load matlab
Singularity> module list

Currently Loaded Modules:
  1) gmp/6.2.1-fasrc01   2) mpfr/4.1.0-fasrc01   3) mpc/1.2.1-fasrc01   4) gcc/12.1.0-fasrc01   5) matlab/R2022b-fasrc01
```

Note that the modules are from the CentOS 7 environment:

```bash
Singularity> module display matlab/R2022b-fasrc01
-----------------------------------------------------------------------------------------------------------------------------------------------------------
   /n/helmod/modulefiles/centos7/Core/matlab/R2022b-fasrc01.lua:
-----------------------------------------------------------------------------------------------------------------------------------------------------------
help([[matlab-R2022b-fasrc01
a high-level language and interactive environment for numerical computation, visualization, and programming

]], [[
]])
whatis("Name: matlab")
whatis("Version: R2022b-fasrc01")
whatis("Description: a high-level language and interactive environment for numerical computation, visualization, and programming")
setenv("MATLAB_HOME","/n/helmod/apps/centos7/Core/matlab/R2022b-fasrc01")
prepend_path("PATH","/n/helmod/apps/centos7/Core/matlab/R2022b-fasrc01/bin")
setenv("MLM_LICENSE_FILE","27000@rclic1")
setenv("ZIENA_LICENSE_NETWORK_ADDR","10.242.113.134:8349")
```

## Submit slurm jobs

If you need to submit a job rather than getting to a shell, you can add 


**NOTE**: You cannot submit slurm jobs from **inside** the container, but you
can submit a slurm job that will execute the container.

## Modify SingularityCE image with CentOS 7

If you need to run your codes in the former operating system (pre June 2023),
you can build a custom SingularityCE image with `proot`. The base image is the
the FASRC CentOS 7 compute node image, and you can add your own
software/library/packages under the `%post` header in the Singularity definition
file.

**Step 1**: Follow steps 1 and 2 in [setup proot](READ.md#build-a-singularityCE-container-from-a-singularity-definition-file)

**Step 2**: Copy the CentOS 7 compute image to your holylabs (or home
directory). The base image file needs to be copied to a directory that you have
read/write access, otherwise it will fail to build your custom image

```bash
[jharvard@holy2c02302 ~]$ cd /n/holylabs/LABS/jharvard_lab/Users/jharvard
[jharvard@holy2c02302 jharvard]$ cp /n/holystore01/SINGULARITY/FAS/centos7/compute-el7-noslurm-2023-02-15.sif .
```

**Step 3**: In definition file `centos7_custom.def`, set `Bootstrap: localimage`
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











