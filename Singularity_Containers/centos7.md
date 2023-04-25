# Running Singularity image with CentOS 7 on Rocky 8

If you absolutely need to still use CentOS 7 after the OS upgrade to Rocky 8,
you can use it with SingularityCE. For more information on SingularityCE, see
our [Singularity documentation](README.md).

We have a Singularity image with CentOS7 and the same environment of compute
nodes (as of March 29th, 2023). In addition, you can access CentOS 7 modules
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

If you need to submit a job rather than getting to a shell, you have to do the
following steps in the appropriate order:

1. launch the singularity image
2. load modules
3. (optional) compile code
4. execute code

If you try to load modules before launching the image, it will try to load
modules from the Rocky 8 host system.

To ensure that steps 2-4 are run within the singularity container, they are
place between `END` (see slurm batch script below).

**NOTE**: You cannot submit slurm jobs from **inside** the container, but you
can submit a slurm job that will execute the container.

Example with a simple `hello_world.f90` fortran code:

```bash
program hello
  print *, 'Hello, World!'
end program hello
```

Slurm batch script `run_singularity_centos7.sh`:

```bash
#!/bin/bash
#SBATCH -J sing_hello           # Job name
#SBATCH -p rocky                # Partition(s) (separate with commas if using multiple)
#SBATCH -c 1                    # Number of cores
#SBATCH -t 0-00:10:00           # Time (D-HH:MM:SS)
#SBATCH --mem=500M              # Memory
#SBATCH -o sing_hello_%j.out    # Name of standard output file
#SBATCH -e sing_hello_%j.err    # Name of standard error file

# start a bash shell inside singularity image
singularity run /n/singularity_images/FAS/centos7/compute-el7-noslurm-2023-03-29.sif <<END

# load modules
module load gcc/12.1.0-fasrc01
module list

# compile code
gfortran hello_world.f90 -o hello.exe

# execute code
./hello.exe
END
```

To ensure that the commands are run within the singularity container, they are
place between `END`.

To submit the slurm batch script:

```bash
sbatch run_singularity_centos7.sh
```

Another option have a bash script with steps 2-4 and then use `singularity run`
to execute the script. For example, `script_inside_container.sh`:

```bash
#!/bin/bash

# load modules
module load gcc/12.1.0-fasrc01
module list

# compile code
gfortran hello_world.f90 -o hello.exe

# execute code
./hello.exe
```

And the slurm batch script `run_singularity_centos7_script.sh` becomes:

```bash
#!/bin/bash
#SBATCH -J sing_hello           # Job name
#SBATCH -p rocky                # Partition(s) (separate with commas if using multiple)
#SBATCH -c 1                    # Number of cores
#SBATCH -t 0-00:10:00           # Time (D-HH:MM:SS)
#SBATCH --mem=500M              # Memory
#SBATCH -o sing_hello_%j.out    # Name of standard output file
#SBATCH -e sing_hello_%j.err	# Name of standard error file

# start a bash shell inside singularity image
singularity run /n/singularity_images/FAS/centos7/compute-el7-noslurm-2023-03-29.sif script_inside_container.sh
```

You can submit a batch job with:

```bash
sbatch run_singularity_centos7_script.sh
```

## Modify SingularityCE image with CentOS 7

If you need to run your codes in the former operating system (pre June 2023),
you can build a custom SingularityCE image with `proot`. The base image is the
the FASRC CentOS 7 compute node image, and you can add your own
software/library/packages under the `%post` header in the Singularity definition
file.

**Step 1**: Follow steps 1 and 2 in [setup proot](README.md#build-a-singularityCE-container-from-a-singularity-definition-file)

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











