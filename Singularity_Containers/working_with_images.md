# Working with SingularityCE images

When working with images you can:

- Start an interactive session, or
- Submit a slurm batch job to run Singularity

For more examples and details, see [SingularityCE quick start
guide](https://docs.sylabs.io/guides/latest/user-guide/quick_start.html#interact-with-images)

## Running SingularityCE images interactively 

For this example, we will use the laughing cow SingularityCE image from
[Sylabs library](https://cloud.sylabs.io/library/).

First, request interactive job (for more details about interactive jobs on
Cannon, see and on FASSE see) and download the laughing cow `lolcow_latest.sif`
SingularityCE image:

```bash
# request interative job
[jharvard@holylogin01 ~]$ salloc -p test -c 1 -t 00-01:00 --mem=4G

# pull image from Sylabs library
[jharvard@holy2c02302 sylabs_lib]$ singularity pull library://lolcow
FATAL:   Image file already exists: "lolcow_latest.sif" - will not overwrite
[jharvard@holy2c02302 sylabs_lib]$ rm lolcow_latest.sif
[jharvard@holy2c02302 sylabs_lib]$ singularity pull library://lolcow
INFO:    Downloading library image
90.4MiB / 90.4MiB [=====================================] 100 % 7.6 MiB/s 0s
```

### Shell

With the `shell` command, you can start a new shell within the container image
and interact with it as if it were a small virtual machine. 

Note that the `shell` command does not source `~/.bashrc` and `~/bash_profile`.
Therefore, the `shell` command is useful if customizations in your `~/.bashrc`
and `~/bash_profile` are not supposed to be sourced within the SingularityCE
container.

```bash
# launch container with shell command
[jharvard@holy2c02302 sylabs_lib]$ singularity shell lolcow_latest.sif

# test some linux commands within container
Singularity> pwd
/n/holylabs/LABS/jharvard_lab/Users/jharvard/sylabs_lib
total 95268
-rwxr-xr-x 1 jharvard jharvard_lab  2719744 Mar  9 14:27 hello-world_latest.sif
drwxr-sr-x 2 jharvard jharvard_lab     4096 Mar  1 15:21 lolcow
-rwxr-xr-x 1 jharvard jharvard_lab 94824197 Mar  9 14:56 lolcow_latest.sif
drwxr-sr-x 2 jharvard jharvard_lab     4096 Mar  1 15:23 ubuntu22.04
Singularity> id
uid=21442(jharvard) gid=10483(jharvard_lab) groups=10483(jharvard_lab)
Singularity> cowsay moo
 _____
< moo >
 -----
        \   ^__^
         \  (oo)\_______
            (__)\       )\/\
                ||----w |
                ||     ||

# exit the container
Singularity> exit
[jharvard@holy2c02302 sylabs_lib]$
```

### Executing commands

The `exec` command allows you to execute a custom command within a container by
specifying the image file. For instance, to execute the `cowsay` program within
the `lolcow_latest.sif` container:

```bash
[jharvard@holy2c02302 sylabs_lib]$ singularity exec lolcow_latest.sif cowsay moo
 _____
< moo >
 -----
        \   ^__^
         \  (oo)\_______
            (__)\       )\/\
                ||----w |
                ||     ||
[jharvard@holy2c02302 sylabs_lib]$ singularity exec lolcow_latest.sif cowsay "hello FASRC"
 _____________
< hello FASRC >
 -------------
        \   ^__^
         \  (oo)\_______
            (__)\       )\/\
                ||----w |
                ||     ||
```

### Run scripts within a container

SingularityCE containers may contain
[runscripts](https://docs.sylabs.io/guides/latest/user-guide/definition_files.html#runscript).
These are user defined scripts that define the actions a container should
perform when someone runs it. The runscript can be triggered with the `run`
command, or simply by calling the container as though it were an executable.

Using the `run` command:

```bash
[jharvard@holy2c02302 sylabs_lib]$ singularity run lolcow_latest.sif
 _____________________________
< Thu Mar 9 15:15:56 UTC 2023 >
 -----------------------------
        \   ^__^
         \  (oo)\_______
            (__)\       )\/\
                ||----w |
                ||     ||
```

Running as the container were an executable file:

```bash
[jharvard@holy2c02302 sylabs_lib]$ ./lolcow_latest.sif
 _____________________________
< Thu Mar 9 15:17:06 UTC 2023 >
 -----------------------------
        \   ^__^
         \  (oo)\_______
            (__)\       )\/\
                ||----w |
                ||     ||
```

## GPU example

First, start an interactive job in the `gpu` or `gpu_test` partition and then
download the Singularity image.

```bash
# request interactive job on gpu_test partition
[jharvard@holylogin01 gpu_example]$ salloc -p gpu_test --gres=gpu:1 --mem 4G -c 4 -t 60

# build singularity image by pulling container from Docker Hub
[jharvard@holygpu7c1309 gpu_example]$ singularity pull docker://tensorflow/tensorflow:latest-gpu
INFO:    Converting OCI blobs to SIF format
INFO:    Starting build...
Getting image source signatures
Copying blob 521d4798507a done
Copying blob 2798fbbc3b3b done
Copying blob 4d8ee731d34e done
Copying blob 92d2e1452f72 done
Copying blob 6aafbce389f9 done
Copying blob eaead16dc43b done
Copying blob 69cc8495d782 done
Copying blob 61b9b57b3915 done
Copying blob eac8c9150c0e done
Copying blob af53c5214ca1 done
Copying blob fac718221aaf done
Copying blob 2047d1a62832 done
Copying blob 9a9a3600909b done
Copying blob 79931d319b40 done
Copying config bdb8061f4b done
Writing manifest to image destination
Storing signatures
2023/03/09 13:52:18  info unpack layer: sha256:eaead16dc43bb8811d4ff450935d607f9ba4baffda4fc110cc402fa43f601d83
2023/03/09 13:52:19  info unpack layer: sha256:2798fbbc3b3bc018c0c246c05ee9f91a1ebe81877940610a5e25b77ec5d4fe24
2023/03/09 13:52:19  info unpack layer: sha256:6aafbce389f98e508428ecdf171fd6e248a9ad0a5e215ec3784e47ffa6c0dd3e
2023/03/09 13:52:19  info unpack layer: sha256:4d8ee731d34ea0ab8f004c609993c2e93210785ea8fc64ebc5185bfe2abdf632
2023/03/09 13:52:19  info unpack layer: sha256:92d2e1452f727e063220a45c1711b635ff3f861096865688b85ad09efa04bd52
2023/03/09 13:52:19  info unpack layer: sha256:521d4798507a1333de510b1f5474f85d3d9a00baa9508374703516d12e1e7aaf
2023/03/09 13:52:21  warn rootless{usr/lib/x86_64-linux-gnu/gstreamer1.0/gstreamer-1.0/gst-ptp-helper} ignoring (usually) harmless EPERM on setxattr "security.capability"
2023/03/09 13:52:54  info unpack layer: sha256:69cc8495d7822d2fb25c542ab3a66b404ca675b376359675b6055935260f082a
2023/03/09 13:52:58  info unpack layer: sha256:61b9b57b3915ef30727fb8807d7b7d6c49d7c8bdfc16ebbc4fa5a001556c8628
2023/03/09 13:52:58  info unpack layer: sha256:eac8c9150c0e4967c4e816b5b91859d5aebd71f796ddee238b4286a6c58e6623
2023/03/09 13:52:59  info unpack layer: sha256:af53c5214ca16dbf9fd15c269f3fb28cefc11121a7dd7c709d4158a3c42a40da
2023/03/09 13:52:59  info unpack layer: sha256:fac718221aaf69d29abab309563304b3758dd4f34f4dad0afa77c26912aed6d6
2023/03/09 13:53:00  info unpack layer: sha256:2047d1a62832237c26569306950ed2b8abbdffeab973357d8cf93a7d9c018698
2023/03/09 13:53:15  info unpack layer: sha256:9a9a3600909b9eba3d198dc907ab65594eb6694d1d86deed6b389cefe07ac345
2023/03/09 13:53:15  info unpack layer: sha256:79931d319b40fbdb13f9269d76f06d6638f09a00a07d43646a4ca62bf57e9683
INFO:    Creating SIF file...
```

Run the container with GPU support, see available GPUs, and check if
`tensorflow` can detect them:

```bash
# run the container
[jharvard@holygpu7c1309 gpu_example]$ singularity shell --nv tensorflow_latest-gpu.sif
Singularity> nvidia-smi
Thu Mar  9 18:57:53 2023
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.85.12    Driver Version: 525.85.12    CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla V100-PCIE...  On   | 00000000:06:00.0 Off |                    0 |
| N/A   35C    P0    25W / 250W |      0MiB / 32768MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  Tesla V100-PCIE...  On   | 00000000:2F:00.0 Off |                    0 |
| N/A   36C    P0    23W / 250W |      0MiB / 32768MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   2  Tesla V100-PCIE...  On   | 00000000:86:00.0 Off |                    0 |
| N/A   35C    P0    25W / 250W |      0MiB / 32768MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   3  Tesla V100-PCIE...  On   | 00000000:D8:00.0 Off |                    0 |
| N/A   33C    P0    23W / 250W |      0MiB / 32768MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+

# check if `tensorflow` can see GPUs
Singularity> python
Python 3.8.10 (default, Jun 22 2022, 20:18:18)
[GCC 9.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from tensorflow.python.client import device_lib
2023-03-09 19:00:15.107804: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
>>> print(device_lib.list_local_devices())
2023-03-09 19:00:20.010087: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-03-09 19:00:24.024427: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1613] Created device /device:GPU:0 with 30960 MB memory:  -> device: 0, name: Tesla V100-PCIE-32GB, pci bus id: 0000:06:00.0, compute capability: 7.0
2023-03-09 19:00:24.026521: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1613] Created device /device:GPU:1 with 30960 MB memory:  -> device: 1, name: Tesla V100-PCIE-32GB, pci bus id: 0000:2f:00.0, compute capability: 7.0
2023-03-09 19:00:24.027583: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1613] Created device /device:GPU:2 with 30960 MB memory:  -> device: 2, name: Tesla V100-PCIE-32GB, pci bus id: 0000:86:00.0, compute capability: 7.0
2023-03-09 19:00:24.028227: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1613] Created device /device:GPU:3 with 30960 MB memory:  -> device: 3, name: Tesla V100-PCIE-32GB, pci bus id: 0000:d8:00.0, compute capability: 7.0

... omitted output ...

incarnation: 3590943835431918555
physical_device_desc: "device: 3, name: Tesla V100-PCIE-32GB, pci bus id: 0000:d8:00.0, compute capability: 7.0"
xla_global_id: 878896533
]
```

## Running SingularityCE images in slurm batch submit jobs

You can also use SingularityCE images within a non-interactive batch script as you
would any other command. If your image contains a run-script then you can use
`singularity run` to execute the run-script in the job. You can also use
`singularity exec` to execute arbitrary commands (or scripts) within the image.

Below is an example batch-job submission script using the laughing cow
`lolcow_latest.sif` to print out information about the native OS of the image.

File `singularity.sbatch`:

```bash
#!/bin/bash
#SBATCH -J singularity_test
#SBATCH -o singularity_test.out
#SBATCH -e singularity_test.err
#SBATCH -p test
#SBATCH -t 0-00:10
#SBATCH -c 1
#SBATCH --mem=4000

# Singularity command line options
singularity exec lolcow_latest.sif cowsay "hello from slurm batch job"
```

Submit a slurm batch job:

```bash
[jharvard@holy2c02302 jharvard]$ sbatch singularity.sbatch
```

Upon the job completion, the standard output is located in the file
`singularity_test.out`:

```bash
 [jharvard@holy2c02302 jharvard]$ cat singularity_test.out
  ____________________________
< hello from slurm batch job >
 ----------------------------
        \   ^__^
         \  (oo)\_______
            (__)\       )\/\
                ||----w |
                ||     ||
```

### GPU example slurm batch script job

File `singularity_gpu.sbatch` (ensure to include the `--nv` flag after
`singularity exec`): 

```bash
#!/bin/bash
#SBATCH -J singularity_gpu_test
#SBATCH -o singularity_gpu_test.out
#SBATCH -e singularity_gpu_test.err
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 0-00:10
#SBATCH -c 1
#SBATCH --mem=4000

# Singularity command line options
singularity exec --nv lolcow_latest.sif nvidia-smi
```

Submit a slurm batch job:

```bash
[jharvard@holy2c02302 jharvard]$ sbatch singularity_gpu.sbatch
```

Upon the job completion, the standard output is located in the file
`singularity_gpu_test.out`:

```bash
$ cat singularity_gpu_test.out
Thu Mar  9 20:40:24 2023
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.85.12    Driver Version: 525.85.12    CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla V100-PCIE...  On   | 00000000:06:00.0 Off |                    0 |
| N/A   35C    P0    25W / 250W |      0MiB / 32768MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```


## Accessing files from a container

Files and directories on the cluster are accessible from within the container.
By default, directories under `/n`, `$HOME`, `$PWD`, and `/tmp` are available at
runtime inside the container.

See these variables on the host operting system:

```bash
[jharvard@holy2c02302 jharvard]$ echo $PWD
/n/holylabs/LABS/jharvard_lab/Lab/jharvard
[jharvard@holy2c02302 jharvard]$ echo $HOME
/n/home01/jharvard
[jharvard@holy2c02302 jharvard]$ echo $SCRATCH
/n/holyscratch01
```

The same variables within the container:

```bash
[jharvard@holy2c02302 jharvard]$ singularity shell lolcow_latest.sif
Singularity> echo $PWD
/n/holylabs/LABS/jharvard_lab/Lab/jharvard
Singularity> echo $HOME
/n/home01/jharvard
Singularity> echo $SCRATCH
/n/holyscratch01
```

You can specify additional directories from the host system such that they can
be accessible from the container. This prcess is called bind mount into your
container and is done with the `--bind` option. 

For instance, if you first create a file `hello.dat` in the
/scratch directory on the host system. Then, you can execute from within the
container by bind mounting /scratch to the /mnt directory inside the container:

```bash
[jharvard@holy2c02302 jharvard]$ echo 'Hello from file in mounted directory!' > /scratch/hello.dat
[jharvard@holy2c02302 jharvard]$ singularity shell --bind /scratch:/mnt lolcow_latest.sif
Singularity> cd /mnt/
Singularity> ls
cache  hello.dat
Singularity> cat hello.dat
Hello from file in mounted directory
```

If you don't use the `--bind` option, the file will not be available in the
directory `/mnt` inside the container:

```bash
[jharvard@holygpu7c1309 sylabs_lib]$ singularity shell lolcow_latest.sif
Singularity> cd /mnt/
Singularity> ls
Singularity>
```

