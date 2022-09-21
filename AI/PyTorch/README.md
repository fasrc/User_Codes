## PyTorch

<img src="Images/pytorch-logo.png" alt="PyTorch-logo" width="200"/>

### What is PyTorch?

[PyTorch](https://pytorch.org) is a GPU accelerated tensor computational framework with a Python front end. Functionality can be easily extended with common Python libraries such as NumPy, SciPy, and Cython. Automatic differentiation is done with a tape-based system at both a functional and neural network layer level. This functionality brings a high level of flexibility and speed as a deep learning framework and provides accelerated NumPy-like functionality.

## Installing PyTorch:

These instructions are intended to help you install PyTorch on the FASRC cluster.

### GPU Support

For general information on running GPU jobs refer to our [user documentation](https://www.rc.fas.harvard.edu/resources/documentation/gpgpu-computing-on-the-cluster).

To set up PyTorch with GPU support in your user environment, please follow the below steps:

(1) Start an interactive job requesting GPUs, e.g.,

```bash
salloc -p gpu -t 0-06:00 --mem=8000 --gres=gpu:1 
```

(2) Load required software modules, e.g.,

```bash
module load python/3.8.5-fasrc01
module load cuda/11.7.1-fasrc01
```

(3) Create a [conda environment](https://conda.io/projects/conda/en/latest/index.html), e.g.,

```bash
conda create -n pt1.12_cuda11.6 python=3.10 pip numpy wheel matplotlib
```

(4) Activate the new *conda* environment:

```bash
source activate pt1.12_cuda11.6
```

(5) Install PyTorch with conda

```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
```
**Note:** If you use *CUDA version 11.6*, you will need to specify the <code>conda-forge</code> conda channel.

**Pull a PyTorch Singularity container**

Alternatively, one can pull and use a PyTorch [singularity](https://docs.sylabs.io/guides/3.5/user-guide/index.html) container:

```bash
singularity pull docker://pytorch/pytorch:latest
```
This will result in the image <code>pytorch_latest.sif</code>. The image then can be used with, e.g.,

```python
$ singularity exec --nv pytorch_latest.sif python
Python 3.7.13 (default, Mar 29 2022, 02:18:16) 
[GCC 7.5.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> print(torch.__version__)
1.12.0
>>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
>>> print('Using device:', device)
Using device: cuda
>>> T = torch.randn(1, 4).to(device)
>>> print(T)
tensor([[-1.1169,  0.3600, -0.3471, -0.7036]], device='cuda:0')
```

## Running PyTorch:

### Run PyTorch Interactively

For an **interactive session** to work with the GPUs you can use following:

```bash
salloc -p gpu -t 0-06:00 --mem=8000 --gres=gpu:1 
```

Load required software modules and source your PyTorch conda environment.

```bash
[username@holygpu2c0716 ~]$ module load python/3.8.5-fasrc01 cuda/11.7.1-fasrc01  && source activate pt1.12_cuda11.6
(pt1.12_cuda11.6)
```

Test PyTorch interactively:

```bash
[username@holygpu7c26301 ~]$ python check_gpu.py 
Using device: cuda

NVIDIA A100-SXM4-40GB
Memory Usage:
Allocated: 0.0 GB
Reserved:  0.0 GB

tensor([[-0.7148,  0.7627, -0.0389, -0.4436]], device='cuda:0')
```

<code>check_gpu.py</code> checks if GPUs are available and if available sets up the device to use them.

```python
#!/usr/bin/env python
import torch

# Setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

# Print out additional information when using CUDA
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Reserved: ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    print()

# Run a small test on the available device
T = torch.randn(1, 4).to(device)
print(T)
```

### Batch Jobs

An example batch-job submission script is included below:

```bash
#!/bin/bash
#SBATCH -c 1
#SBATCH -N 1
#SBATCH -t 0-00:30
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH -o pytorch_%j.out 
#SBATCH -e pytorch_%j.err 

# Load software modules and source conda environment
module load python/3.8.5-fasrc01
module load cuda/11.7.1-fasrc01
source activate pt1.12_cuda11.6

# Run program
srun -c 1 --gres=gpu:1 python check_gpu.py 
```

If you name the above batch-job submission script <code>run.sbatch</code>, for instance, the job is submitted with:

```bash
sbatch run.sbatch
```

## References:

* [Official PyTorch website](https://pytorch.org)
* [PyTorch Tutorials](https://pytorch.org/tutorials/)
