## PyTorch

<img src="Images/pytorch-logo.png" alt="PyTorch-logo" width="200"/>

### What is PyTorch?

[PyTorch](https://pytorch.org) is a GPU accelerated tensor computational framework with a Python front end. Functionality can be easily extended with common Python libraries such as NumPy, SciPy, and Cython. Automatic differentiation is done with a tape-based system at both a functional and neural network layer level. This functionality brings a high level of flexibility and speed as a deep learning framework and provides accelerated NumPy-like functionality.

### Installing PyTorch:

These instructions are intended to help you install PyTorch on the FASRC cluster.

#### GPU Support

For general information on running GPU jobs refer to our [user documentation](https://www.rc.fas.harvard.edu/resources/documentation/gpgpu-computing-on-the-cluster).

To set up PyTorch with GPU support in your user environment, please follow the below steps:

(1) Start an interactive job requesting GPUs, e.g.,

```bash
$ srun --pty -p gpu -t 0-06:00 --mem=8000 --gres=gpu:1 /bin/bash
```

(2) Load required software modules, e.g.,

```bash
$ module load python/3.6.3-fasrc02
$ module load cuda/10.0.130-fasrc01
$ module load cudnn/7.4.1.5_cuda10.0-fasrc01
```

(3) Create a [conda environment](https://conda.io/projects/conda/en/latest/index.html), e.g.,

```bash
$ conda create -n pt1.3_cuda10 python=3.7 pip numpy wheel matplotlib
```

(4) Activate the new *conda* environment:

```bash
$ source activate pt1.3_cuda10
(pt1.3_cuda10)
```

(5) Install PyTorch with conda

```bash
$ conda install pytorch=1.3 torchvision cudatoolkit=10.0 -c pytorch
```

### Running PyTorch:

#### Run PyTorch Interactively

For an **interactive session** to work with the GPUs you can use following:

```bash
$ srun --pty -p gpu -t 0-06:00 --mem=8000 --gres=gpu:1 /bin/bash
```

Load required software modules and source your PyTorch conda environment.

```bash
[username@holygpu2c0716 ~]$ module load cuda/10.0.130-fasrc01 cudnn/7.4.1.5_cuda10.0-fasrc01 python/3.6.3-fasrc02 && source activate pt1.3_cuda10
(pt1.3_cuda10)
```

Test PyTorch interactively:

```bash
(pt1.3_cuda10) $ python check_gpu.py 
Using device: cuda

Tesla V100-PCIE-32GB
Memory Usage:
Allocated: 0.0 GB
Cached:    0.0 GB

tensor([[-1.2709,  2.0035]], device='cuda:0')
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
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
    print()

# Run a small test on the available device
T = torch.randn(1,2).to(device)
print(T)
```

#### Batch Jobs

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
module load Anaconda3/5.0.1-fasrc02
module load cuda/10.0.130-fasrc01 cudnn/7.4.1.5_cuda10.0-fasrc01

# Run program
srun -c 1 --gres=gpu:1 python check_gpu.py 
```

If you name the above batch-job submission script <code>run.sbatch</code>, for instance, the job is submitted with:

```bash
$ sbatch run.sbatch
```

### References:

* [Official PyTorch website](https://pytorch.org)
* [PyTorch Tutorials](https://pytorch.org/tutorials/)
