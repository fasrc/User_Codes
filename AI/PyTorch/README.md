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

<pre>
$ salloc -p gpu -t 0-06:00 --mem=8000 --gres=gpu:1 
</pre>

(2) Load required software modules, e.g.,

<pre>
$ module load python/3.8.5-fasrc01
$ module load cuda/11.4.2-fasrc01
</pre>

(3) Create a [conda environment](https://conda.io/projects/conda/en/latest/index.html), e.g.,

<pre>
$ conda create -n pt1.8_cuda111 python=3.8 pip numpy wheel matplotlib
</pre>

(4) Activate the new *conda* environment:

<pre>
$ source activate pt1.8_cuda111
(pt1.8_cuda111)
</pre>

(5) Install PyTorch with conda

<pre>
$ conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
</pre>

### Running PyTorch:

#### Run PyTorch Interactively

For an **interactive session** to work with the GPUs you can use following:

<pre>
$ salloc -p gpu -t 0-06:00 --mem=8000 --gres=gpu:1 
</pre>

Load required software modules and source your PyTorch conda environment.

<pre>
[username@holygpu2c0716 ~]$ module load python/3.8.5-fasrc01 cuda/11.4.2-fasrc01  && source activate pt1.8_cuda111
(pt1.7_cuda102)
</pre>

Test PyTorch interactively:

<pre>
(pt1.8_cuda111) $ python check_gpu.py 
Using device: cuda

A100-PCIE-40GB
Memory Usage:
Allocated: 0.0 GB
Reserved:  0.0 GB

tensor([[-0.4709,  0.2093,  0.0426, -1.0579]], device='cuda:0')
</pre>

<code>check_gpu.py</code> checks if GPUs are available and if available sets up the device to use them.

<pre>
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
</pre>

#### Batch Jobs

An example batch-job submission script is included below:

<pre>
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
module load cuda/11.1.0-fasrc01
source activate pt1.8_cuda111

# Run program
srun -c 1 --gres=gpu:1 python check_gpu.py 
</pre>

If you name the above batch-job submission script <code>run.sbatch</code>, for instance, the job is submitted with:

<pre>
$ sbatch run.sbatch
</pre>

### References:

* [Official PyTorch website](https://pytorch.org)
* [PyTorch Tutorials](https://pytorch.org/tutorials/)
