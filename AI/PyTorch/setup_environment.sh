#!/bin/bash

# These instructions set up a conda> environment with PyTorch version & CUDA version 

# Start an interactive job requesting GPUs
salloc -p gpu -t 0-06:00 --mem=8000 --gres=gpu:1

# Load required software modules
module load python/3.10.13-fasrc01

# Create a conda environment
mamba create -n pt2.3.0_cuda12.1 python=3.10 pip wheel

# Activate the conda environment
source activate pt2.3.0_cuda12.1

# Install CUDA toolkit version 12.1.0
mamba install -c "nvidia/label/cuda-12.1.0" cuda-toolkit=12.1.0

# Install PyTorch
mamba install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install additional Python packages
mamba install -c conda-forge numpy scipy pandas matplotlib seaborn h5py jupyterlab jupyterlab-spellchecker scikit-learn
