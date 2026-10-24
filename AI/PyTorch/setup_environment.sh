#!/bin/bash

# These instructions set up a conda> environment with PyTorch version & CUDA version 

# Start an interactive job requesting GPUs
salloc -p gpu -t 0-06:00 --mem=8000 --gres=gpu:1

# Load required software modules
module load python

# Create a conda environment
mamba create -n pt2.14.1_cuda13.2 pip wheel -y

# Activate the conda environment
source activate pt2.14.1_cuda13.2

# Install PyTorch and cuda 13.2
pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu132

# Install additional Python packages (optional)
mamba install -c conda-forge numpy scipy pandas matplotlib seaborn h5py jupyterlab jupyterlab-spellchecker scikit-learn -y
