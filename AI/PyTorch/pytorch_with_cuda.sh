#!/bin/bash

# PyTorch with CUDA 11.8 from a software module

# Start an interactive job on a GPU node
salloc -p gpu -t 0-06:00 --mem=8000 --gres=gpu:1

# Load required modules
module load python
module load cuda/11.8.0-fasrc01

# Create a conda environment and activate it
mamba create -n pt2.2.0_cuda11.8 python=3.10 pip wheel -y
source activate pt2.2.0_cuda11.8

# Install PyTorch
mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install additional packages
mamba install pandas scikit-learn matplotlib seaborn jupyterlab -y