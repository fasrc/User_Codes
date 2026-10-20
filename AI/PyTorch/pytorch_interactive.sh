#!/bin/bash 
# Start an interactive session
salloc -p gpu -t 0-06:00 --mem=8000 --gres=gpu:1

# Load required modules and activate conda environment
module load python
source activate pt2.14.1_cuda13.2

# Test PyTorch interactively
python check_gpu.pyp
