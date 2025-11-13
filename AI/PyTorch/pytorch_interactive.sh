#!/bin/bash 
# Start an interactive session
salloc -p gpu -t 0-06:00 --mem=8000 --gres=gpu:1

# Load required modules and activate conda environment
module load python/3.10.12-fasrc01
source activate pt2.3.0_cuda12.1

# Test PyTorch interactively
python check_gpu.pyp