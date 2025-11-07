#!/bin/bash

# Check PyTorch version
python -c 'import torch;print(torch.__version__)'

# Check if CUDA is available
python -c 'import torch;print(torch.cuda.is_available())'

# Check number of GPU devices
python -c 'import torch;print(torch.cuda.device_count())'

# Check current GPU device
python -c 'import torch;print(torch.cuda.current_device())'

# Get the name of the GPU device
python -c 'import torch;print(torch.cuda.get_device_name(0))'