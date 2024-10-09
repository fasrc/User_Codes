#!/bin/bash

salloc --partition=test --time=02:00:00 --mem=8G --cpus-per-task=2
module load python
export PYTHONNOUSERSITE=yes
mamba create --name claude_env python -y
source activate claude_env
pip install anthropic
conda deactivate