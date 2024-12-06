#!/bin/bash
#SBATCH -J multi_proc        # job name
#SBATCH -o multi_proc.out    # standard output file
#SBATCH -e multi_proc.err    # standard error file
#SBATCH --cpus-per-task=1   # number of cores 
#SBATCH --partition=test    # partition
#SBATCH --time=0-01:00      # time in D-HH:MM
#SBATCH --mem=10G           # memory in GB

# Load required modules
module load python

# Build the environment
sh multiprocbuild_env.sh

# Activate the environment
mamba activate multiproc_env

# Install pip packages
pip install jupyterlab swifter
