#!/bin/bash
#SBATCH -J np_pandas        # job name
#SBATCH -o np_pandas.out    # standard output file
#SBATCH -e np_pandas.err    # standard error file
#SBATCH --cpus-per-task=1   # number of cores 
#SBATCH --partition=test    # partition
#SBATCH --time=0-01:00      # time in D-HH:MM
#SBATCH --mem=10G           # memory in GB

# Load required modules
module load python

# Build the environment
sh build_env.sh

# Activate the environment
mamba activate my_env

# Run program
python numpy_pandas_ex.py
