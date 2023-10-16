#!/bin/bash
#SBATCH -J np_pandas        # job name
#SBATCH -o np_pandas.out    # standard output file
#SBATCH -e np_pandas.err    # standard error file
#SBATCH -c 1                # number of cores 
#SBATCH -p test             # partition
#SBATCH -t 0-00:30          # time in D-HH:MM
#SBATCH --mem=12G           # memory in GB

# Load required modules
module load python
mamba activate my_env

# Run program
python numpy_pandas_ex.py
