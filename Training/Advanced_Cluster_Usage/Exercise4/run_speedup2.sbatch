#!/bin/bash
#SBATCH -J speedup_figure2
#SBATCH -o speedup_figure2.out
#SBATCH -e speedup_figure2.err
#SBATCH -p test
#SBATCH -t 30
#SBATCH -c 1
#SBATCH -n 1
#SBATCH --mem=4G

# Setup envirionment
module load python/3.10.13-fasrc01
source activate python-3.10_env

# Run program
python speedup_mpi2.py
