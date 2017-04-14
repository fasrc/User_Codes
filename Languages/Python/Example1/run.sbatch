#!/usr/bin/env bash
#SBATCH -J mc_pi
#SBATCH -o mc_pi.out
#SBATCH -e mc_pi.err
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -p serial_requeue
#SBATCH -t 0-00:30
#SBATCH --mem=4000

# Load required modules and activate conda environment
source new-modules.sh
module load python/3.6.0-fasrc01
source activate python3_env1

# Run program
python mc_pi.py
