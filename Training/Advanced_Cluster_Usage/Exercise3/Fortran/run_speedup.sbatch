#!/bin/bash
#SBATCH -J speedup
#SBATCH -o speedup.out
#SBATCH -e speedup.err
#SBATCH -t 0-00:30
#SBATCH -p test
#SBATCH -c 1
#SBATCH --mem=4G

# --- Set up environment ---
module load python/3.10.13-fasrc01
source activate /n/home06/pkrastev/holystore01/conda/python-3.10_env #python-3.10_env

# --- Run the python code speedup.py ---
python speedup.py
