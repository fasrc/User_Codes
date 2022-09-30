#!/bin/bash
#SBATCH -J knitro_test
#SBATCH -o knitro_test_python.out
#SBATCH -e knitro_test_python.err
#SBATCH -p test
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 30
#SBATCH --mem=40000

# --- Load required modules ---
module load knitro/10.1.2-fasrc01

# --- Run the program ---
srun -n 1 -c 1 python knitro_test_python.py
