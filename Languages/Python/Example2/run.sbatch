#!/bin/bash
#SBATCH -J tex_demo
#SBATCH -o tex_demo.out
#SBATCH -e tex_demo.err
#SBATCH -c 1
#SBATCH -p serial_requeue
#SBATCH -t 0-00:30
#SBATCH --mem=4000

# Load required modules
module load python/3.9.12-fasrc01
module load texlive/2018.06.15-fasrc01

# Run program
python tex_demo.py
