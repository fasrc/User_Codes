#!/usr/bin/env bash
#SBATCH -J tex_demo
#SBATCH -o tex_demo.out
#SBATCH -e tex_demo.err
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -p serial_requeue
#SBATCH -t 0-00:30
#SBATCH --mem=4000

# Load required modules
source new-modules.sh
module load python/2.7.6-fasrc01
module load dvipng/1.14-fasrc01

# Run program
python tex_demo.py
