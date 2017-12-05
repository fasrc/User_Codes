#!/usr/bin/env bash
#SBATCH -J test
#SBATCH -o test.out
#SBATCH -e test.err
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 0-00:30
#SBATCH -p shared
#SBATCH --mem=4000

# Load required modules
source new-modules.sh
module load python/2.7.6-fasrc01

# Run program
sh test.sh


