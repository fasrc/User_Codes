#!/bin/bash
#SBATCH -J se
#SBATCH -o se.out
#SBATCH -e se.err
#SBATCH -p general
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 30
#SBATCH --mem=16000

# Load required modules
source new-modules.sh
module load matlab/R2016a-fasrc01

# Run program
srun -n 1 -c 4 matlab-default -nosplash -nodesktop -nodisplay -r "prun"
