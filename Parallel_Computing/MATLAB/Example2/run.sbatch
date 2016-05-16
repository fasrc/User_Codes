#!/bin/bash
#SBATCH -J thread_test
#SBATCH -o thread_test.out
#SBATCH -e thread_test.err
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 30
#SBATCH -p general
#SBATCH --mem=32000

# Load required software modules
source new-modules.sh
module load matlab/R2015b-fasrc01

# Run program
matlab-default -nosplash -nodesktop -nodisplay -r "thread_test;exit"

