#!/bin/bash
#SBATCH -J thread_test
#SBATCH -o thread_test.out
#SBATCH -e thread_test.err
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -t 30
#SBATCH -p general
#SBATCH --mem=32000

# Load required software modules
source new-modules.sh
module load matlab/R2016a-fasrc01

# Run program
srun -n 1 -c 8 matlab-default -nosplash -nodesktop -nodisplay -r "thread_test;exit"

