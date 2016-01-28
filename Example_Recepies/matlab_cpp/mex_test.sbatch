#!/bin/bash
#SBATCH -J mex_test
#SBATCH -o mex_test.out
#SBATCH -e mex_test.err
#SBATCH -p serial_requeue
#SBATCH -t 30
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=4000

# Load required modules
source new-modules.sh
module load matlab/R2015a-fasrc01
module load gcc/4.7.4-fasrc01
module load gsl/2.1-fasrc02

# Run MATLAB
matlab -nosplash -nodesktop -nodisplay -r "mex_test"
