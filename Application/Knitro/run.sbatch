#!/bin/bash
#SBATCH -J knitro_test
#SBATCH -o knitro_test.out
#SBATCH -e knitro_test.err
#SBATCH -p serial_requeue
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 30
#SBATCH --mem=40000

# Load required modules
source new-modules.sh
module load matlab/R2015b-fasrc01
module load knitro/9.1.0-fasrc01

# Run Schrodinger
matlab -nosplash -nodesktop -nodisplay -r "knitro_test;exit"
