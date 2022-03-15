#!/bin/bash
#SBATCH -J serial_sum
#SBATCH -o serial_sum.out
#SBATCH -e serial_sum.err
#SBATCH -p test
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 0-00:30
#SBATCH --mem=4000

# Load required software modules
module load matlab/R2021a-fasrc01
srun -n 1 -c 1 matlab -nodesktop -nodisplay -nosplash -r "serial_sum(100); exit"

