#!/bin/bash
#SBATCH -J serial_sum
#SBATCH -o serial_sum.out
#SBATCH -e serial_sum.err
#SBATCH -p serial_requeue
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 0-00:30
#SBATCH --mem=4000

# Load required software modules
source new-modules.sh
module load matlab/R2016b-fasrc01
srun -n 1 -c 1 matlab -nodesktop -nodisplay -nosplash -r "serial_sum(100); exit"

