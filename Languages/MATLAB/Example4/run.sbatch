#!/bin/bash
#SBATCH -J array_test
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 0-00:30
#SBATCH -o array_test_%a.out
#SBATCH -e array_test_%a.err
#SBATCH -p serial_requeue
#SBATCH --mem=4000
#SBATCH --array=17-32

# Load required modules
module load matlab/R2021a-fasrc01

# Run program
srun -n 1 -c 1 matlab -nosplash -nodesktop -nodisplay -r "serial_sum($SLURM_ARRAY_TASK_ID);exit"

