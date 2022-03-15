#!/bin/bash
#SBATCH -J array_test
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 0-00:30
#SBATCH -o array_test_%a.out
#SBATCH -e array_test_%a.err
#SBATCH -p test
#SBATCH --mem=4000
#SBATCH --array=1-3

# Load required modules
module load matlab/R2021a-fasrc01

# Seed for random number generator
iseed=$(($SLURM_ARRAY_JOB_ID+$SLURM_ARRAY_TASK_ID))
echo "iseed = $iseed"

# Run program
srun -n 1 -c 1 matlab -nosplash -nodesktop -nodisplay -r "rnd_test(10, -2, 2, $iseed); exit"
