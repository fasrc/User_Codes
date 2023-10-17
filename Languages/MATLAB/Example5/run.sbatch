#!/bin/bash
#SBATCH -J array_test_rnd          # job name
#SBATCH -o array_test_rnd_%a.out   # standard output file
#SBATCH -e array_test_rnd_%a.err   # standard error file
#SBATCH -p test                    # partition
#SBATCH -c 1                       # number of cores
#SBATCH -t 0-00:30                 # time in D-HH:MM
#SBATCH --mem=4000                 # memory in MB
#SBATCH --array=1-3                # array indices

# Load required modules
module load matlab

# Seed for random number generator
iseed=$(($SLURM_ARRAY_JOB_ID+$SLURM_ARRAY_TASK_ID))
echo "iseed = $iseed"

# Run program
srun -c $SLURM_CPUS_PER_TASK matlab -nosplash -nodesktop -nodisplay -r "rnd_test(10, -2, 2, $iseed); exit"
