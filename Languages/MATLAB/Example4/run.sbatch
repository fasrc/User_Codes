#!/bin/bash
#SBATCH -J array_test           # job name
#SBATCH -o array_test_%a.out    # standard output file
#SBATCH -e array_test_%a.err    # standard error file
#SBATCH -p shared               # partition
#SBATCH -c 1                    # number of cores
#SBATCH -t 0-00:30              # time in D-HH:MM
#SBATCH --mem=4000              # memory in MB
#SBATCH --array=17-32           # array indices

# Load required modules
module load matlab

# Run program
srun -c $SLURM_CPUS_PER_TASK matlab -nosplash -nodesktop -nodisplay -r "serial_sum($SLURM_ARRAY_TASK_ID);exit"

