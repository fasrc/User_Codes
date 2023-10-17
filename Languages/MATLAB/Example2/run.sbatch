#!/bin/bash
#SBATCH -J serial_sum       # job name
#SBATCH -o serial_sum.out   # standard output file
#SBATCH -e serial_sum.err   # standard error file
#SBATCH -p test             # partition
#SBATCH -c 1                # number of cores
#SBATCH -t 0-00:30          # time in D-HH:MM
#SBATCH --mem=4000          # memory in MB

# Load required software modules
module load matlab
srun -c $SLURM_CPUS_PER_TASK matlab -nodesktop -nodisplay -nosplash -r "serial_sum(100); exit"

