#!/bin/bash
#SBATCH -J multi_fig       # job name
#SBATCH -o multi_fig.out   # standard output file
#SBATCH -e multi_fig.err   # standard error file
#SBATCH -p test            # partition
#SBATCH -c 1               # number of cores
#SBATCH -t 0-00:30         # time in D-HH:MM
#SBATCH --mem=4000         # memory in MB

# Load required software modules
module load matlab
srun -c $SLURM_CPUS_PER_TASK matlab -nosplash -nodesktop -nodisplay -r "multi_fig"

