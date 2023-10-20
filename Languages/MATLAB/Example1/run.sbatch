#!/bin/bash
#SBATCH -J pi_monte_carlo       # job name
#SBATCH -o pi_monte_carlo.out   # standard output file
#SBATCH -e pi_monte_carlo.err   # standard error file
#SBATCH -p test                 # partition
#SBATCH -c 1                    # number of cores
#SBATCH -t 0-00:30              # time in D-HH:MM
#SBATCH --mem=4000              # memory in MB

# Load required software modules
module load matlab
srun -c $SLURM_CPUS_PER_TASK matlab -nosplash -nodesktop -nodisplay -r "pi_monte_carlo"

