#!/bin/bash
#SBATCH -J parallel_monte_carlo        # job name
#SBATCH -o parallel_monte_carlo.out    # standard output file
#SBATCH -e parallel_monte_carlo.err    # standard error file
#SBATCH -c 8                           # number of cores (SLURM_CPUS_PER_TASK)
#SBATCH -t 00:30:00                    # time in HH:MM:SS
#SBATCH -p serial_requeue              # partition
#SBATCH --mem=14GB                     # memory in GB

# Load required software modules
module load matlab

# Run program
srun -c $SLURM_CPUS_PER_TASK matlab -nosplash -nodesktop -r parallel_monte_carlo
