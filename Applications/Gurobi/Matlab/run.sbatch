#!/bin/bash
#SBATCH -J gurobi_test          # job name
#SBATCH -o gurobi_test.out      # standard output file
#SBATCH -e gurobi_test.err      # standard error file
#SBATCH -p test                 # partition
#SBATCH -c 1                    # number of cores
#SBATCH -t 0-00:30              # time in D-HH:MM
#SBATCH --mem=4000              # memory in MB

# --- Load required software modules ---
module load matlab/R2025b-fasrc01 gurobi/12.0.1-fasrc01

# --- Run the Matlab code ---
srun -c $SLURM_CPUS_PER_TASK matlab -nosplash -nodesktop -nodisplay -r "mip1; exit;"
