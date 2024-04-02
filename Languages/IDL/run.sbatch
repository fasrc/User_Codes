#!/bin/bash
#SBATCH -J pi_monte_carlo
#SBATCH -o pi_monte_carlo.out
#SBATCH -e pi_monte_carlo.err
#SBATCH -p serial_requeue
#SBATCH -c 1
#SBATCH -t 0-00:30
#SBATCH --mem=4000

# Load required software modules
module load IDL/8.5.1-fasrc01
idl -e pi_monte_carlo
