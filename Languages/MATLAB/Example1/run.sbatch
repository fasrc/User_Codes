#!/bin/bash
#SBATCH -J pi_monte_carlo
#SBATCH -o pi_monte_carlo.out
#SBATCH -e pi_monte_carlo.err
#SBATCH -p test
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 0-00:30
#SBATCH --mem=4000

# Load required software modules
module load matlab/R2021a-fasrc01
srun -n 1 -c 1 matlab -nosplash -nodesktop -nodisplay -r "pi_monte_carlo"

