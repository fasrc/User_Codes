#!/bin/bash
#SBATCH -J knitro_test
#SBATCH -o knitro_test.out
#SBATCH -e knitro_test.err
#SBATCH -p test
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 30
#SBATCH --mem=40000

# --- Load required modules ---
module load matlab/R2022b-fasrc01
module load knitro/13.2.0-fasrc01

# --- Run the program ---
srun -n 1 -c 1 matlab -nosplash -nodesktop -nodisplay -r "knitro_test;exit"
