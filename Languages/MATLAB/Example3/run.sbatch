#!/bin/bash
#SBATCH -J multi_fig
#SBATCH -o multi_fig.out
#SBATCH -e multi_fig.err
#SBATCH -p test
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 0-00:30
#SBATCH --mem=4000

# Load required software modules
module load matlab/R2021a-fasrc01
srun -n 1 -c 1 matlab -nosplash -nodesktop -nodisplay -r "multi_fig"

