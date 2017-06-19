#!/bin/bash
#SBATCH -J count_down
#SBATCH -o count_down.out
#SBATCH -e count_down.err
#SBATCH -p serial_requeue
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 0-00:30
#SBATCH --mem=4000

# Load required software modules
source new-modules.sh
module load R/3.2.0-fasrc01

# Run program
srun -n 1 -c 1 R --vanilla < count_down.R 

