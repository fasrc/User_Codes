#!/bin/bash
#SBATCH -J sum_array
#SBATCH -o sum_array.out
#SBATCH -e sun_array.err
#SBATCH -p serial_requeue
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 0-00:30
#SBATCH --mem=4000

# Load required software modules
source new-modules.sh
module load perl/5.10.1-fasrc04
srun -n 1 -c 1 perl sum_array.pl << INPUT
100
INPUT

