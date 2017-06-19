#!/bin/bash
#SBATCH -J my_stata_job
#SBATCH -o my_stata_job.out
#SBATCH -e my_stata_job.err
#SBATCH -p serial_requeue
#SBATCH -t 0-00:30
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=4000

# Load required modules
source new-modules.sh
module load stata/13.0-fasrc01

# Run program
srun -n 1 -c 1 stata -b do test.do
