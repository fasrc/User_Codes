#!/bin/bash
#SBATCH -J my_stata_job
#SBATCH -o my_stata_job.out
#SBATCH -e my_stata_job.err
#SBATCH -p general
#SBATCH -t 0-00:30
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000

# Load required modules
source new-modules.sh
module load stata/13.0-fasrc01

# Run STATA in parallel
stata-mp -b do test.do
