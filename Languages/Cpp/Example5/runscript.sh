#!/bin/bash
#SBATCH -J dot_prod            # job name
#SBATCH -o dot_prod.out        # standard output file
#SBATCH -e dot_prod.err        # standard error file
#SBATCH -p serial_requeue      # partition
#SBATCH -c 1                   # number of cores
#SBATCH -t 0-00:30             # time in D-HH:MM
#SBATCH --mem=4000             # total memory

# load required modules
# (these must be the same modules that were used for compiling)
module load gcc

# run code
./dot_prod.x
