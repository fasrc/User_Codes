#!/bin/bash
#SBATCH -J arrays_and_pointers            # job name
#SBATCH -o arrays_and_pointers.out        # standard output file
#SBATCH -e arrays_and_pointers.err        # standard error file
#SBATCH -p serial_requeue                 # parition
#SBATCH -c 1                              # number of cores
#SBATCH -t 0-00:30                        # time in D-HH:MM
#SBATCH --mem=4000                        # total memory

# load required modules
# (these must be the same modules that were used for compiling)
module load gcc

# run code
./arrays_and_pointers.x
