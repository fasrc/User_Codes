#!/bin/bash
#SBATCH -J void_point            # job name
#SBATCH -o void_point.out        # standard output file
#SBATCH -e void_point.err        # standard error file
#SBATCH -p serial_requeue        # partition
#SBATCH -c 1                     # number of cores
#SBATCH -t 0-00:30               # time in D-HH:MM
#SBATCH --mem=4000               # total memory

# load required modules
# (these must be the same modules that were used for compiling)
module load gcc

# run code
./void_point.x
