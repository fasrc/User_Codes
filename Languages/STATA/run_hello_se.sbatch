#!/bin/bash
#SBATCH -J stata_hello        # job name
#SBATCH -o stata_hello.out    # standard output file
#SBATCH -e stata_hello.err    # standard error file
#SBATCH -p serial_requeue     # partition
#SBATCH -t 0-00:30            # time in D-HH:MM
#SBATCH -N 1                  # number of nodes
#SBATCH -c 1                  # number of cores
#SBATCH --mem=4000            # total memory in MB

# Load required modules
module load stata/17.0-fasrc01

# Run program
stata-se -b hello_se.do
