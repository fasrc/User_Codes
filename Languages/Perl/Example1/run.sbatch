#!/bin/bash
#SBATCH -J sum_array
#SBATCH -o sum_array.out
#SBATCH -e sun_array.err
#SBATCH -p serial_requeue
#SBATCH -c 1
#SBATCH -t 0-00:30
#SBATCH --mem=4000

# Run program
perl sum_array.pl << INPUT
100
INPUT

