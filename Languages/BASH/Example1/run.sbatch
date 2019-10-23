#!/bin/bash
#SBATCH -J bash_sum
#SBATCH -o bash_sum.out
#SBATCH -e bash_sum.err
#SBATCH -p shared
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 0-00:30
#SBATCH --mem=2G

# Run the program
srun -n 1 -c 1 sh sum.sh 100
