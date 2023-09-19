#!/bin/bash
#SBATCH -J ode_test
#SBATCH -o ode_test.out
#SBATCH -e ode_test.err
#SBATCH -p test
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 0-00:30
#SBATCH --mem=4G

# Set up Julia and run the program
#export PATH=$PATH:/n/holylabs/LABS/jharvard_lab/Users/jharvard/software/julia-1.9.3/bin
export PATH=$PATH:/n/home06/pkrastev/holyscratch01/Work/Julia/julia-1.9.3/bin
srun -n 1 -c 1 julia ode_test.jl
