#!/bin/bash
#SBATCH -J ode_test
#SBATCH -o ode_test.out
#SBATCH -e ode_test.err
#SBATCH -p test
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 0-00:30
#SBATCH --mem=4G

# Run the program using Julia
julia ode_test.jl
