#!/bin/bash
#SBATCH -J pi_monte_carlo
#SBATCH -o pi_monte_carlo.out
#SBATCH -e pi_monte_carlo.err
#SBATCH -p test
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 0-00:30
#SBATCH --mem=2G

# Run the program using Julia
julia pi_monte_carlo.jl
