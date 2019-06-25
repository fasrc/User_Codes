#!/bin/bash
#SBATCH -J ode_test
#SBATCH -o ode_test.out
#SBATCH -e ode_test.err
#SBATCH -p shared
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 0-00:30
#SBATCH --mem=4G

# Load required software modules
module load julia/1.1.1-fasrc01
srun -n 1 -c 1 julia ode_test.jl
