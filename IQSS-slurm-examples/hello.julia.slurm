#!/bin/bash
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 15 # Runtime in minutes
#SBATCH --mem=4000
#SBATCH -p serial_requeue # Partition to submit to
#SBATCH --open-mode=append
#SBATCH -o %x_%j.out # Standard out goes to this file
#SBATCH -e %x_%j.err # Standard err goes to this file
# Commands to run go on next lines:
module load julia/1.1.1-fasrc01
chmod u+x ./hello.jl
./hello.jl

# To submit the job:
# sbatch hello.julia.slurm

