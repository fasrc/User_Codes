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
module load intel/17.0.4-fasrc01 R/3.5.0-fasrc01
chmod u+x ./hello.R
./hello.R    # Run R hello world.

# To submit the job:
# sbatch hello.R.slurm
