#!/bin/bash
#SBATCH -J hdf5_test          # job name
#SBATCH -o hdf5_test.out      # standard output file
#SBATCH -e hdf5_test.err      # standard error file
#SBATCH -p serial_requeue     # partition name
#SBATCH -t 00:30:00           # time in HH:MM:SS
#SBATCH -N 1                  # number of nodes
#SBATCH -n 2                  # number of mpi tasks
#SBATCH --mem=4000            # memory in MB

# Load required modules (use same modules used for compilation)
module load gcc/14.2.0-fasrc01  openmpi/5.0.5-fasrc01 hdf5/1.14.4-fasrc01

# Run program
./hdf5_test.x
