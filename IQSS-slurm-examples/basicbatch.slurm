#!/bin/bash
#SBATCH -n 1 # Number of cores requested; default: 1
#SBATCH -N 1 # Ensure that all cores are on one machine; default: unset- causes poor performance
#SBATCH -t 15 # Runtime in minutes; job killed after this amount of time; default: 10
#SBATCH -p serial_requeue # Partition to submit to; default: serial_requeue
#SBATCH --mem=100 # Memory shared across all cores in MB (see also –mem-per-cpu)
#SBATCH --open-mode=append; default: overwrite
#SBATCH -o %x_%j.out # Standard out goes to this file; default: slurm-JOBID.out
#SBATCH -e %x_%j.err # Standard err goes to this file; default: slurm-JOBID.err

# Will need to request more than 100MB for some jobs in this repo

# Commands to run go on next lines
# Start by loading module(s) for your application:
load     # put my application module(s) here

# Run the application


# To submit this job:
# sbatch basicbatch.slurm
