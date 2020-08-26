#!/bin/bash
#SBATCH -J multitask.example # A single jobname for the array of jobs
#SBATCH -n 5 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -p serial_requeue # Partition to submit to
#SBATCH --mem=4000 # 4GB
#SBATCH -t 15 # Runtime in minutes
#SBATCH --open-mode=append
#SBATCH -o %x_%A_%a.out # Standard output
#SBATCH -e %x_%A_%a.err # Standard error
# Commands to run go on next lines:

module purge > /dev/null 2>&1
module load gcc/7.1.0-fasrc01 R/3.5.0-fasrc01
module load intel/17.0.4-fasrc01 R/3.5.0-fasrc01
chmod u+x multitask.example.R 
./multitask.example.R "${SLURM_ARRAY_TASK_ID}"   # Run R hello world.

# To submit the job:
# sbatch --array=1-5 multitask.example.R.slurm
