#!/bin/bash
#SBATCH -J job_array_test
#SBATCH -o output_%a.out
#SBATCH -e error_%a.err
#SBATCH -p serial_requeue
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 30
#SBATCH --mem=4000
#SBATCH --array=100,200,300

# Run application
./pro.x ${SLURM_ARRAY_TASK_ID}
