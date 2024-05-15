#!/bin/bash

#SBATCH --partition=test  # Change partition name based on Cannon or FASSE and resources require
#SBATCH --nodes 1
#SBATCH --cpus-per-task=1
#SBATCH -t 1:00:00
#SBATCH --mem=1G
#SBATCH --job-name="Fastqc_arrayjob"
#SBATCH --output=%A-%a.out
#SBATCH --error=%A-%a.err
#SBATCH --array=1-3%3 ## This will submit an array of 3 jobs, all 3 at the same time


#Run fastqc on each sample using the SLURM_ARRAY_TASK_ID environmental variable
singularity exec /cvmfs/singularity.galaxyproject.org/f/a/fastqc:0.12.1--hdfd78af_0 fastqc wgEncode${SLURM_ARRAY_TASK_ID}_Sub.fq 
