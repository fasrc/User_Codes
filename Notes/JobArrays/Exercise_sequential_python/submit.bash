#!/bin/bash

#SBATCH -J getMax 		##set job name 
#SBATCH --partition=test # Change partition name based on Cannon or FASSE and resources require 
#SBATCH --nodes=1		#set node number to 1
#SBATCH --ntasks=1		#set number of tasks (CPUs) to 1
#SBATCH --array=1-4%4 		#create 4 array jobs, run all 4 at a time.
#SBATCH --output=%A-%a.out	#set output filename with main job ID and task array ID
#SBATCH --error=%A-%a.err	#set error filename with main job ID and task array ID

# Purge any existing modules on the compute node to get a clean
# environment for running your program:
module purge

# Load a python environment
module load discovery anaconda3/2021.05 
source activate 

# Run the python code on a particular list, output into a particular file:
python getMax.py list$SLURM_ARRAY_TASK_ID > output.list$SLURM_ARRAY_TASK_ID
