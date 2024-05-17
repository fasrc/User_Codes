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
module load python 
source activate 

# Run the python code on a particular list, where each list to be
# processed could be read from a file (all-lists).  Each job task
# array would process a row corresponding to its job ID and the output
# of the python code would be redirected to its corresponding output
# file.
list_name=`sed "${SLURM_ARRAY_TASK_ID}q;d" all-lists`  ## get current row from file
python getMax.py $list_name > output.$list_name        ## process current filename
