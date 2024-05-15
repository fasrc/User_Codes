#!/bin/bash

# This script runs the array job spawned using the main.sh script on a
# single node on test partition to process folders starting with
# name sub-blast located inside a work directory.

#SBATCH -J slurm_python
#SBATCH --partition=test  # Change partition name based on Cannon or FASSE and resources required 
#SBATCH -o %A_%a.o 
#SBATCH -e %A_%a.e 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --mem=4G 


# Change 'BASEDIR' and 'WORKDIR' based on desired locations
BASEDIR=$PWD
WORKDIR=$PWD/masks

# Find sub-directories inside $WORKDIR with 'sub' in their names and
# redirect the output to filelist.txt
find $WORKDIR -type d -name "sub*" > filelist.txt

cd $BASEDIR
echo "In $BASEDIR"

dirname=$(awk "NR==${SLURM_ARRAY_TASK_ID}" filelist.txt)

echo "Job array ID: $SLURM_ARRAY_JOB_ID , sub-job $SLURM_ARRAY_TASK_ID is running!"
echo "Sub-job $SLURM_ARRAY_TASK_ID is processing $dirname"

# Do science here

echo "Done processing $dirname"
