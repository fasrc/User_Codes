#!/bin/bash

# This script submits an array job on a single node with the 
# highest array index job launching the next set of job recursively.

# Usage:
#   sbatch recursive-array.sh multiplier-initial-value jobs-max-limit
#   For example:
#   sbatch recursive-array.sh 1 60000

#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=100M
#SBATCH --array=[0-9]

# Does not redirect output or error to files.

# Check if max limit for jobs is provided by the user or not
JOBLIMIT=$2
if [ -z "$JOBLIMIT" ] || [ $# -lt 2 ]
then
    echo "This script requires total number of jobs needed to be run as its 2nd command line argument"
    exit
fi

echo "JOBLIMIT is $JOBLIMIT"

if [ $JOBLIMIT -lt $SLURM_ARRAY_TASK_COUNT ]
then
    echo "The max limit for jobs needs to at least match the total number of array jobs being launched"
    exit
fi

# To keep track of Iteration, create a counter
MULTIPLIER=$1
if [ -z "$MULTIPLIER" ]
then
  MULTIPLIER=1
fi

WORKDIR=$PWD
echo "Main directory for recursive-array script is: $WORKDIR"

# Create a directory corresponding to each iteration,
# Slurm Job ID, & Slurm Array Index.
DIRNAME=d${MULTIPLIER}-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}
mkdir $DIRNAME

# Switch to that directory, capture output, and 
# execute your science there
echo "Entering directory $DIRNAME" 
pushd $DIRNAME
echo "In $DIRNAME" >> screenlog.out

echo "Multiplier is $MULTIPLIER" >> screenlog.out

# Determine my task-id
MY_TASK=$((MULTIPLIER * 10 + SLURM_ARRAY_TASK_ID))

echo "Job array ID: $SLURM_ARRAY_JOB_ID , sub-job $SLURM_ARRAY_TASK_ID is running!" >> screenlog.out
echo "Highest job array index value is $SLURM_ARRAY_TASK_MAX" >> screenlog.out
echo "Number of tasks in array job is $SLURM_ARRAY_TASK_COUNT" >> screenlog.out
echo "MY_TASK is $MY_TASK" >> screenlog.out

# Science happens here
python $WORKDIR/test_function_script-mj.py > test-array-express-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.out

# Switch back to the working directory
echo "Exiting current directory and moving back to directory of origin" >> screenlog.out
popd

# Quit when MULTIPLIER reaches USERLIMIT
if [ $MULTIPLIER -eq $((JOBLIMIT / SLURM_ARRAY_TASK_COUNT)) ]
then
  exit
fi

# Start next generation of iterations only when array index matches
# the highest array index
if [ $SLURM_ARRAY_TASK_ID -eq $SLURM_ARRAY_TASK_MAX ]
then
   echo "Continuing Next Iteration"
   sbatch recursive-array.sh $((MULTIPLIER + 1)) $JOBLIMIT
   ERROR=$?
   if [ $ERROR -ne 0 ] 
   then    
     echo "This iteration failed. Submit the script again: sbatch recursive-array.sh $((MULTIPLIER + 1)) $JOBLIMIT"
     exit
   fi
fi
