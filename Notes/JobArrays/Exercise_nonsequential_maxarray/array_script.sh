#!/bin/bash

# This script runs the array job spawned using the main.sh script on a
# single node on a partition to process entries with a given keyword
# in their name and are located inside a work directory. The keyword
# is provided as a command line argument in main.sh.

# This script uses OFFSET, provided as a command line argument, to
# operate on the corresponding line number in joblist.txt that stores
# all the entries matching the $KEYWORD criterion.

# Usage:                                                                                                                                              
# array_script.sh <input-folder-name> <type> <keyword> <output-folder-name> <offset>                                                                     
# For example: array_script.sh directories f job.sh output 0                        

#SBATCH -J max_array_sample
#SBATCH --partition=test  # Change partition name based on Cannon or FASSE and resources require                                  
#SBATCH --reservation=bootcamp_cpu_2023
#SBATCH -o %A_%a.o 
#SBATCH -e %A_%a.e 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --mem=4G 

# Declare 'BASEDIR' and 'WORKDIR' based on desired locations
BASEDIR=$PWD
WORKDIR=$PWD/$1
OUTDIR=$PWD/$4

# Check if the type for entries, file (f) or directory (d) is provided                                                               
TYPE=$2 
if [ -z "TYPE" ] || [ $# -lt 4 ] 
then 
    echo "The script needs the type for entries, files (f) or directory (d), to be specified."
    exit
fi

echo "type of entry is $TYPE" 

# Check if the keyword to search for desired input files is provided                                                                
# by the user or not                                                                                                                    
KEYWORD=$3
if [ -z "$KEYWORD" ] || [ $# -lt 4 ]
then
    echo "The script, main.sh, requires the keyword, to search for desired entries, as its 3rd command line argument."
    exit
fi

echo "keyword is $KEYWORD"

OFFSET=$5
if [ -z "$OFFSET" ]
then
  OFFSET=0
fi

# Find entries inside $WORKDIR using joblist.txt 
ENTRIES=$(wc -l joblist.txt | awk '{print $1}')
echo "Number of entries are $ENTRIES"

# If no more folders to process, then exit                                                                                                                                                                                                       
FOLDER_NUMBER=$((OFFSET + SLURM_ARRAY_TASK_ID))
if [ $((FOLDER_NUMBER)) -ge ${ENTRIES} ]
then
  exit
fi

echo "At Offset $OFFSET and folder number $FOLDER_NUMBER for task ID $SLURM_ARRAY_TASK_ID"

cd $BASEDIR
echo "In $BASEDIR"

entryname=$(awk "NR==$((FOLDER_NUMBER+1))" joblist.txt)

echo "Job array ID: $SLURM_ARRAY_JOB_ID , sub-job $SLURM_ARRAY_TASK_ID is running!"
echo "Sub-job $SLURM_ARRAY_TASK_ID is processing $entryname"

# Do science here
DIRNAME=$OUTDIR-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}
mkdir $DIRNAME
pushd $DIRNAME > /dev/null

$entryname
echo "$entryname" > $DIRNAME/output.log

popd > /dev/null

echo "Done processing $entryname"
