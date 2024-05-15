#!/bin/bash

# Bash script to list the number of entries with a given keyword to be
# processed that are located inside a working directory. The script
# accepts this keyword as a command line argument. It then submits a
# Slurm job as an array job of array size equal to the number of
# entries to be processed with a batch script.

# Usage:
# ./main.sh input-folder-name type keyword output-folder-name
# For example: ./main.sh directories f job.sh output

# Change 'WORKDIR' based on desired location
WORKDIR=$PWD/$1

echo "Workdir is $WORKDIR"

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

# Count number of entries inside $WORKDIR with keyword in their
# names
ENTRIES=$(find $WORKDIR -type $TYPE -name "*$KEYWORD*" | wc -l)

# Find entries inside $WORKDIR with given keyword in their names and                                                              
# redirect the output to joblist.txt                                                                                                    
rm -f joblist.txt
find $WORKDIR -type $TYPE -name "*$KEYWORD*" 2> /dev/null > joblist.txt

echo "Number of entries are $ENTRIES"

# ENTRIES is the number of files/folders that need to be processed
# Each iteration of this loop schedules a sbatch with an array size of
# LIMIT. Thus, each iteration of this loop will process ENTRIES,
# compare it to MAX_ARRAY_SIZE, and assign the leser value to
# LIMIT. The OFFSET and ENTRIES will be updated based on their
# original values and the value of LIMIT.
OFFSET=0
MAX_ARRAY_SIZE=1000
while [[ $ENTRIES -gt 0 ]]
do
    LIMIT=$(( ENTRIES > MAX_ARRAY_SIZE ? MAX_ARRAY_SIZE - 1 : ENTRIES - 1 ))
    sbatch --array=0-$LIMIT array_script.sh $1 $2 $3 $4 $OFFSET
    OFFSET=$((OFFSET + (LIMIT + 1) ))
    ENTRIES=$(( ENTRIES - (LIMIT + 1) ))
done
