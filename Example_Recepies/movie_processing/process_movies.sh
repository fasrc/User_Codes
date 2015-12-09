#!/bin/bash
#=====================================================================
# USAGE:   bash process_movies.sh <CVS_FILE_NAME> <MOVIE_EXTENTION>
# Example: bash process_movies.sh movie_info.csv mov
#===================================================================== 
ext=$2
label="10fps720p"
INPUT=$1
OLDIFS=$IFS
IFS=,
[ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }
while read filename timecode
do
    echo "Filename: $filename"
    echo "Timecode: $timecode"
    name1=${filename}
    base_name=${filename%.*}
    name2=${base_name}_${label}.${ext}
    echo ffmpeg -ss ${timecode} -i ${name1}
    echo ffmpeg -i ${name1} -c:v libx264 -r 10 -s 1280x720 -b:v 5000k \
	-threads 4 ${name2}
    echo ""
done < $INPUT
IFS=$OLDIFS
echo "All DONE."
