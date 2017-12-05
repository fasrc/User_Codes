#!/bin/bash 
i=0
for N in {1..5}
do
    i=$(($i+1))
    printf "Iteration: %d\n" "$i"
    python test.py file_${N}.txt
    printf "\n"
done
