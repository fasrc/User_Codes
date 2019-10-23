#!/bin/bash
#=================================================
# Program: sum.sh
#          Sum up integers from 1 to N, where N
#          is read from the command line
#
# Example usage: sh sum.sh 100
#=================================================
n=$1
k=0
for i in `seq 1 $n`
do
    k=$(($k+$i))
done
echo -e "Sum of integers from 1 to $n is $k."
