#!/usr/bin/env python
"""
Program: serial_sum.py
         Returns the sum of integers from 1 through N
"""
import os
def serial_sum(x):
    k = 0
    s = 0
    while k < x:
        k = k + 1
        s = s + k
    return s

def main():
    N = int(os.environ['SLURM_ARRAY_TASK_ID'])
    res = serial_sum(N)
    print( "The sum of integers from 1 through {0:d} is {1:d}.".format(N, res) )
    
if __name__ == "__main__":
    main()
