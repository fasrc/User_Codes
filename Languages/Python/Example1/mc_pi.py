#!/usr/bin/env python
"""
Program: mc_pi.py
         Monte-Carlo calculation of PI
"""
import random

N = 100000

pi = 3.1415926535897

count = 0
for i in range(N):
    x = random.random()
    y = random.random()
    z = x*x + y*y
    if z <= 1.0:
        count = count + 1

PI = 4.0*count/N

print( "Exact value of PI: {0:7.5f}".format(pi) )
print( "Estimate of PI: {0:7.5f}".format(PI) )
