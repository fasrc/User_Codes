#!/usr/bin/env python
"""
Program: mc_pi.py
         Monte-Carlo calculation of PI
"""
import numpy as np
import numpy.random as ran

N = 100000

pi = np.pi

iseed = 99
ran.seed(iseed) # Random seed

count = 0
for i in range(N):
    x = ran.rand(1)
    y = ran.rand(1)
    z = x*x + y*y
    if z <= 1.0:
        count = count + 1

PI = 4.0*count/N

print( "Exact value of PI: {0:7.5f}".format(pi) )
print( "Estimate of PI: {0:7.5f}".format(PI) )
