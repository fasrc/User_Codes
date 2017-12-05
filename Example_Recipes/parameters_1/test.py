#!/usr/bin/env python
"""
Program: test.py
         Sums up elements of a vector
"""
import sys
import numpy as np

infile = sys.argv[1]
darr = np.loadtxt(infile, unpack=True)
r = np.sum(darr)
print ("Sum of random vector elements: %8.4f" % r)

