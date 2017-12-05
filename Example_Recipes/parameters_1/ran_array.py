#!/usr/bin/env python
"""
Program: ran_array.py
         Generate a random vector
"""
import numpy as np
import scipy as sp

n = np.arange(100)
for i in n:
    r = np.random.random()
    print ( "%8.6f" % r )
