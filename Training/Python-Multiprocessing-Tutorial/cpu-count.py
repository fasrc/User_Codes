#!/usr/bin/env python3

# Based on main.py in 
# https://stackoverflow.com/questions/1006289/how-to-find-out-the-number-of-cpus-using-python/55423170#55423170

# In order to use multiple cores/CPUs, need to request a compute node using --cpus-per-task flag, as shown:
# salloc --partition=test --nodes=1 --cpus-per-task=10 --mem=10GB --time=00:30:00

import multiprocessing
import os

print(multiprocessing.cpu_count())
print(os.cpu_count())
print(len(os.sched_getaffinity(0)))
