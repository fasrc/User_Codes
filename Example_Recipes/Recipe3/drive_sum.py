"""
Program: drive_sum.py
         Driver for sum.f90
"""
import subprocess
import numpy as np

for n in np.arange(1, 101):
    command = ["./sum.x", "-n", str(n)]
    subprocess.call( command )

