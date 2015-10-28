#!/usr/bin/env python
#=====================================================================
# Program: binary.py
# Purpose: Converts integers into binaries
# Run: python binary.py
#=====================================================================
for i in range(0,256):
    j = bin(i)[2:].rjust(16,'0')
    k = int(str(j),2)
    print j, k
quit()
