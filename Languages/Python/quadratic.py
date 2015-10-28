#!/usr/bin/env python
#=====================================================================
# Program: quadratic.py
# Purpose: Solving Quadratic Equations with Python (real roots only)
# Run: python quadratic.py
#=====================================================================
from math import sqrt

# Get the input.......................................................
print " This program solves quadratic equation of the form "
print " ax**2 + bx + c = 0 "
print " Enter a "
a = float(input())
print " Enter b "
b = float(input())
print " Enter c "
c = float(input())

d = b**2 - ( 4.0 * a * c )

if d > 0.0:
    print " Case 1:  D > 0 "
    print " There are 2 different real roots. "
    x1 = ( -b + sqrt( d ) ) / ( 2.0 * a )
    x2 = ( -b - sqrt( d ) ) / ( 2.0 * a )
    print ' x1 = ', x1
    print ' x2 = ', x2
elif d == 0.0:
    print " Case 2: D = 0 "
    print " There are 2 equal real roots. "
    x = -b / ( 2.0 * a )
    print ' x1 = x2 = ', x
else:
    print " Case 3: D < 0 "
    print " There are no REAL roots in this case."
    print " There are COMPLEX roots only."

print "End of program."

quit()
