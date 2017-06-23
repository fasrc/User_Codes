### Purpose:

Interpolation is a method of computing a new data point from a set of discrete set of known data points. This is frequently required in science and engineering, where one could have a number of data points obtained by measurement or simulation, representing the value of the function for a limited number of points of the independent variable. It is often required to estimate, or interpolate, the value of the function for a set of intermediate values of the independent variable.

The specific example estimates the Legendre polynomial, $P_5(x)$, for a set of values of the independent variable $x$ in the interval $[-1, 1]$.

### Contents:

* <code>lagint.f90</code>: Function to perform Lagrange interpolation
* <code>interp\_test.f90</code>: Fortran source code - driver for <code>lagint.f90</code>

### Compile:

**Intel Fortran Compiler (ifort):** 

```bash
ifort -o interp_test.x lagint.f90 interp_test.f90 -O2
```

**GNU Fortran Compiler (gfortran):** 

```bash
gfortran -o interp_test.x lagint.f90 interp_test.f90 -O2
```

### Example Output:

```
        x         P_5(x)      Exact
   -1.000000   -1.000000   -1.000000
   -0.777778    0.417209    0.417162
   -0.555556    0.041927    0.041914
   -0.333333   -0.333316   -0.333333
   -0.111111   -0.196459   -0.196464
    0.111111    0.196459    0.196464
    0.333333    0.333316    0.333333
    0.555556   -0.041927   -0.041914
    0.777778   -0.417209   -0.417162
    1.000000    1.000000    1.000000
```
