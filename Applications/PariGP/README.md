# PARI/GP

<img src="Images/parigp.png" alt="pari-logo" width="200"/>

## What is PARI/GP?

From [PARI/GP docs](https://pari.math.u-bordeaux.fr/): "PARI/GP is a cross
platform and open-source computer algebra system designed for fast computations
in number theory: factorizations, algebraic number theory, elliptic curves,
modular forms, *L* functions... It also contains a wealth of functions to
compute with mathematical entities such as matrices, polynomials, power series,
algebraic numbers, etc., and a lot of transcendental functions as well as
numerical sumation and integration routines.  PARI is also available as a C
library to allow for faster computations."

## PARI/GP in the FASRC Cannon cluster

PARGI/GP is available through [FASRC
Lmod](https://docs.rc.fas.harvard.edu/kb/modules-intro/). There is one release
to run PARI/GP sequentially and another release to run in parallel through MPI.
You can distinguish which module is which by their module dependencies. For
example, as of August 2023, two `pari` modules are available:


```
[jharvard@boslogin04 ~]$ module spider pari

-------------------------------------------------------------------------------
  pari:
-------------------------------------------------------------------------------
    Description:
      PARI/GP is a cross platform and open-source computer algebra system
      designed for fast computations in number theory: factorizations, algebraic
      number theory, elliptic curves, modular forms, L functions.

     Versions:
        pari/2.15.4-fasrc01
        pari/2.15.4-fasrc02

-------------------------------------------------------------------------------
  For detailed information about a specific "pari" package (including how to
  load the modules) use the module's full name.  Note that names that have a
  trailing (E) are extensions provided by other modules.
  For example:

     $ module spider pari/2.15.4-fasrc02
-------------------------------------------------------------------------------
```

The pari module compiled with **MPI** can only be loaded when the compilers gcc
12 and openmpi 4.1.4 have been loaded as explained when you execute `module
spider pari/2.15.4-fasrc02`:

```
[jharvard@boslogin04 ~]$ module spider pari/2.15.4-fasrc02

-------------------------------------------------------------------------------
  pari: pari/2.15.4-fasrc02
-------------------------------------------------------------------------------
    Description:
      PARI/GP is a cross platform and open-source computer algebra system
      designed for fast computations in number theory: factorizations, algebraic
      number theory, elliptic curves, modular forms, L functions.


    You will need to load all module(s) on any one of the lines below before the
    "pari/2.15.4-fasrc02" module is available to load.

      gcc/12.2.0-fasrc01  openmpi/4.1.4-fasrc01

    Help:
      pari-2.15.4-fasrc02
      PARI/GP is a cross platform and open-source computer algebra system
      designed for fast computations in number theory: factorizations, algebraic
      number theory, elliptic curves, modular forms, L functions.
```

## Examples

### Sequential

Here, we show how to request an interactive job and launch PARI/GP:

```bash
[jharvard@boslogin04 ~]$ salloc -p test --mem 4G -t 01:00:00
[jharvard@holy2c02302 ~]$ module load pari/2.15.4-fasrc01
[jharvard@holy2c02302 ~]$ gp
                      GP/PARI CALCULATOR Version 2.15.4 (released)
              amd64 running linux (x86-64/GMP-6.1.2 kernel) 64-bit version
                 compiled: Aug  2 2023, gcc version 8.5.0 20210514 (GCC)
                                threading engine: single
                     (readline v7.0 enabled, extended help enabled)

                         Copyright (C) 2000-2022 The PARI Group

PARI/GP is free software, covered by the GNU General Public License, and comes WITHOUT ANY WARRANTY WHATSOEVER.

Type ? for help, \q to quit.
Type ?18 for how to get moral (and possibly technical) support.

parisize = 8000000, primelimit = 500000
? 47!
%1 = 258623241511168180642964355153611979969197632389120000000000
? log(2)
%2 = 0.69314718055994530941723212145817656807
? log(1+x)
%3 = x - 1/2*x^2 + 1/3*x^3 - 1/4*x^4 + 1/5*x^5 - 1/6*x^6 + 1/7*x^7 - 1/8*x^8 + 1/9*x^9 - 1/10*x^10 + 1/11*x^11 - 1/12*x^12 + 1/13*x^13 - 1/14*x^14 + 1/15*x^15 + O(x^16)
? quit
Goodbye!
```

If you would like to submit a batch job, you can use the file
`runscript_seq.sh` and a the PARI/GP file `seq_pari.gp`:

https://github.com/fasrc/User_Codes/blob/f3015bf9076bef15cee9b3d09cb7bcd36134778a/Applications/PariGP/runscript_seq.sh#L1-L15

https://github.com/fasrc/User_Codes/blob/f3015bf9076bef15cee9b3d09cb7bcd36134778a/Applications/PariGP/seq_pari.gp#L1-L3

Then, you can submit the job with the command:

```bash
sbatch runscript_seq.sh
```




