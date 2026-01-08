# PARI/GP

<img src="Images/parigp.png" alt="pari-logo" width="200"/>

See our documentation on [User Docs for an overview of PARI/GP on Cannon](https://docs.rc.fas.harvard.edu/kb/parigp/).

## What is PARI/GP?

From [PARI/GP docs](https://pari.math.u-bordeaux.fr/): "PARI/GP is a cross
platform and open-source computer algebra system designed for fast computations
in number theory: factorizations, algebraic number theory, elliptic curves,
modular forms, *L* functions... It also contains a wealth of functions to
compute with mathematical entities such as matrices, polynomials, power series,
algebraic numbers, etc., and a lot of transcendental functions as well as
numerical sumation and integration routines.  PARI is also available as a C
library to allow for faster computations."


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
[`runscript_seq.sh` link](https://github.com/fasrc/User_Codes/blob/f3015bf9076bef15cee9b3d09cb7bcd36134778a/Applications/PariGP/runscript_seq.sh#L1-L15) and a the PARI/GP file [`seq_pari.gp` link](https://github.com/fasrc/User_Codes/blob/f3015bf9076bef15cee9b3d09cb7bcd36134778a/Applications/PariGP/seq_pari.gp#L1-L3):

Then, you can submit the job with the command:

```bash
sbatch runscript_seq.sh
```

### Parallel
Here, we show how to request an interactive job and run a `.gp` file interactively

```bash
[jharvard@boslogin04 ~]$ salloc -p test --mem 12G -t 01:00:00 -c 4
[jharvard@holy2c02302 ~]$ module load module load gcc/12.2.0-fasrc01  openmpi/4.1.4-fasrc01 pari/2.15.4-fasrc02
[jharvard@holy2c02302 ~]$ srun -n $SLURM_CPUS_PER_TASK --mpi=pmix gp < par_pari.gp
                  GP/PARI CALCULATOR Version 2.15.4 (released)
          amd64 running linux (x86-64/GMP-6.2.1 kernel) 64-bit version
                compiled: Aug  1 2023, gcc version 12.2.0 (GCC)
                             threading engine: mpi
                (readline v7.0 disabled, extended help enabled)

                     Copyright (C) 2000-2022 The PARI Group

PARI/GP is free software, covered by the GNU General Public License, and comes
WITHOUT ANY WARRANTY WHATSOEVER.

Type ? for help, \q to quit.
Type ?18 for how to get moral (and possibly technical) support.

parisize = 8000000, primelimit = 500000, nbthreads = 3
%2 = (x)->ispseudoprime(2^x-1)
apply comparison
cpu time = 917 ms, real time = 922 ms.
%4 = [1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
cpu time = 313 ms, real time = 313 ms.
%5 = [1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
select comparison
cpu time = 917 ms, real time = 920 ms.
%7 = [2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127, 521, 607, 1279, 2203, 2281]
cpu time = 312 ms, real time = 313 ms.
%8 = [2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127, 521, 607, 1279, 2203, 2281]
Goodbye!
```

If you would like to submit a batch job, you can use the file
[`runscript_par.sh` link](https://github.com/fasrc/User_Codes/blob/0837d5c42d02a487fce8e4e67e530335766d5d63/Applications/PariGP/runscript_par.sh#L1-L15) and a the PARI/GP file [`par_pari.gp` link](https://github.com/fasrc/User_Codes/blob/0837d5c42d02a487fce8e4e67e530335766d5d63/Applications/PariGP/par_pari.gp#L1-L11
):

Then, you can submit the job with the command:

```bash
sbatch runscript_par.sh
```