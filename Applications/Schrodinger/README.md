#### PURPOSE:

Schrodinger is a scientific leader in computational chemistry, providing software solutions and services for life sciences and materials research.

This example illustrates using Schrodinger in a batch mode on the Odyssey cluster at Harvard University. The specific example computes properties of
water molecule using Jaguar.

#### CONTENTS:

(1) H20.in: Input data file

(2) run.sbatch: Batch job submission script for sending the job to the queue.
                       
#### EXAMPLE USAGE:

	source new-modules.sh
	module load schrodinger/2014.3-fasrc01
	sbatch run.sbatch


#### EXAMPLE OUTPUT:

The following files are generated after the job completes:

    H2O.01.in  H2O.01.mae  H2O.log  H2O.out  test_water.err  test_water.out

Below are contents of H2O.out:

```
[pkrastev@sa01 test2]$ cat H2O.out 
Job H2O started on holy2a13308.rc.fas.harvard.edu at Fri Apr 29 14:41:59 2016
jobid: holy2a13308-0-5723aaf5

  +--------------------------------------------------------------------+
  |  Jaguar version 8.5, release  13                                   |
  |                                                                    |
  |  Copyright Schrodinger, Inc.                                       |
  |  All Rights Reserved.                                              |
  |                                                                    |
  |  The following have contributed to Jaguar (listed alphabetically): |
  |  Mike Beachy, Art Bochevarov, Dale Braden, Yixiang Cao,            |
  |  Chris Cortis, Rich Friesner, Bill Goddard, Hod Greeley,           |
  |  Tom Hughes, Jean-Marc Langlois, Daniel Mainz, Rob Murphy,         |
  |  Dean Philipp, Tom Pollard, Murco Ringnalda.                       |
  |                                                                    |
  |  Use of this program should be acknowledged in publications as:    |
  |                                                                    |
  |  Jaguar, version 8.5, Schrodinger, Inc., New York, NY, 2014.       |
  |                                                                    |
  |  A. D. Bochevarov, E. Harder, T. F. Hughes, J. R. Greenwood,       |
  |  D. A. Braden, D. M. Philipp, D. Rinaldo, M. D. Halls,             |
  |  J. Zhang, R. A. Friesner, "Jaguar: A High-Performance Quantum     |
  |  Chemistry Software Program with Strengths in Life and Materials   |
  |  Sciences", Int. J. Quantum Chem., 2013, 113(18), 2110-2142.       |
  +--------------------------------------------------------------------+
  
  start of program pre
  
  
 --------echo of input .in file:-----------------
JOB: H2O
EXEC: /n/sw/schrodinger2014-3/jaguar-v85013/bin/Linux-x86_64
SCRATCH: /scratch/pkrastev/H2O
OUTPUT: /n/home06/pkrastev/workdirs/schrodinger/test2
RESTARTJOB: H2O.01
&gen
&
&echo
&
&zmat
O       0.0000000000000   0.0000000000000  -0.1135016000000
H1      0.0000000000000   0.7531080000000   0.4540064000000
H2      0.0000000000000  -0.7531080000000   0.4540064000000
&
MAEFILE: H2O.mae
 ------------end of .in file---------------------
  
  Job name: H2O
  Executables used: /n/sw/schrodinger2014-3/jaguar-v85013/bin/Linux-x86_64
  Temporary files : /scratch/pkrastev/H2O
  Maestro file (input):  H2O.mae
  Maestro file (output): H2O.01.mae
  
  
  basis set:             6-31g**         
  net molecular charge:    0
  multiplicity:            1
  
  number of basis functions....           25
  
 Input geometry:
                                   angstroms
  atom               x                 y                 z
  O             0.0000000000      0.0000000000     -0.1135016000 
  H1            0.0000000000      0.7531080000      0.4540064000 
  H2            0.0000000000     -0.7531080000      0.4540064000 
   
  principal moments of inertia:
        amu*angstrom^2:        0.57652        1.14322        1.71974
                g*cm^2: 9.57332390E-41 1.89836056E-40 2.85569295E-40
   
  rotational constants:
               cm^(-1):     29.24036473     14.74574897      9.80243631
                   GHz:    876.60408151    442.06643298    293.86964764
   
  Molecular weight:      18.01 amu
   
  Stoichiometry: H2O
  Molecular Point Group: C2v     
	Molecule translated to center of mass 
	Molecule reoriented along symmetry axes
  Point Group used: C2v
  
  
 Symmetrized geometry:
                                   angstroms
  atom               x                 y                 z
  O             0.0000000000      0.0000000000     -0.0635125859 
  H1            0.0000000000      0.7531080000      0.5039954141 
  H2            0.0000000000     -0.7531080000      0.5039954141 
   
  nuclear repulsion energy.......       9.330006048 hartrees
 
 Non-default options chosen:
  
 Temporary integer options set:
     73=  2
  
  Peak memory usage:    520 Mb
  
  end of program pre
   
 Time(pre)  user:        0.2  user+sys:        0.2  wallclock:        1.9
  
  start of program onee
  
  smallest eigenvalue of S:    1.979E-02
  number of canonical orbitals.....           25
  Peak memory usage:    660 Mb
  
  end of program onee
   
 Time(onee)  user:        0.2  user+sys:        0.2  wallclock:        0.3
  
  start of program hfig
  
  initial wavefunction generated automatically from atomic wavefunctions
 
  Irreducible     Total no   No of occupied orbitals 
  representation  orbitals   Shell_1  Shell_2    ...
  A1                12          3
  A2                 2          0
  B1                 4          1
  B2                 7          1
  ------------------------
  Orbital occupation/shell    1.000
 
  Peak memory usage:    599 Mb
  
  end of program hfig
   
 Time(hfig)  user:        0.1  user+sys:        0.1  wallclock:        0.3
  
  start of program probe
  
  Peak memory usage:    599 Mb
  
  end of program probe
   
 Time(probe)  user:        0.1  user+sys:        0.1  wallclock:        0.3
  
  start of program grid
  
  
   grid     grid set  grid #  grid sym
  ------    --------  ------  --------
  coarse        0        0        1
  medium        2        1        4
  fine          0        0        1
  ultrafine     4        2        4
  charge        0        0        1
  gradient      4        2        4
  density       0        0        1
  DFT-fine      0        0        1
  DFT-med.      0        0        1
  DFT-grad      0        0        1
  DFT-der2      0        0        1
  DFT-cphf      0        0        1
  LMP2-enrg     4        2        4
  LMP2-grad     2        1        4
  DFT-cphf2     0        0        1
  PBF-dens      0        0        1
  plotting      0        0        1
  Rel-grad    -17        0        1
  
   
  number of gridpoints: 
     atom         O       H1       H2    total
  grid # 1       40       69        0      109
  grid # 2      140      263        0      403
   
  Peak memory usage:    617 Mb
  
  end of program grid
   
 Time(grid)  user:        0.1  user+sys:        0.1  wallclock:        0.3
  
  start of program rwr
  
  Peak memory usage:    521 Mb
  
  end of program rwr
   
 Time(rwr)  user:        0.2  user+sys:        0.2  wallclock:        0.3
  
  start of program scf
  
 number of electrons..........         10
 number of alpha electrons....          5
 number of beta electrons.....          5
 number of orbitals, total....         25
 number of doubly-occ'd orbs..          5
 number of open shell orbs....          0
 number of occupied orbitals..          5
 number of virtual orbitals...         20
 number of hamiltonians.......          1
 number of shells.............          1
 SCF type: HF
  
       i  u  d  i  g                   
       t  p  i  c  r                                 RMS    maximum
       e  d  i  u  i                       energy  density   DIIS  
       r  t  s  t  d       total energy    change  change    error 
  
etot    1  N  N  5  M     -75.75389050468           1.3E-02  2.6E-01
etot    2  Y  Y  6  M     -76.00999984809  2.6E-01  3.2E-03  4.7E-02
etot    3  N  Y  2  U     -76.02240478774  1.2E-02  1.0E-03  1.7E-02
etot    4  Y  Y  6  M     -76.02340960900  1.0E-03  3.9E-04  5.3E-03
etot    5  Y  Y  6  M     -76.02355192636  1.4E-04  1.3E-04  9.7E-04
etot    6  N  Y  2  U     -76.02360634202  5.4E-05  3.6E-05  2.5E-04
etot    7  N  N  2  U     -76.02360707695  7.3E-07  0.0E+00  0.0E+00
  
  
 Energy components, in hartrees:
   (A)  Nuclear repulsion............       9.33000604793
   (E)  Total one-electron terms.....    -123.34015789160
   (I)  Total two-electron terms.....      37.98654476672
   (L)  Electronic energy............     -85.35361312488  (E+I)
   (N)  Total energy.................     -76.02360707695  (A+L)
  
 SCFE: SCF energy: HF       -76.02360707695 hartrees   iterations:   7
  
 
 HOMO energy:    -0.49762
 LUMO energy:     0.21539
 
  Orbital energies (hartrees)/symmetry label: 
   -20.55723 A1         -1.34648 A1         -0.71401 B2         -0.56819 A1     
    -0.49762 B1          0.21539 A1          0.30812 B2          1.01702 B2     
     1.09293 A1          1.13455 A1          1.16897 B1          1.29525 B2     
     1.41155 A1          1.80246 A2          1.82987 A1     
 
  Peak memory usage:    740 Mb
  
  end of program scf
   
 Time(scf)  user:        0.3  user+sys:        0.3  wallclock:        0.6

 Total cpu seconds     user:       0.401   user+sys:       0.401
  
 Total elapsed time:    8 seconds

Job H2O completed on holy2a13308.rc.fas.harvard.edu at Fri Apr 29 14:42:07 2016
[pkrastev@sa01 test2]$ cp run.sbatch /n/home06/pkrastev/Computer/User_Codes/Application/Schrodinger
[pkrastev@sa01 test2]$ cp H2O.in /n/home06/pkrastev/Computer/User_Codes/Application/Schrodinger
[pkrastev@sa01 test2]$ ls
H2O.01.in  H2O.01.mae  H2O.in  H2O.log  H2O.out  run.sbatch  test_water.err  test_water.out
```

#### REFERENCES:

* [Official Scrodinger documentation](http://www.schrodinger.com/supportdocs/18/)
* [Jaguar documentation (release 2014-3)] (https://hpc.nih.gov/apps/schrodinger/docs-2014-3/jaguar/jaguar_user_manual.pdf)

