## Purpose

This example illustrates using STATA in batch mode on the FASRC cluster at Harvard University.

For how to use STATA interactivly, see this other [Stata documentation](https://docs.rc.fas.harvard.edu/kb/stata-on-cluster/).

## Contents

* `test.do`: STATA do file
* `run.sbatch`: Batch job submission script for sending the single-core job to the queue.
* `hello_mp.do`: STATA multiprocessor (mp) do file
* `run_mp.sbatch`: Batch job submission script for sending the multi-core job to the queue.

## Single-core job

This example shows how to run a Stata do file in serial (i.e., sequential, single core).
 
### Example STATA do file

```
sysuse auto
describe
summarize
generate price2 = 2*price 
describe
exit
``` 
                       
### Batch-Job Submission Script

```bash
#!/bin/bash
#SBATCH -J my_stata_job       # job name
#SBATCH -o my_stata_job.out   # standard output file
#SBATCH -e my_stata_job.err   # standard error file
#SBATCH -p serial_requeue     # partition
#SBATCH -t 0-00:30            # time in D-HH:MM
#SBATCH -N 1                  # number of nodes
#SBATCH -c 1                  # number of cores
#SBATCH --mem=4000            # total memory in MB

# Load required modules
module load stata/17.0-fasrc01

# Run program
stata-se -b test.do
```

### Example Usage

```bash
sbatch run.sbatch
```

### Example Output

```
[jharvard@holylogin02 STATA]$ cat test.log

  ___  ____  ____  ____  ____ ©
 /__    /   ____/   /   ____/      17.0
___/   /   /___/   /   /___/       SE—Standard Edition

 Statistics and Data Science       Copyright 1985-2021 StataCorp LLC
                                   StataCorp
                                   4905 Lakeway Drive
                                   College Station, Texas 77845 USA
                                   800-STATA-PC        https://www.stata.com
                                   979-696-4600        stata@stata.com

Stata license: 32-user network perpetual
Serial number: 501706311472
  Licensed to: Harvard Research Computing
               Cambridge MA

Notes:
      1. Stata is running in batch mode.
      2. Unicode is supported; see help unicode_advice.
      3. Maximum number of variables is set to 5,000; see help set_maxvar.

. do "test.do"

. sysuse auto
(1978 automobile data)

. describe

Contains data from /n/sw/stata-17/ado/base/a/auto.dta
 Observations:            74                  1978 automobile data
    Variables:            12                  13 Apr 2020 17:45
                                              (_dta has notes)
-------------------------------------------------------------------------------
Variable      Storage   Display    Value
    name         type    format    label      Variable label
-------------------------------------------------------------------------------
make            str18   %-18s                 Make and model
price           int     %8.0gc                Price
mpg             int     %8.0g                 Mileage (mpg)
rep78           int     %8.0g                 Repair record 1978
headroom        float   %6.1f                 Headroom (in.)
trunk           int     %8.0g                 Trunk space (cu. ft.)
weight          int     %8.0gc                Weight (lbs.)
length          int     %8.0g                 Length (in.)
turn            int     %8.0g                 Turn circle (ft.)
displacement    int     %8.0g                 Displacement (cu. in.)
gear_ratio      float   %6.2f                 Gear ratio
foreign         byte    %8.0g      origin     Car origin
-------------------------------------------------------------------------------
Sorted by: foreign

. summarize

    Variable |        Obs        Mean    Std. dev.       Min        Max
-------------+---------------------------------------------------------
        make |          0
       price |         74    6165.257    2949.496       3291      15906
         mpg |         74     21.2973    5.785503         12         41
       rep78 |         69    3.405797    .9899323          1          5
    headroom |         74    2.993243    .8459948        1.5          5
-------------+---------------------------------------------------------
       trunk |         74    13.75676    4.277404          5         23
      weight |         74    3019.459    777.1936       1760       4840
      length |         74    187.9324    22.26634        142        233
        turn |         74    39.64865    4.399354         31         51
displacement |         74    197.2973    91.83722         79        425
-------------+---------------------------------------------------------
  gear_ratio |         74    3.014865    .4562871       2.19       3.89
     foreign |         74    .2972973    .4601885          0          1

. generate price2 = 2*price

. describe

Contains data from /n/sw/stata-17/ado/base/a/auto.dta
 Observations:            74                  1978 automobile data
    Variables:            13                  13 Apr 2020 17:45
                                              (_dta has notes)
-------------------------------------------------------------------------------
Variable      Storage   Display    Value
    name         type    format    label      Variable label
-------------------------------------------------------------------------------
make            str18   %-18s                 Make and model
price           int     %8.0gc                Price
mpg             int     %8.0g                 Mileage (mpg)
rep78           int     %8.0g                 Repair record 1978
headroom        float   %6.1f                 Headroom (in.)
trunk           int     %8.0g                 Trunk space (cu. ft.)
weight          int     %8.0gc                Weight (lbs.)
length          int     %8.0g                 Length (in.)
turn            int     %8.0g                 Turn circle (ft.)
displacement    int     %8.0g                 Displacement (cu. in.)
gear_ratio      float   %6.2f                 Gear ratio
foreign         byte    %8.0g      origin     Car origin
price2          float   %9.0g
-------------------------------------------------------------------------------
Sorted by: foreign
     Note: Dataset has changed since last saved.

. exit

end of do-file
```

## Multi-core job

This example shows how to run a Stata do file in multi-core mode (or
multiprocessor, mp).

### Example STATA do file

Note that you will have to change the number of processor in thei first line of
do file to match how many cores you request on the `run_mp.sbatch` file.

```
set processors 4
display "Hello, World!"
```

### Batch-Job Submission Script

```bash
#!/bin/bash
#SBATCH -J my_stata_job       # job name
#SBATCH -o my_stata_job.out   # standard output file
#SBATCH -e my_stata_job.err   # standard error file
#SBATCH -p serial_requeue     # partition
#SBATCH -t 0-00:30            # time in D-HH:MM
#SBATCH -N 1                  # number of nodes
#SBATCH -c 4                  # number of cores
#SBATCH --mem=4000            # total memory in MB

# Load required modules
module load stata/17.0-fasrc01

# Run program
stata-mp -b hello_mp.do
```

### Example Usage

```bash
sbatch run_mp.sbatch
```

### Example Output

```
[jharvard@holylogin02 STATA]$ cat hello_mp.log

  ___  ____  ____  ____  ____ ©
 /__    /   ____/   /   ____/      17.0
___/   /   /___/   /   /___/       MP—Parallel Edition

 Statistics and Data Science       Copyright 1985-2021 StataCorp LLC
                                   StataCorp
                                   4905 Lakeway Drive
                                   College Station, Texas 77845 USA
                                   800-STATA-PC        https://www.stata.com
                                   979-696-4600        stata@stata.com

Stata license: 32-user 64-core network perpetual
Serial number: 501706311472
  Licensed to: Harvard Research Computing
               Cambridge MA

Notes:
      1. Stata is running in batch mode.
      2. Unicode is supported; see help unicode_advice.
      3. More than 2 billion observations are allowed; see help obs_advice.
      4. Maximum number of variables is set to 5,000; see help set_maxvar.

. do "hello_mp.do"

. set processors 4
    The maximum number of processors or cores being used is 4.  It can be set
    to any number between 1 and 4.

. display "Hello, World!"
Hello, World!

.
end of do-file
```

## References

* [STATA website](http://www.stata.com)
* [STATA documentation (release 2014-3)](https://www.stata.com/features/documentation)

