#### Purpose:

This example illustrates using STATA on the Odyssey cluster at Harvard University.

#### Contents:

(1) test.do: Input file (STATA "do" file)

(2) run.sbatch: Batch job submission script for sending the job to the queue.
                       
#### Example Usage:

```bash
source new-modules.sh
module load stata/13.0-fasrc01
sbatch run.sbatch
```

#### Example Output:

```
[pkrastev@sa01 STATA]$ cat test.log

  ___  ____  ____  ____  ____ (R)
 /__    /   ____/   /   ____/
___/   /   /___/   /   /___/   13.1   Copyright 1985-2013 StataCorp LP
  Statistics/Data Analysis            StataCorp
                                      4905 Lakeway Drive
     MP - Parallel Edition            College Station, Texas 77845 USA
                                      800-STATA-PC        http://www.stata.com
                                      979-696-4600        stata@stata.com
                                      979-696-4601 (fax)

2-user 64-core Stata network perpetual license:
       Serial number:  501306209182
         Licensed to:  Harvard Research Computing
                       Cambridge, MA

Notes:
      1.  (-v# option or -set maxvar-) 5000 maximum variables
      2.  Command line editing disabled
      3.  Stata running in batch mode


****** Odyssey Notes ******

Please see this page for great tips on running stata more efficiently:

    http://nber.org/stata/efficient/

and great tips from our Odyssey community here:

    https://groups.google.com/forum/?utm_medium=email&utm_source=footer#!topic/
> odybot/SSkMrlEbYi4

***************************

. do test.do 

. sysuse auto
(1978 Automobile Data)

. describe

Contains data from /n/sw/stata-13/ado/base/a/auto.dta
  obs:            74                          1978 Automobile Data
 vars:            12                          13 Apr 2013 17:45
 size:         3,182                          (_dta has notes)
-------------------------------------------------------------------------------
              storage   display    value
variable name   type    format     label      variable label
-------------------------------------------------------------------------------
make            str18   %-18s                 Make and Model
price           int     %8.0gc                Price
mpg             int     %8.0g                 Mileage (mpg)
rep78           int     %8.0g                 Repair Record 1978
headroom        float   %6.1f                 Headroom (in.)
trunk           int     %8.0g                 Trunk space (cu. ft.)
weight          int     %8.0gc                Weight (lbs.)
length          int     %8.0g                 Length (in.)
turn            int     %8.0g                 Turn Circle (ft.)
displacement    int     %8.0g                 Displacement (cu. in.)
gear_ratio      float   %6.2f                 Gear Ratio
foreign         byte    %8.0g      origin     Car type
-------------------------------------------------------------------------------
Sorted by:  foreign

. summarize

    Variable |       Obs        Mean    Std. Dev.       Min        Max
-------------+--------------------------------------------------------
        make |         0
       price |        74    6165.257    2949.496       3291      15906
         mpg |        74     21.2973    5.785503         12         41
       rep78 |        69    3.405797    .9899323          1          5
    headroom |        74    2.993243    .8459948        1.5          5
-------------+--------------------------------------------------------
       trunk |        74    13.75676    4.277404          5         23
      weight |        74    3019.459    777.1936       1760       4840
      length |        74    187.9324    22.26634        142        233
        turn |        74    39.64865    4.399354         31         51
displacement |        74    197.2973    91.83722         79        425
-------------+--------------------------------------------------------
  gear_ratio |        74    3.014865    .4562871       2.19       3.89
     foreign |        74    .2972973    .4601885          0          1

. generate price2 = 2*price 

. describe

Contains data from /n/sw/stata-13/ado/base/a/auto.dta
  obs:            74                          1978 Automobile Data
 vars:            13                          13 Apr 2013 17:45
 size:         3,478                          (_dta has notes)
-------------------------------------------------------------------------------
              storage   display    value
variable name   type    format     label      variable label
-------------------------------------------------------------------------------
make            str18   %-18s                 Make and Model
price           int     %8.0gc                Price
mpg             int     %8.0g                 Mileage (mpg)
rep78           int     %8.0g                 Repair Record 1978
headroom        float   %6.1f                 Headroom (in.)
trunk           int     %8.0g                 Trunk space (cu. ft.)
weight          int     %8.0gc                Weight (lbs.)
length          int     %8.0g                 Length (in.)
turn            int     %8.0g                 Turn Circle (ft.)
displacement    int     %8.0g                 Displacement (cu. in.)
gear_ratio      float   %6.2f                 Gear Ratio
foreign         byte    %8.0g      origin     Car type
price2          float   %9.0g                 
-------------------------------------------------------------------------------
Sorted by:  foreign
     Note:  dataset has changed since last saved

. exit

end of do-file
```

#### REFERENCES:

* [STATA website](http://www.stata.com)
* [STATA documentation (release 2014-3)](https://www.stata.com/features/documentation)

