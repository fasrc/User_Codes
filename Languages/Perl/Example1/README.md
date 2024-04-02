#### Purpose:

Example code to illustrate Perl use on the FASRC cluster. The specific example generates a random vector of dimension N and sums up its elements.

#### Contents:

* <code>sum\_array.pl</code>: Perl source code
* <code>run.sbatch</code>: Batch-job submission script

#### Perl code:

```perl
#!/usr/bin/env perl
#==========================================================================
# Program: sum_array.pl
# Purpose: Creates a random array and sums up its elements
#          Array domension is supplied by the user.
#
# RUN:     perl sum_array.pl
#==========================================================================
print "Program generates a random array and prints out its elements.\n";
print "Please, enter array dimension: \n";
$N = <STDIN>;
for ( $i = 0; $i <= $N; $i++ ){
    $random_number= rand(); 
    $darr[$i] = $random_number;
#    print $darr[$i], "\n";
}
$isum = 0;
for ( $j = 0; $j <= $N; $j++ ){
    $isum = $isum + $darr[$j];
}
print "Array dimension: ", $N, "\n";
print "Sum of array elements: ", $isum, "\n";
```

#### Example Batch-Job Submission Script:

```bash
#!/bin/bash
#SBATCH -J sum_array
#SBATCH -o sum_array.out
#SBATCH -e sum_array.err
#SBATCH -p serial_requeue
#SBATCH -c 1
#SBATCH -t 0-00:30
#SBATCH --mem=4000

# Run program
perl sum_array.pl << INPUT
100
INPUT
```

#### Example Usage:

```bash
sbatch run.sbatch
```
#### Example Output:

```
Program generates a random array and prints out its elements.
Please, enter array dimension:
Array dimension: 100

Sum of array elements: 51.5883410109547
```
