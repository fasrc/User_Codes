#### Purpose:

Example code to illustrate R use on the FASRC cluster. The specific example prins out integers from 10 down to 1.

#### Contents:

* [`count_down.R`](count_down.R): R source code
* [`run.sbatch`](run.sbatch): Batch-job submission script
* [`count_down.Rout`](count_down.Rout) : Output file
* [`count_down.out`](count_down.out) : Output file

#### R source code:

```r
#===========================================================
# Program: count_down.R
#
# Run:     R --vanilla < count_down.R         
#===========================================================

# Function CountDown........................................
CountDown <- function(x)
{
  print( x )
  while( x != 0 )
  {
    Sys.sleep(1)
    x <- x - 1
    print( x )
  }
}

# Call CountDown............................................
CountDown( 10 )
```

#### Example Batch-Job Submission Script:

```bash
#!/bin/bash
#SBATCH -J count_down2        # job name
#SBATCH -o count_down.out     # standard output file
#SBATCH -e count_down.err     # standard error file
#SBATCH -p shared             # partition
#SBATCH -c 1                  # number of cores
#SBATCH -t 0-00:30            # time in D-HH:MM
#SBATCH --mem=4000            # memory in MB

# Load required software modules
module load R

# Option 1: run R program and keep output in standard output file 
#           count_down.out (specified above) and error messages in standard 
#           error file count_down.err (specified above)
Rscript --vanilla count_down.R

# Option 2: run R program and keep output and error messages in count_down.Rout
#Rscript --vanilla count_down.R > count_down.Rout 2>&1

# Option 3: run R program and keep output in count_down.Rout
#           and error messages in standard error file count_down.err 
#           (specified above)
#Rscript --vanilla count_down.R > count_down.Rout
```

#### Example Usage:

```bash
sbatch run.sbatch
```
#### Example Output:

Content of file `count_down.out`:

```
[1] 10
[1] 9
[1] 8
[1] 7
[1] 6
[1] 5
[1] 4
[1] 3
[1] 2
[1] 1
[1] 0
```

Content of file `count_down.Rout`:

```
[1] 10
[1] 9
[1] 8
[1] 7
[1] 6
[1] 5
[1] 4
[1] 3
[1] 2
[1] 1
[1] 0
```
