## Purpose

Program calculates sum of integers from 1 to 100.

### Compile

GNU compilers, e.g.,

```bash
module load gcc/12.2.0-fasrc01
gfortran -o sum.x sum.f90 -O2
``` 

Intel compilers, e.g.,

```bash
module load intel/23.2.0-fasrc01
ifort -o sum.x sum.f90 -O2
``` 

### Example batch-job submission script

```bash
#!/usr/bin/env bash
#SBATCH -J sum_test
#SBATCH -o sum_job.out
#SBATCH -e sum_job.err
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -p test
#SBATCH -t 0-00:30
#SBATCH --mem=4G

# Load required software modules
module load intel/23.2.0-fasrc01

# Run the program
./sum.x
```

### Submit job

```bash
sbatch run.sbatch
```

### Example output

```
The sum of integers from 1 to 100 is 5050.
STOP End of program.
```
