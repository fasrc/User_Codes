### Purpose:

Example of using C on the Odyssey cluster.

### Contents:

* <code>sum.c</code>: Computes integer sum from 1 to N (N is read from command line)
* <code>run.sbatch</code>: Example batch-job submission script

### Compile:

GNU compilers, e.g.,

```bash
source new-modules.sh
module load gcc/6.3.0-fasrc01
gcc -o sum.x sum.c -O2
``` 

Intel compilers, e.g.,

```bash
source new-modules.sh
module load intel/17.0.2-fasrc01
icc -o sum.x sum.c -O2
``` 

### Example batch-job submission script:

```bash
#!/usr/bin/env bash
#SBATCH -J c_test_job
#SBATCH -o c_test_job.out
#SBATCH -e c_test_job.err
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 0-00:30
#SBATCH --mem=4000

# Load required software modules
source new-modules.sh
module load intel/17.0.2-fasrc01

# Run program
./sum.x 100
```

### Submit job:

```bash
sbatch run.sbatch
```

### Example output:

```
Welcome! This program prints out the sum of 1 to 100 
Sum of 1 to 100 is 5050 
End of program. 
```

### References:

* [C Tutorial (tutorialspoint.com)](https://www.tutorialspoint.com/cprogramming)
* [C Language Tutorial (Drexel U)](https://www.physics.drexel.edu/~valliere/General/C_basics/c_tutorial.html)
