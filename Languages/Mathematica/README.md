#### Purpose:

Example code to illustrate **Mathematica** use on the FASRC cluster. The specific example computes PI via Monte-Carlo method.

#### Contents:

* <code>pi\_monte\_carlo.m</code>: Mathematica source code
* <code>run.sbatch</code>: Batch-job submission script

#### Example Batch-Job Submission Script:

```bash
#!/bin/bash
#SBATCH -J pi_monte_carlo
#SBATCH -o pi_monte_carlo.out
#SBATCH -e pi_monte_carlo.err
#SBATCH -p test
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 0-00:30
#SBATCH --mem=4000

# Load required software modules
module load mathematica/13.3.0-fasrc01
srun -n 1 -c 1 math -script pi_monte_carlo.m
```

#### Example Usage:

```bash
module load mathematica/13.3.0-fasrc01
sbatch run.sbatch
```
#### Example Output:

```
$ cat pi_monte_carlo.out
3.141592653589793
3.1411491999999996
19.342046`7.738047405306663
```
