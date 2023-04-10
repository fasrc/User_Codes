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
#SBATCH -p serial_requeue
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 0-00:30
#SBATCH --mem=4000

# Load required software modules
source new-modules.sh
module load mathematica/11.1.1-fasrc01
srun -n 1 -c 1 math -script pi_monte_carlo.m
```

#### Example Usage:

```bash
source new-modules.sh
module load mathematica/11.1.1-fasrc01
sbatch run.sbatch
```
#### Example Output:

```
cat pi_monte_carlo.out
3.141592653589793
3.1414188
58.931534`8.221892739513308
```
