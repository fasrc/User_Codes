#### Purpose:

Example code to illustrate IDL use on the FASRC cluster. The specific example computes PI via Monte-Carlo method.

#### Contents:

* <code>pi\_monte\_carlo.pro</code>: IDL source code
* <code>run.sbatch</code>: Batch-job submission script

#### Example Batch-Job Submission Script:

```bash
#!/bin/bash
#SBATCH -J pi_monte_carlo
#SBATCH -o pi_monte_carlo.out
#SBATCH -e pi_monte_carlo.err
#SBATCH -p serial_requeue
#SBATCH -c 1
#SBATCH -t 0-00:30
#SBATCH --mem=4000

module load IDL/8.5.1-fasrc01
idl -e pi_monte_carlo
```

#### Example Usage:

```bash
module load IDL/8.5.1-fasrc01
sbatch run.sbatch
```
#### Example Output:

```
cat pi_monte_carlo.out 
Computed PI:       3.1435720
Exact PI:      3.14159
```

