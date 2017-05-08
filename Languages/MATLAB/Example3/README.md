#### Purpose:

MATLAB example code illustrating multi-figures in MATLAB in batch mode (without GUI). Please note the command line options <code>-nosplash -nodesktop -nodisplay</code>. The specific example generates the figure <code>figure.png</code>

#### Contents:

* <code>multi\_fig.m</code>: MATLAB source code
* <code>run.sbatch</code>: Batch-job submission script
* <code>figure.png</code>: Output figure

#### Example Batch-Job Submission Script:

```bash
#!/bin/bash
#SBATCH -J multi_fig
#SBATCH -o multi_fig.out
#SBATCH -e multi_fig.err
#SBATCH -p serial_requeue
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 0-00:30
#SBATCH --mem=4000

# Load required software modules
source new-modules.sh
module load matlab/R2016b-fasrc01
srun -n 1 -c 1 matlab -nosplash -nodesktop -nodisplay -r "multi_fig"
```

#### Example Usage:

```bash
source new-modules.sh
module load matlab/R2016b-fasrc01
sbatch run.sbatch
```

#### Example Output:

![figure.png](figure.png)
