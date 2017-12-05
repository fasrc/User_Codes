#### PURPOSE:

Example workflow illustrating running the same code over a set of data files. This solution uses a bash-shell script to iterate over the data files. The specific example uses 5 data files.

#### CONTENTS:

(1) <code>ran_array.py</code>: Python source code that creates a random vector of dimension 100.

(2) <code>test.py</code>: Python source code that reads a random vector from an external file  and computes the sum of vector elements. It takes as an argument the name of the data file.

(3) <code>test.sh</code>: Bash shell script used to iterate over individual data files.

(4) <code>run.sbatch</code>: Batch job submission script for sending the job to the queue.

(5) <code>file\_1.txt</code>, <code>file\_2.txt</code>, <code>file\_3.txt</code>, <code>file\_4.txt</code>, <code>file\_5.txt</code>: Input data files. 

#### Example Batch-Job Submission Script:

```bash
#!/usr/bin/env bash
#SBATCH -J test
#SBATCH -o test.out
#SBATCH -e test.err
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 0-00:30
#SBATCH -p shared
#SBATCH --mem=4000

# Load required modules
source new-modules.sh
module load python/2.7.6-fasrc01

# Run program
sh test.sh
```

#### EXAMPLE USAGE:

```
sbatch run.sbatch
```

#### EXAMPLE OUTPUT:

```
Iteration: 1
Sum of random vector elements:  46.3284

Iteration: 2
Sum of random vector elements:  55.0836

Iteration: 3
Sum of random vector elements:  55.0548

Iteration: 4
Sum of random vector elements:  48.6370

Iteration: 5
Sum of random vector elements:  53.1691
```
