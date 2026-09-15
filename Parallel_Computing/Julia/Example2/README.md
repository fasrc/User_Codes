This example demonstrates Monte Carlo computation of pi by MPI simulation.

### Contents
- <code>compute_pi.jl</code>: Julia source code
- <code>compute_pi.sh</code>: slurm submission script.

### Example Usage:

```bash
sbatch compute_pi.sh
```
 
### Example Output:

```bash
$ cat compute_pi.out
Estimate of pi is: 3.14167883
```