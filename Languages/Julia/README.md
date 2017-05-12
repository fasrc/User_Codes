#### Purpose:

This example illustrates **Julia** use on the Odyssey cluster. The specific example evaluates PI via Monte-Carlo method.

#### Contents:

* <code>pi\_monte\_carlo.jl</code>: Monte-Carlo computation of PI
* <code>run.sbatch</code>: Batch-job submission script

#### Julia code:

```julia
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Program: pi_monte_carlo.jl
#          Monte-Carlo calculation of PI
#
# Usage: julia pi_monte_carlo.jl
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
function montepi(n)
   R = 1.0
   s = 0
   for i = 1: n
      x = R * rand()
      y = R * rand()
      if x^2 + y^2 <= R^2
         s = s + 1
      end
   end
   return 4.0*s/n
end

# Main program
for n in 10.^(3:8)
    p = montepi(n)
    println("N = $n: PI = $p")
end
```

#### Example Batch-Job Submission Script:

```bash
#!/bin/bash
#SBATCH -J pi_monte_carlo
#SBATCH -o pi_monte_carlo.out
#SBATCH -e pi_monte_carlo.err
#SBATCH -p general
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 0-00:30
#SBATCH --mem=4000

# Load required software modules
source new-modules.sh
module load julia/0.4.6-fasrc01
srun -n 1 -c 1 julia pi_monte_carlo.jl
```

#### Example Usage:

```bash
sbatch run.sbatch
```

#### Example Output:

```
N = 1000: PI = 3.084
N = 10000: PI = 3.1336
N = 100000: PI = 3.14116
N = 1000000: PI = 3.139972
N = 10000000: PI = 3.140994
N = 100000000: PI = 3.1418198
```

