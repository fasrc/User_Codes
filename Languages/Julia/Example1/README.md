#### Purpose:

This example illustrates the use of Julia on the Harvard University FAS cluster. Specifically, it evaluates PI via a Monte-Carlo method.


#### Contents:

* <code>pi\_monte\_carlo.jl</code>: Julia source code
* <code>run.sbatch</code>: Batch-job submission script

#### Julia code:

```julia
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Program: pi_monte_carlo.jl
#          Monte-Carlo calculation of PI
#
# Usage: julia pi_monte_carlo.jl
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
function montepi(n::Int)
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
for i in 3: 8
    n = 10^i
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
#SBATCH -p shared
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 0-00:30
#SBATCH --mem=2G

# Load required software modules
module load julia/1.1.1-fasrc01
srun -n 1 -c 1 julia pi_monte_carlo.jl
```

#### Example Usage:

```bash
sbatch run.sbatch
```

#### Example Output:

```bash
$ cat pi_monte_carlo.out 
N = 1000: PI = 3.228
N = 10000: PI = 3.1528
N = 100000: PI = 3.13816
N = 1000000: PI = 3.142652
N = 10000000: PI = 3.1413172
N = 100000000: PI = 3.14176752
```

