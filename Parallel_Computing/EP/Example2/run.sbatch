#!/bin/bash
#SBATCH -J test_job
#SBATCH -p testg
#SBATCH -c 1
#SBATCH -t 00:20:00
#SBATCH --mem=4G
#SBATCH -o test_job.out
#SBATCH -e test_job.err

# Load software environment
module load python/3.10.13-fasrc01

# Execute code
for i in 100 200 300; do
   export inp=$i
   srun -n 1 -c 1 python serial_sum.py > output_${inp}.out
done

