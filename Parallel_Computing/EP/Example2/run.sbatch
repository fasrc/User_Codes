#!/bin/bash
#SBATCH -J test_job
#SBATCH -p test
#SBATCH -c 1
#SBATCH -t 00:10:00
#SBATCH --mem=4G
#SBATCH -o test_job.out
#SBATCH -e test_job.err

# Load software environment
module load R/4.3.1-fasrc01

input=serial_sum.R

# Execute code
for i in 100 200 300; do
   export inp=$i
   srun -n 1 -c 1 R CMD BATCH $input $input.${inp}.out
done

