#!/bin/bash
#SBATCH -J array_test
#SBATCH -p shared
#SBATCH -c 1
#SBATCH -t 00:10:00
#SBATCH --mem=4G
#SBATCH -o %A-%a.o
#SBATCH -e %A-%a.e
#SBATCH --array=100,200,300

# Load software environment
module load R/3.5.1-fasrc01

input=serial_sum.R

# Execute code
srun -n 1 -c 1 R CMD BATCH $input $input.$SLURM_ARRAY_TASK_ID.out 
