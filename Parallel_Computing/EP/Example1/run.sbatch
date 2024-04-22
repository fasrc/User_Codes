#!/bin/bash
#SBATCH -J array_test
#SBATCH -p test
#SBATCH -c 1
#SBATCH -t 00:20:00
#SBATCH --mem=4G
#SBATCH -o %A-%a.o
#SBATCH -e %A-%a.e
#SBATCH --array=100,200,300

# Load software environment
module load python/3.10.13-fasrc01

# Execute code
srun -c 1 python serial_sum.py > output_${SLURM_ARRAY_TASK_ID}.out
