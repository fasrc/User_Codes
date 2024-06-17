#!/bin/bash
#SBATCH -J count_down2        # job name
#SBATCH -o count_down.out     # standard output file
#SBATCH -e count_down.err     # standard error file
#SBATCH -p shared             # partition
#SBATCH -c 1                  # number of cores
#SBATCH -t 0-00:30            # time in D-HH:MM
#SBATCH --mem=4000            # memory in MB

# Load required software modules
module load R

# Option 1: run R program and keep output in standard output file 
#           count_down.out (specified above) and error messages in standard 
#           error file count_down.err (specified above)
Rscript --vanilla count_down.R

# Option 2: run R program and keep output and error messages in count_down.Rout
#Rscript --vanilla count_down.R > count_down.Rout 2>&1

# Option 3: run R program and keep output in count_down.Rout
#           and error messages in standard error file count_down.err 
#           (specified above)
#Rscript --vanilla count_down.R > count_down.Rout

