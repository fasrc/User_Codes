#!/bin/bash
#SBATCH --job-name=Rfuture
#SBATCH --output=test_%j.out
#SBATCH --error=test_%j.err
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=4000
#SBATCH -t 0-00:10               # Runtime in D-HH:MM, minimum of 10 minutes

# load modules
module load R/4.4.1-fasrc01

# set R lirbary path
export R_LIBS_USER=$HOME/apps/R/4.4.1

### slow_square_multisession.R 
# run R program and keep output and error messages in slow_square_multisession.Rout
Rscript --vanilla future_slow_square.R > future_slow_square.Rout 2>&1 

# run R program and keep output in future_slow_square.Rout
# and error messages in error file
#Rscript --vanilla future_slow_square.R > future_slow_square.Rout 

