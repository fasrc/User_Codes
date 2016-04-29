#!/bin/bash
#SBATCH -J test_water
#SBATCH -o test_water.out
#SBATCH -e test_water.err
#SBATCH -p serial_requeue
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 30
#SBATCH --mem=40000

# Load required modules
source new-modules.sh
module load 

# Run Schrodinger
$SCHRODINGER/jaguar run -WAIT H2O.in
