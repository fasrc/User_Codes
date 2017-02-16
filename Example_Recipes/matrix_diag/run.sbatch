#!/bin/bash
#SBATCH -J matrix_diag
#SBATCH -p serial_requeue
#SBATCH -o matrix_diag.out
#SBATCH -e matrix_diag.err
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 0-00:30
#SBATCH --mem=4000
source new-modules.sh
module load intel/15.0.0-fasrc01
module load intel-mkl/11.0.0.079-fasrc02
./matrix_diag.x
