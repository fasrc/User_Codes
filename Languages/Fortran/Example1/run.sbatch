#!/usr/bin/env bash
#SBATCH -J lanczos_job
#SBATCH -o lanczos_job.out
#SBATCH -e lanczos_job.err
#SBATCH -p test
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 0-00:30
#SBATCH --mem=4G

# Load required software modules
module load gcc/13.2.0-fasrc01

# Run the program
./lanczos.x <<EOF
100
5
100
EOF
