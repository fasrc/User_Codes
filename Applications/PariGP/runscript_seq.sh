#!/bin/bash
#SBATCH -J seq_pari           # job name
#SBATCH -o seq_pari_%j.out    # standard output file
#SBATCH -e seq_pari_%j.err    # standard error file
#SBATCH -p test               # partition
#SBATCH -n 1                  # ntasks
#SBATCH -N 1                  # number of nodes
#SBATCH -t 00:30:00           # time in HH:MM:SS
#SBATCH --mem 400             # memory in megabytes

# --- Load the required software modules., e.g., ---
module load pari/2.15.4-fasrc01

# --- Run the executable ---
gp < seq_pari.gp

