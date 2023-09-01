#!/bin/bash
#SBATCH -J par_pari           # job name
#SBATCH -o par_pari_%j.out    # standard output file
#SBATCH -e par_pari_%j.err    # standard error file
#SBATCH -p test               # partition
#SBATCH -n 16                 # ntasks
#SBATCH -N 2                  # number of nodes
#SBATCH -t 00:30:00           # time in HH:MM:SS
#SBATCH --mem-per-cpu=400     # memory in megabytes

# --- Load the required software modules., e.g., ---
module load gcc/12.2.0-fasrc01  openmpi/4.1.4-fasrc01 pari/2.15.4-fasrc02

# --- Run the executable ---
srun -n $SLURM_NTASKS --mpi=pmix gp < par_pari.gp
