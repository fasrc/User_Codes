#!/bin/bash
#SBATCH -J mpitest           # job name
#SBATCH -o mpitest.out       # standard output file
#SBATCH -e mpitest.err       # standard error file
#SBATCH -p shared            # partition
#SBATCH -n 8                 # ntasks
#SBATCH -t 00:30:00          # time in HH:MM:SS
#SBATCH --mem-per-cpu=500    # memory in megabytes

# --- Load the required software modules., e.g., ---
module load intel/23.0.0-fasrc01 intelmpi/2021.8.0-fasrc01

# --- Run the executable ---
# with intelmpi, you need to ensure it uses pmi2 instead of pmix
srun -n $SLURM_NTASKS --mpi=pmi2 ./mpitest.x

