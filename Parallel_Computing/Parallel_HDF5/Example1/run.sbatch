#!/bin/bash
#SBATCH -J parallel_hdf5           # job name
#SBATCH -o parallel_hdf5.out       # name of standard output file
#SBATCH -e parallel_hdf5.err       # name of standard error file
#SBATCH -p test                    # partition
#SBATCH -t 00:10:00                # time in HH:MM:SS
#SBATCH --nodes=2                  # number of nodes
#SBATCH --ntasks=8                 # number of total tasks
#SBATCH --mem=4000                 # memory

# Load required modules
module load intel/21.2.0-fasrc01 openmpi/4.1.1-fasrc01 hdf5/1.12.1-fasrc01

# Run program
srun -n $SLURM_NTASKS -N $SLURM_NNODES --mpi=pmix ./parallel_hdf5.x
